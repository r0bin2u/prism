from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src.model.classifier import ABSAClassifier
from src.model.dataset import ABSADataset, collate_fn
from src.model.loss import MixedDistillationLoss

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            true_labels = labels.argmax(dim=-1)

            correct += (preds == true_labels).sum().item()
            total += len(preds)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(true_labels.cpu().tolist())

    acc = correct / max(total, 1)

    # macro f1
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    model.train()
    return {"accuracy": acc, "macro_f1": f1}


def run(config_path: Path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tc = config["training"]
    set_seed(tc["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tc["model_name"])

    # datasets
    filtered_dir = PROJECT_ROOT / "data" / "soft_labels" / "filtered"
    train_paths = [
        filtered_dir / "llm_soft_labels.jsonl",
        filtered_dir / "human_soft_labels.jsonl",
    ]
    val_path = PROJECT_ROOT / "data" / "splits" / "val.jsonl"

    train_dataset = ABSADataset(train_paths, tokenizer, max_length=tc["max_length"])
    # validation uses human labels only, convert to same format on the fly
    val_soft_path = filtered_dir / "human_soft_labels.jsonl"
    if val_soft_path.exists():
        val_dataset = ABSADataset([val_soft_path], tokenizer, max_length=tc["max_length"])
    else:
        val_dataset = ABSADataset([val_path], tokenizer, max_length=tc["max_length"])

    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    # count data composition
    human_count = sum(1 for s in train_dataset.samples if not s["is_soft_label"])
    llm_count = sum(1 for s in train_dataset.samples if s["is_soft_label"])
    logger.info("Train composition: %d human + %d LLM", human_count, llm_count)

    train_loader = DataLoader(
        train_dataset, batch_size=tc["batch_size"],
        shuffle=True, num_workers=4, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tc["batch_size"],
        shuffle=False, num_workers=4, collate_fn=collate_fn,
    )

    # model
    model = ABSAClassifier(tc["model_name"]).to(device)

    # optimizer: differential learning rate
    encoder_params = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": tc["learning_rate"]},
        {"params": classifier_params, "lr": tc["classifier_lr"]},
    ], weight_decay=0.01)

    # scheduler: linear warmup + cosine decay
    total_steps = len(train_loader) * tc["epochs"]
    warmup_steps = int(total_steps * tc["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # loss
    criterion = MixedDistillationLoss(alpha=tc["alpha"])

    # AMP
    use_amp = tc.get("amp", False) and device.type == "cuda"
    amp_dtype = getattr(torch, tc.get("amp_dtype", "bfloat16"), torch.bfloat16)

    # MLflow
    import mlflow
    mlflow.set_tracking_uri(str(PROJECT_ROOT / config["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    best_f1 = 0.0
    patience_counter = 0
    save_dir = PROJECT_ROOT / "models" / "best_model"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    sharpening_T = config["soft_label"]["sharpening_T"]

    with mlflow.start_run(run_name=f"deberta-alpha{tc['alpha']}-T{sharpening_T}"):
        mlflow.log_params({
            "model": tc["model_name"],
            "alpha": tc["alpha"],
            "lr": tc["learning_rate"],
            "classifier_lr": tc["classifier_lr"],
            "batch_size": tc["batch_size"],
            "epochs": tc["epochs"],
            "soft_label_temperature": sharpening_T,
            "human_train_count": human_count,
            "llm_train_count": llm_count,
        })

        global_step = 0
        for epoch in range(tc["epochs"]):
            model.train()
            epoch_losses = {"total": 0, "human": 0, "llm": 0}
            n_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                is_soft = batch["is_soft_label"].to(device)
                weights = batch["sample_weight"].to(device)

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(input_ids, attention_mask)
                        loss, loss_dict = criterion(logits, labels, is_soft, weights)
                else:
                    logits = model(input_ids, attention_mask)
                    loss, loss_dict = criterion(logits, labels, is_soft, weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_losses["total"] += loss_dict["total_loss"]
                epoch_losses["human"] += loss_dict["human_loss"]
                epoch_losses["llm"] += loss_dict["llm_loss"]
                n_batches += 1
                global_step += 1

                if global_step % 100 == 0:
                    mlflow.log_metrics({
                        "train/total_loss": loss_dict["total_loss"],
                        "train/human_loss": loss_dict["human_loss"],
                        "train/llm_loss": loss_dict["llm_loss"],
                    }, step=global_step)

            # epoch averages
            avg_total = epoch_losses["total"] / max(n_batches, 1)
            avg_human = epoch_losses["human"] / max(n_batches, 1)
            avg_llm = epoch_losses["llm"] / max(n_batches, 1)

            # validation
            val_metrics = evaluate(model, val_loader, device)
            val_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["accuracy"]

            logger.info(
                "Epoch %d/%d | loss=%.4f (human=%.4f, llm=%.4f) | val_f1=%.4f val_acc=%.4f",
                epoch + 1, tc["epochs"], avg_total, avg_human, avg_llm, val_f1, val_acc,
            )

            mlflow.log_metrics({
                "val/macro_f1": val_f1,
                "val/accuracy": val_acc,
                "train/epoch_total_loss": avg_total,
                "train/epoch_human_loss": avg_human,
                "train/epoch_llm_loss": avg_llm,
            }, step=epoch)

            # early stopping + save best
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0

                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / "model.pt")
                tokenizer.save_pretrained(save_dir)

                # save config alongside model
                with open(save_dir / "config.yaml", "w") as f:
                    yaml.dump(config, f)

                logger.info("Saved best model (f1=%.4f)", best_f1)
            else:
                patience_counter += 1
                if patience_counter >= tc["early_stopping_patience"]:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # update versions.yaml
        versions_path = PROJECT_ROOT / "configs" / "versions.yaml"
        if versions_path.exists():
            with open(versions_path) as f:
                versions = yaml.safe_load(f)
            versions["model"]["version"] = "v1.0"
            versions["model"]["changelog"] = f"DeBERTa-v3 alpha={tc['alpha']} best_f1={best_f1:.4f}"
            versions["model"]["checkpoint_path"] = str(save_dir)
            versions["api"]["model_version"] = "v1.0"
            with open(versions_path, "w") as f:
                yaml.dump(versions, f)

        mlflow.log_artifact(str(PROJECT_ROOT / "configs" / "config.yaml"))
        mlflow.log_artifact(str(versions_path))

    print(f"\nTraining complete. Best val macro_f1: {best_f1:.4f}")
    print(f"Model saved to: {save_dir}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
