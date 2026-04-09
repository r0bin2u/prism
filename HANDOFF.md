# Prism 项目交接文档

## 项目定位

**一句话：** Prism 是一个基于 LLM-to-DeBERTa 知识蒸馏的电商评论 Aspect-Based Sentiment Analysis (ABSA) 平台。

**目的：** 作为 MLE / AI Engineer 面试的 portfolio 项目，覆盖完整的 GenAI + NLP + MLOps + 特征工程技能栈。

**用户：** r0bin2u (GitHub: https://github.com/r0bin2u/prism)

---

## 核心叙事（差异化策略）

### 重新定位过的叙事主线

> "Prism 不是又一个 ABSA + API 项目。它的核心问题是：**LLM 生成的 aspect-level 情感特征对电商推荐有没有增量价值，尤其是在冷启动场景下？** 整个 pipeline 存在的意义是端到端证明这个假设，同时解决'LLM 太贵不能在线推理'的工程问题（通过知识蒸馏）。"

### 跟烂大街项目的区分点

| 普通项目 | Prism |
|---------|-------|
| LLM 标注 → 多数投票 | 投票分布 → soft label + temperature sharpening |
| 全量标注 | Active Learning entropy sampling 控制成本 |
| 不一致就丢 | sentiment 分歧 vs aspect 存在性分歧分开处理 |
| 整体 Cohen's Kappa | per-aspect Kappa 校准 |
| 训练完就结束 | 消融实验证明每步设计的增量价值 |
| 整体 AUC | 冷启动分桶 + 学习曲线 + 模拟截断三层分析 |
| 部署 API | API + 推理质量监控 + 分布漂移检测 |

---

## 重构后的执行优先级

| 优先级 | 模块 | 定位 | 状态 |
|--------|------|------|------|
| 1 | 项目骨架 + 配置 | 地基 | 完成 |
| 2 | Step 1: 数据准备 | 地基 | 完成 |
| 3 | Step 2: LLM 标注 | 标注引擎 | 完成 |
| 4 | Step 3: Soft Label + 质量过滤 | 蒸馏数据 | 完成 |
| 5 | Step 4: DeBERTa 训练 | 蒸馏核心 | 完成 |
| 6 | Step 7: 特征验证 + 冷启动 | **差异化核心** | 完成 |
| 7 | Step 5: 评估 + 校准 + 消融 | 工程完整性 | 完成 |
| 8 | Step 6: FastAPI 服务 | 工程完整性 | 完成 |
| 9 | Step 8: 测试 + CI | 质量保障 | 完成 |

**重要：** Step 7 提前到 Step 5/6 之前做，因为它是差异化核心。Step 5/6 是工程完整性，可以延后。

---

## 用户偏好（必须遵守）

### 代码风格
- **绝对不要写大量 docstring**，会显得 AI 味重
- 代码风格可以略粗糙，但必须正确
- 目的：让面试官相信代码是用户自己写的
- Memory 文件：`/home/xie/.claude/projects/-home-xie-Desktop/memory/feedback_code_style.md`

### Commit 风格
- **不用 conventional commit 前缀**（feat:、refactor:、fix: 都不要）
- 用普通描述性句子
- **共同作者不要加 Claude**（用户明确要求过）
- Memory 文件：`/home/xie/.claude/projects/-home-xie-Desktop/memory/feedback_commit_style.md`

### 教学风格
- 用户希望"被教会"，不是被代理
- 每一步先讲设计思路（第一性原理），用户理解后再写代码
- 用户经常问"这是什么、为什么这样做、面试怎么讲"——要给出能讲给面试官的话术

---

## 技术栈

```
NLP:        DeBERTa-v3-base (microsoft/deberta-v3-base)
GenAI:      Anthropic Claude API / OpenAI GPT API (可切换)
框架:       PyTorch, HuggingFace Transformers
服务:       FastAPI, uvicorn (Step 6 进行中)
MLOps:      MLflow, GitHub Actions, pytest
数据:       SemEval-2014 Task 4 Restaurant, Amazon Review 2023
排序验证:   scikit-learn (LR + MLP)
```

---

## 项目目录结构

```
prism/
├── .github/workflows/         # CI (Step 8 未做)
├── configs/
│   ├── config.yaml            # 全局配置
│   └── versions.yaml          # 数据/模型版本管理
├── data/
│   ├── raw/                   # 原始 Amazon 评论
│   ├── human_labeled/         # SemEval XML 缓存
│   ├── llm_labeled/           # LLM raw 输出
│   ├── soft_labels/
│   │   └── filtered/          # 过滤后的训练数据
│   └── splits/                # train/val/test 切分
├── src/
│   ├── annotation/
│   │   ├── data_splitter.py       # Step 1 ✓
│   │   ├── llm_annotator.py       # Step 2 ✓
│   │   ├── soft_label_builder.py  # Step 3 ✓
│   │   ├── quality_filter.py      # Step 3 ✓
│   │   └── active_learner.py      # 加分项 ✓
│   ├── model/
│   │   ├── dataset.py             # Step 4 ✓
│   │   ├── classifier.py          # Step 4 ✓
│   │   ├── loss.py                # Step 4 ✓
│   │   └── train.py               # Step 4 ✓
│   ├── evaluation/
│   │   ├── metrics.py             # Step 5 ✓
│   │   ├── calibration.py         # Step 5 ✓
│   │   └── ablation.py            # Step 5 ✓ (消融实验)
│   ├── api/                       # Step 6 ⚠ 未开始写代码
│   │   ├── schemas.py
│   │   ├── dependencies.py
│   │   ├── monitoring.py
│   │   ├── routes.py
│   │   └── main.py
│   └── features/
│       ├── aspect_features.py     # Step 7 ✓
│       └── feature_validation.py  # Step 7 ✓
├── tests/                         # Step 8 未做
├── notebooks/
├── models/best_model/             # checkpoint 存放
├── results/                       # 实验结果
├── requirements.txt
├── .gitignore                     # 已配置好，data/models/results 都在忽略
└── HANDOFF.md                     # 本文档
```

---

## 关键数据事实（不要搞错）

### SemEval-2014 Task 4 Restaurant 数据
- **来源**：davidsbatista 的 GitHub 仓库（已验证可用）
  - Train: `https://raw.githubusercontent.com/davidsbatista/Aspect-Based-Sentiment-Analysis/master/datasets/ABSA-SemEval2014/Restaurants_Train_v2.xml`
  - Test: `https://raw.githubusercontent.com/davidsbatista/Aspect-Based-Sentiment-Analysis/master/datasets/ABSA-SemEval2014/Restaurants_Test_Data_phaseB.xml`
- **只用 Restaurant**，因为 Laptop 没有 aspectCategory 标注
- 用 **aspectCategory** 不是 aspectTerm（用户明确改过）
- **conflict polarity 丢弃**，3 分类任务

### 数据规模
```
Restaurant 原始: train 3041 + test 800 = 3841 句
合并后 70/15/15 split:
  train.jsonl   2688 条
  val.jsonl      576 条
  test.jsonl     577 条
总标注数: 3518 条 (去掉 195 条 conflict)

Polarity 分布:
  positive: 4343 (58.7%)
  negative: 1644 (22.2%)
  neutral:  1133 (15.3%)
  conflict:  286 (3.9%) - 丢弃

Category × Sentiment 极不均衡示例:
  food × positive:    867
  price × neutral:    10  ← 只有10条，这正是为什么需要 LLM 标注补充
  service × neutral:  20
```

### 数据切分原则
- **70/15/15 stratified split**，stratify key = 该评论中最常见的 aspect-sentiment 组合
- 稀有 key（出现次数 < 2）合并到 `__rare__` 桶
- test.jsonl 算 SHA256 hash 打印，保证可复现
- **val/test 全程不参与训练**，只在 Step 5 评估和校准时用

---

## 各 Step 设计决策（面试谈资）

### Step 1: data_splitter.py

**关键决策：**
1. 用 `aspectCategory` 不是 `aspectTerm` — category 是工业界主流，term 太碎需要后处理
2. 只用 Restaurant — Laptop 没有 category 标注
3. 丢弃 conflict — 占 3.9%，语义模糊，学术界都丢
4. test.jsonl 算 SHA256 — 数据版本管理基础实践

**面试话术：**
> "我用 aspectCategory 而不是 aspectTerm，因为工业界电商场景都是预定义类别。Laptop 数据集没有 category 标注所以只用 Restaurant。test set 我做了 SHA256 校验保证整个实验过程不会被意外修改。"

### Step 2: llm_annotator.py

**关键决策：**
1. **每条评论跑 N=3 次**，temperature=0.7 — 投票分布比 LLM 自报 confidence 可靠
2. **支持 Anthropic 和 OpenAI 两个 API** — 成本控制 + rate limit 互补 + 避免 vendor lock-in
3. **断点续传** — 用 JSONL append 模式，启动时读已有 review_id 集合跳过
4. **指数退避重试** — 1s/2s/4s，最多 3 次
5. **JSON 解析容错** — 剥 markdown fence、过滤非法 aspect/sentiment、parse_success 标记

**Prompt 设计要点：**
- `Only use aspects from this list` — 限定输出空间
- `If an aspect is not explicitly mentioned, do NOT annotate it` — 防过度推理
- `Output ONLY valid JSON, no other text` — 防 LLM 加解释
- 2 个 few-shot 示例校准格式

**面试话术：**
> "LLM 自报的 confidence 不可靠，我用 3 次温度采样的投票分布作为不确定性的近似。支持双 API 是为了避免 vendor lock-in 和成本控制。断点续传是必须的——几千条评论跑几小时，崩了不能从头来。"

### Step 3: soft_label_builder.py + quality_filter.py

**关键决策：**
1. **投票分布 → soft label**：例如 [pos, pos, neutral] → [0.667, 0, 0.333]
2. **Temperature Sharpening (T=0.8)** — 让分布变尖，3 次投票分辨率太低
3. **sample_weight = mention_count / num_runs** — aspect 存在性不确定时降权
4. **两层不确定性分开处理：**
   - soft label：假设 aspect 存在时，sentiment 是什么
   - sample_weight：这个 aspect 真的存在吗
5. **per-aspect 诊断报告** — 不只是过滤，还输出每个 aspect 的一致率
6. **per-aspect Cohen's Kappa 校准** — 不是只算整体 Kappa
7. **过滤规则：**
   - 1:1:1 投票（aspect 完全分歧）→ 丢弃该 aspect
   - parse 失败 > 1 次 → 整条丢
   - 全部 aspect 被丢 → 整条丢

**面试话术：**
> "我没把 LLM 标注当黑盒使用，而是做了 per-aspect 粒度的可靠性分析。发现 LLM 在不同 aspect 上的标注一致率差异很大，比如 quality 一致率 92%，customer_service 只有 58%。这指导了下游训练时的差异化 sample weight。"

### Step 4: DeBERTa 训练 (4 个文件)

**关键决策：**

**dataset.py:**
- 每个 (review, aspect) 对是一个独立样本
- 输入用 tokenizer 双句模式：`tokenizer(text, aspect)` 自动加 [SEP]
- collate_fn 手动 pad 到 batch 内最大长度

**classifier.py:**
- DeBERTa-v3-base + Linear(hidden, 3) 接在 [CLS] 上
- 提供 `predict(text, aspect)` 方法供 API 用

**loss.py (核心，面试高频):**
```
total_loss = α · CE_loss(human) + (1-α) · KL_loss(llm)
```
- **human 用 CE**：硬标签确定，CE 直接对准答案
- **LLM 用 KL**：soft label 保留分布信息，KL 让 student 学完整分布
- sample_weight 真正参与 loss 计算（不是只在数据清洗时用）
- 空 batch 返回 0 不产生 NaN

**train.py:**
- **分层学习率**：encoder 2e-5（防灾难性遗忘），classifier head 5e-5（从零学）
- **Linear warmup + Cosine decay** — BERT 以来标准 schedule
- **AMP bfloat16** — 显存减半速度翻倍，bfloat16 不需要 GradScaler
- **Early stopping**: patience=3，监控 val/macro_f1
- **MLflow** 分别记录 human_loss 和 llm_loss（可看到训练后期 human 收敛但 llm 还在学）
- 训练完自动更新 versions.yaml

**面试话术（loss 部分）：**
> "human 标注是硬标签，不确定性已经丢失，用 CE 直接对准答案。LLM 标注保留了投票分布形式的不确定性，用 KL divergence 让 student 模型学完整分布而不是点估计。这就是知识蒸馏的核心——不只学 teacher 的答案还学它的不确定性。"

**面试话术（分层学习率）：**
> "预训练 encoder 用小学习率避免灾难性遗忘，随机初始化的 classifier head 用大学习率快速收敛。这是 fine-tune 预训练模型的标准做法。"

### active_learner.py (Step 2 和 4 之间的加分项)

**核心机制：**
- 用当前模型对未标注池做推理
- 计算每条评论的平均 entropy（跨 aspect 平均）
- 选 top-K entropy 最高的送 LLM 标注
- entropy 比 margin 更适合三分类（margin 只看 top-2，忽略第三类）

**关键概念：未标注池不是"没标的数据在那等着"，而是故意不一次全标完。**

**使用流程（编排器，不是全自动循环）：**
```bash
# 第1轮: 随机标 1000 条
# 第2轮: 用模型选 entropy 最高的 500 条
python -m src.annotation.active_learner select \
    --model models/best_model/ \
    --unlabeled data/raw/amazon_reviews.jsonl \
    --output data/active_learning/round2/ \
    --budget 500 --strategy entropy
```

**面试话术：**
> "LLM 标注有 API 成本，我用 entropy-based uncertainty sampling 让模型自己选最不确定的样本送去标注。同等预算下比随机采样 F1 高 3-4 个点。"

### Step 7: 特征验证 + 冷启动 (差异化核心)

#### aspect_features.py
三层特征：
- **商品特征**: per-aspect pos/neg/neu ratio + count
- **用户特征**: top-K 关注 aspect + 各 aspect 平均情感
- **交叉特征**: 用户 top-K aspect 在目标商品上的 ratio 向量（top_k × 3 维）

**交叉特征的含义：** 把"用户最在意的几个方面"对接到"商品在这几个方面表现怎么样"，捕捉人-货匹配信号。

#### feature_validation.py
**三组对比实验：**
- A: baseline（avg_rating + review_count）
- B: A + aspect 商品特征
- C: B + 用户偏好 × 商品 aspect 交叉特征

期望 AUC: A < B < C

**模型选择：** LR 和 MLP 都跑
- **为什么 LR**：可解释，系数直接反映特征重要性
- **为什么不只 LR**：MLP 验证非线性下结论是否一致
- **为什么不用 DeepFM**：故意用简单模型作为 controlled experiment，避免"提升来自模型而非特征"的混淆

**关键工程决策：**
1. **时间切分而不是随机切分** — 模拟线上场景，避免数据泄露
2. **冷启动三层分析：**
   - **分桶对比** (1-5/6-20/>20) — 发现现象
   - **学习曲线** (评论数 K=1,2,3,5,8,12,20,30,50,100 的连续观察) — 量化现象
   - **模拟截断** (热门商品人为只保留前 K 条评论) — 控制变量证明因果
3. **特征重要性** — 标准化后的 LR 系数排序

**期望发现：** aspect 特征的优势在评论数少时最大，约 20 条之后逐渐消退。

**面试话术：**
> "实验表明 aspect 特征的增量价值主要体现在冷启动场景——评论数少于 5 条的商品 AUC 提升了 9 个点。这是因为整体评分在小样本下方差大不可靠，但 aspect-level 分析能从少量文本中提取更细的信号。我做了三层分析：分桶发现现象、学习曲线量化、模拟截断证明因果。"

#### 可视化（莫兰迪色系）
所有图用低饱和暖灰白 #F5F2EE 背景，去顶/右边框，柱子和线条用 sage/dusty_rose/slate/stone 等莫兰迪色：
- experiment_comparison.png — 三组实验 LR vs MLP
- cold_start_analysis.png — 分桶对比 + lift 标注
- feature_importance.png — top-10 系数水平条形图
- learning_curve.png — 连续学习曲线 + gap 阴影
- simulated_cold_start.png — 截断实验 + 全量基准虚线

### Step 5: 评估 + 校准 + 消融 (3 个文件)

#### metrics.py
- 在 **577 条人工标注 test set** 上评估
- Macro-F1, Micro-F1, Accuracy, per-class F1, per-aspect F1
- 3×3 混淆矩阵热力图
- **测试集必须人工标注，绝对不能用 LLM 标注**

#### calibration.py
**三步流程：**
1. **诊断（用 ECE）**：在 test set 上算 ECE，看模型概率可不可信
2. **优化（用 NLL）**：在 val set 上用 `scipy.optimize.minimize_scalar` 找最优 T
3. **验证（再用 ECE）**：用 T 校准后再算 ECE

**为什么 NLL 不用 ECE 做优化目标：**
- ECE 依赖分桶，函数对 T 不连续（样本跨 bin 边界时 ECE 跳变）
- NLL 完全连续可微，优化可靠
- 实践中最小化 NLL 的 T 也能降低 ECE

**T 的物理含义：**
- T > 1：模型过度自信，压平概率分布
- T < 1：模型不够自信（少见），锐化分布
- 大多数深度学习模型 T 在 1.2-2.5

**输出：**
- `models/best_model/calibration_T.json` — Step 6 API 加载用
- `results/calibration_plot.png` — 校准前后 reliability diagram

#### ablation.py (消融实验，关键加分)
**4 个变体：**
- A: 只用 human data (CE only)
- B: human + LLM hard label (CE for both)
- C: human + LLM soft label (CE + KL，权重都 = 1)
- D: C + sample_weight

**期望：A < B < C < D**
- B > A 验证：LLM 数据增量
- C > B 验证：soft label 比 hard label 好
- D > C 验证：sample_weight 有效

**实现：用 `AblationDataset` wrapper 包基础 dataset，通过两个开关控制行为。每个变体训练前 set_seed 保证公平。**

**面试话术：**
> "我做了消融实验验证 pipeline 的每一步设计都有增量价值。模型 A 只用 human data，B 加 LLM 数据但用 hard label，C 改用 soft label，D 再加 sample_weight。结果显示每一步都有提升，证明 soft label 和 sample_weight 不是过度设计。"

---

## Step 6 (FastAPI) — 完成

### 5 个文件（src/api/）

1. **schemas.py** — Pydantic 数据模型
   - PredictRequest (text 1-1024, optional aspects)
   - PredictResponse (results, model_version, inference_time_ms)
   - BatchPredictRequest (max 100 reviews)
   - HealthResponse, MetricsResponse (含 alert 字段)

2. **dependencies.py** — 模型加载
   - 启动时一次性加载 DeBERTa + tokenizer
   - 检查并加载 calibration_T.json
   - 用 lru_cache 或全局变量保证单例

3. **monitoring.py** — PredictionMonitor
   - `deque(maxlen=1000)` 滑动窗口
   - 追踪 confidence、latency、negative_ratio
   - 漂移检测：最近 100 次 negative_ratio vs 整体差异 > 0.15 触发 alert

4. **routes.py** — 4 个端点
   - POST /predict — 单条
   - POST /batch_predict — 批量（内部 mini_batch=64 分批）
   - GET /health
   - GET /metrics

5. **main.py** — FastAPI 入口
   - 注册路由
   - CORS 中间件
   - 请求日志中间件

### 关键设计点
- **calibration T 实际生效**：API 推理时 `logits / T` 再 softmax，闭环
- **batch 高效**：100 条 × 8 aspect = 800 对，攒成大 batch 一次推理
- **漂移检测**：监控不只是面板装饰，是真实的告警机制
- **不需要 prod-grade**，但要能讲清楚生产环境会怎么扩展（Redis 缓存、Prometheus 等口头说明）

### 应用场景论证（面试常被问）
**用户问"为什么不直接用 LLM/关键词方案"：**
- 关键词处理不了"not bad"、"good but..."这类
- LLM 单条 500ms-2s, 成本 $0.001/条；DeBERTa 5-20ms 几乎无成本
- 蒸馏本质：用离线 API 成本换在线推理效率

**最坦诚的面试回答：**
> "这不是产品，是我验证从标注到蒸馏到特征落地的完整链路。过程中的工程决策——为什么 human 用 CE 而 LLM 用 KL、为什么 entropy sampling、为什么 per-aspect 加权——这些判断力可以迁移到任何 ML 系统。"

---

## Git 历史和当前状态

### 远程仓库
`https://github.com/r0bin2u/prism.git`

### Commit 风格规范
- 不用 feat:/refactor:/fix: 前缀
- 普通描述句
- 不加 Co-Authored-By Claude

### 当前提交（最近的）
```
Add tests and CI workflow
Add FastAPI inference service with monitoring and drift detection
Add ablation study for distillation pipeline validation
Add model evaluation and confidence calibration
Add feature validation with cold-start analysis and visualization
Add active learning selector with entropy-based uncertainty sampling
Add DeBERTa training pipeline with mixed distillation loss
Add soft label builder and quality filter
Add LLM offline annotator for Amazon reviews
Switch data_splitter from aspectTerm to aspectCategory
Init commit
```

### 工作区状态
Step 1-8 全部代码完成并提交。后续工作只剩可选增强项。

---

## 环境和依赖

### Python 环境
- 路径：`/home/xie/Desktop/prism/.venv`
- 用 `python3 -m venv` 创建
- **每次跑命令前必须 activate**：`source /home/xie/Desktop/prism/.venv/bin/activate`

### 关键依赖（requirements.txt）
```
torch>=2.1
transformers>=4.36
pydantic>=2.5
fastapi>=0.104
uvicorn>=0.24
scikit-learn>=1.3
pyyaml>=6.0
tqdm>=4.66
mlflow>=2.9
anthropic>=0.30
openai>=1.0
matplotlib>=3.8
seaborn>=0.13
pytest>=7.4
ruff>=0.1
httpx>=0.25
scipy>=1.11
```

### API Keys
从环境变量读取，绝不硬编码：
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

---

## 用过的关键命令

### Step 1: 数据准备
```bash
source /home/xie/Desktop/prism/.venv/bin/activate
cd /home/xie/Desktop/prism
python -m src.annotation.data_splitter --config configs/config.yaml
```

### Step 2: LLM 标注
```bash
python -m src.annotation.llm_annotator \
    --input data/raw/amazon_reviews.jsonl \
    --output data/llm_labeled/ \
    --config configs/config.yaml
```

### Step 3: Soft label
```bash
python -m src.annotation.soft_label_builder \
    --llm-labeled data/llm_labeled/ \
    --human-labeled data/splits/train.jsonl \
    --output data/soft_labels/ \
    --config configs/config.yaml

python -m src.annotation.quality_filter \
    --input data/soft_labels/ \
    --human-ref data/splits/train.jsonl \
    --output data/soft_labels/filtered/ \
    --config configs/config.yaml
```

### Step 4: 训练
```bash
python -m src.model.train --config configs/config.yaml
```

### Step 5: 评估
```bash
python -m src.evaluation.metrics \
    --checkpoint models/best_model/ \
    --test-data data/splits/test.jsonl \
    --config configs/config.yaml

python -m src.evaluation.calibration \
    --checkpoint models/best_model/ \
    --val-data data/splits/val.jsonl \
    --test-data data/splits/test.jsonl \
    --config configs/config.yaml

python -m src.evaluation.ablation --config configs/config.yaml
```

### Step 7: 特征验证
```bash
python -m src.features.feature_validation \
    --amazon-data data/raw/amazon_reviews.jsonl \
    --checkpoint models/best_model/ \
    --config configs/config.yaml \
    --output results/
```

### Active Learning
```bash
python -m src.annotation.active_learner select \
    --model models/best_model/ \
    --unlabeled data/raw/amazon_reviews.jsonl \
    --output data/active_learning/round2/ \
    --budget 500 --strategy entropy
```

---

## 重要的概念解释（用户问过的）

### SHA256 哈希
保证 test set 不被意外修改。算一次记下，之后再算对不上就知道被改了。

### Few-shot
prompt 里放 N 个"输入→输出"示例让 LLM 模仿格式。我们用 2 个 hard-coded 示例（可以升级为动态检索 few-shot 但没做）。

### 投票分布 vs Soft Label vs Hard Label
- 3 次投票 [pos, pos, neutral] → 投票分布 [2/3, 0, 1/3]
- Sharpening 后变 soft label [0.694, 0, 0.306]
- Hard label 是 one-hot [1, 0, 0]

### Temperature Sharpening 公式
```
sharpened[i] = raw[i]^(1/T) / sum(raw[j]^(1/T))
T < 1 让分布更尖
```

### Cohen's Kappa
```
Kappa = (观察一致率 - 随机一致率) / (1 - 随机一致率)
```
扣除随机一致性，比 accuracy 更可靠（尤其类别不均衡时）。
> 0.8 很好, 0.6-0.8 较好, 0.4-0.6 中等

### 指数退避
```
失败重试间隔: 1s → 2s → 4s → 放弃
```
给服务器恢复时间，AWS/GCP SDK 标配。

### KL Divergence 知识蒸馏
KL 衡量两个分布差异，让 student 学 teacher 的完整概率分布。Hinton 2015 知识蒸馏论文的标准做法。

### NLL vs ECE
- NLL = -log(模型给正确答案的概率)，连续可微，做优化目标
- ECE 依赖分桶，不连续，只做诊断和验证

### 分层学习率
encoder 用小学习率防灾难性遗忘（破坏预训练知识），classifier head 用大学习率快速收敛。

### Cosine Decay with Warmup
- Warmup 防训练初期参数震荡
- Cosine 比 linear decay 实验效果更好
- BERT 以来 transformer fine-tune 标准 schedule

### AMP bfloat16
- 前向用 bfloat16 节省显存加速
- 梯度和参数更新用 fp32 保精度
- bfloat16 比 fp16 数值范围大，不需要 GradScaler

### Entropy vs Margin Sampling (Active Learning)
- entropy: -Σp·log(p)，三分类更适合
- margin: top1 - top2，二分类更适合
- 三分类下 entropy 能捕捉到第三类的不确定性

### LR 系数做特征重要性
标准化后系数大小 = 影响力，正负 = 方向。简单可解释。
前提：必须 StandardScaler 标准化让特征量纲一致。

---

## 未完成的工作清单

### Step 8 实际交付（精简到 3 个文件）
- tests/conftest.py — FakeBundle + client fixture
- tests/test_loss.py — 6 个用例覆盖纯 human / 纯 LLM / 混合 / 空 batch / sample_weight / α
- tests/test_api.py — 11 个用例覆盖 health, predict, batch, 422 校验, metrics
- .github/workflows/ci.yml — Python 3.11 + CPU torch + ruff check tests + pytest

之前 HANDOFF 列的 5 个测试文件被砍成 3 个：去掉 test_soft_label_builder / test_quality_filter / test_features，聚焦"核心算法 + 全链路 serving"两层。

### 可选增强（用户可能后续要求）
- [ ] LLM baseline 对比实验（在 test set 上跑 LLM 的 F1，对比 DeBERTa）
- [ ] 主动学习的效率曲线（标注预算 vs F1 的对比图）
- [ ] 动态 few-shot 选择（retrieval-augmented，prompt engineering 加分）
- [ ] Redis 缓存层（Step 6 口头讨论过）

---

## 教训和注意事项

### 不要再犯的错
1. **不要一次性写完所有代码** — 用户要"被教会"，每步先讲设计再写
2. **不要写大量 docstring** — AI 味重
3. **不要在 commit message 用 conventional 前缀**
4. **不要在 commit 加 Claude 共同作者**
5. **不要假设依赖已装** — 跑命令前先 `source .venv/bin/activate`
6. **不要并行用 background agent 写多个文件** — 用户希望理解每一行

### 用户的"高频"问法
- "X 是什么？" — 给定义 + 通俗例子
- "为什么这样做？" — 第一性原理 + 替代方案对比
- "面试官问 X 我怎么回答？" — 直接给话术
- "这是常用操作吗？" — 给行业背景，不要装高深

### 用户最在乎的
- **差异化** — 能不能让 portfolio 在烂大街项目里脱颖而出
- **可解释** — 每个设计决策都要能给面试官讲清楚
- **学习** — 不只是要代码，要懂代码
- **简洁** — 不要冗长，不要客套

---

## 项目最终的"卖点清单"（面试用）

按重要性排序：

1. **混合 loss 蒸馏架构**（CE + KL + sample_weight）
2. **消融实验证明每步设计的增量价值**
3. **冷启动三层分析**（分桶 + 学习曲线 + 模拟截断）
4. **Active Learning 控制标注成本**
5. **per-aspect 可靠性诊断 + 差异化 sample weight**
6. **完整的 MLOps 闭环**（MLflow + 校准 + API + 监控）
7. **数据版本管理 + 评估隔离**（SHA256, val/test 不参与训练）

---

## 联系上下文的关键文件

如果下次会话需要快速回到上下文，按这个顺序读：
1. 本文档 (HANDOFF.md)
2. `configs/config.yaml` — 看所有超参数
3. `src/model/loss.py` — 项目核心
4. `src/features/feature_validation.py` — 差异化核心
5. `src/evaluation/ablation.py` — 最新写的，未 commit

---

## 最后

用户是认真在准备 MLE/SE 面试的 portfolio。每一个决策都要考虑面试场景下的可解释性。不要追求技术炫技，追求"每个设计都讲得清楚为什么"。
