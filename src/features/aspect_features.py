from __future__ import annotations

from collections import defaultdict

SENTIMENTS = ["positive", "negative", "neutral"]


def compute_product_features(reviews_with_aspects, aspect_list):
    """Given a list of DeBERTa predictions for a product's reviews,
    compute per-aspect sentiment ratios.

    reviews_with_aspects: [{"aspects": [{"aspect": str, "sentiment": str, "confidence": float}]}]
    Returns: dict of feature_name -> float
    """
    # accumulate counts
    counts = defaultdict(lambda: defaultdict(int))
    for review in reviews_with_aspects:
        for a in review.get("aspects", []):
            asp = a["aspect"]
            sent = a["sentiment"]
            if asp in aspect_list and sent in SENTIMENTS:
                counts[asp][sent] += 1

    features = {}
    for asp in aspect_list:
        total = sum(counts[asp].values())
        if total == 0:
            # no reviews mention this aspect, use uniform default
            for s in SENTIMENTS:
                features[f"{asp}_{s}_ratio"] = 1.0 / len(SENTIMENTS)
            features[f"{asp}_review_count"] = 0
        else:
            for s in SENTIMENTS:
                features[f"{asp}_{s}_ratio"] = counts[asp][s] / total
            features[f"{asp}_review_count"] = total

    return features


def compute_user_preference(user_reviews_with_aspects, aspect_list, top_k=3):
    """From a user's review history, extract which aspects they care about most.

    Returns: {
        "top_aspects": [str, ...],  # top-K by mention frequency
        "{aspect}_avg_sentiment": float,  # avg sentiment score per aspect
    }
    """
    mention_count = defaultdict(int)
    sentiment_sum = defaultdict(float)
    sentiment_n = defaultdict(int)

    for review in user_reviews_with_aspects:
        for a in review.get("aspects", []):
            asp = a["aspect"]
            sent = a["sentiment"]
            if asp not in aspect_list:
                continue
            mention_count[asp] += 1
            # convert sentiment to numeric: positive=1, neutral=0.5, negative=0
            score = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}[sent]
            sentiment_sum[asp] += score
            sentiment_n[asp] += 1

    # rank by mention frequency
    sorted_aspects = sorted(mention_count.keys(), key=lambda x: -mention_count[x])
    top_aspects = sorted_aspects[:top_k]

    # pad if user has fewer than top_k aspects
    while len(top_aspects) < top_k:
        for asp in aspect_list:
            if asp not in top_aspects:
                top_aspects.append(asp)
            if len(top_aspects) == top_k:
                break

    result = {"top_aspects": top_aspects}
    for asp in aspect_list:
        if sentiment_n[asp] > 0:
            result[f"{asp}_avg_sentiment"] = sentiment_sum[asp] / sentiment_n[asp]
        else:
            result[f"{asp}_avg_sentiment"] = 0.5  # neutral default

    return result


def compute_cross_features(user_pref, product_features, top_k=3):
    """Cross user preference with product aspect features.

    For each of the user's top-K aspects, pull that aspect's pos/neg/neu ratio
    from the product features. Returns a fixed-length float vector.

    Output dimension: top_k * 3
    """
    top_aspects = user_pref["top_aspects"][:top_k]
    cross = []
    for asp in top_aspects:
        for s in SENTIMENTS:
            key = f"{asp}_{s}_ratio"
            cross.append(product_features.get(key, 1.0 / len(SENTIMENTS)))
    return cross
