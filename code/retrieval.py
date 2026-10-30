"""Offline, per-user message retrieval with a persistent local embedding cache."""

import hashlib
import os
import pickle
import re
from datetime import date, datetime, time, timezone
import math

import pandas as pd


DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
MESSAGES_PATH = os.path.join(DATASET_DIR, "messages.csv")
CACHE_PATH = os.path.join(DATASET_DIR, ".cache", "message_retrieval.pkl")
_INDEX = None
_EMBEDDER = None


def _supported_multilingual_model():
    """Ask FastEmbed for its supported models and select a multilingual model."""
    try:
        from fastembed import TextEmbedding

        models = TextEmbedding.list_supported_models()
        for model in models:
            name = str(model.get("model", model.get("model_name", "")))
            if "multilingual" in name.lower():
                return name, TextEmbedding(model_name=name)
    except Exception:
        # FastEmbed may be absent, its model registry unavailable, or the local
        # model not usable. In every such case retrieval stays strictly local.
        return None
    return None


def _normalised_date(value):
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _stdlib_tfidf(texts):
    """Small character-ngram TF-IDF fallback if sklearn is not installed."""
    documents = []
    document_frequency = {}
    for text in texts:
        padded = f"  {str(text).lower()}  "
        features = {padded[i:i + n] for n in range(2, 6) for i in range(max(0, len(padded) - n + 1))}
        documents.append(features)
        for feature in features:
            document_frequency[feature] = document_frequency.get(feature, 0) + 1
    vectors = []
    count = max(len(texts), 1)
    for features in documents:
        weights = {feature: (1 + __import__("math").log((count + 1) / (document_frequency[feature] + 1)))
                   for feature in features}
        norm = sum(value * value for value in weights.values()) ** 0.5 or 1.0
        vectors.append({feature: value / norm for feature, value in weights.items()})
    return ("stdlib_tfidf", document_frequency, vectors)


def _build_index():
    frame = pd.read_csv(MESSAGES_PATH, dtype=str, keep_default_na=False)
    frame.columns = frame.columns.str.strip()
    with open(MESSAGES_PATH, "rb") as messages_file:
        digest = hashlib.sha256(messages_file.read()).hexdigest()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as cache_file:
                cached = pickle.load(cache_file)
            if cached.get("digest") == digest:
                return cached
        except Exception:
            pass

    texts = frame.get("message_text", pd.Series(dtype=str)).fillna("").astype(str).tolist()
    embedder = None
    supported_model = _supported_multilingual_model()
    if supported_model is not None:
        model_name, embedder = supported_model
        try:
            vectors = [list(map(float, vector)) for vector in embedder.embed(texts)]
            index = {"digest": digest, "mode": "fastembed", "frame": frame, "vectors": vectors,
                     "model": model_name}
        except Exception:
            embedder = None
    if embedder is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), norm="l2")
            matrix = vectorizer.fit_transform(texts)
            index = {"digest": digest, "mode": "sklearn_tfidf", "frame": frame,
                     "vectorizer": vectorizer, "matrix": matrix}
        except ImportError:
            index = {"digest": digest, "mode": "stdlib_tfidf", "frame": frame,
                     "tfidf": _stdlib_tfidf(texts)}
    with open(CACHE_PATH, "wb") as cache_file:
        pickle.dump(index, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    return index


def _get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    return _INDEX


def retrieve_messages(user_id, as_of_date, query, k=5):
    """Return up to k messages for this user sent strictly before as_of_date."""
    index = _get_index()
    frame = index["frame"]
    cutoff = _normalised_date(as_of_date)
    if cutoff is None:
        return []
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    # Date-only cutoffs represent midnight at the start of that date.
    if isinstance(as_of_date, date) and not isinstance(as_of_date, datetime):
        cutoff = datetime.combine(as_of_date, time.min, tzinfo=timezone.utc)
    sent = pd.to_datetime(frame["sent_at"], errors="coerce", utc=True)
    mask = frame["user_id"].astype(str).eq(str(user_id)) & sent.lt(pd.Timestamp(cutoff))
    positions = [int(pos) for pos in range(len(frame)) if bool(mask.iloc[pos])]
    if not positions or int(k) <= 0:
        return []
    query = str(query or "")
    scores = []
    if index["mode"] == "fastembed":
        # FastEmbed yields unit-normalized vectors; cosine similarity is dot product.
        global _EMBEDDER
        from fastembed import TextEmbedding

        if _EMBEDDER is None:
            _EMBEDDER = TextEmbedding(model_name=index["model"])
        query_vector = list(map(float, next(_EMBEDDER.embed([query]))))
        scores = [sum(a * b for a, b in zip(query_vector, index["vectors"][pos])) for pos in positions]
    elif index["mode"] == "sklearn_tfidf":
        query_vector = index["vectorizer"].transform([query])
        scores = (index["matrix"][positions] @ query_vector.T).toarray().ravel().tolist()
    else:
        _, document_frequency, vectors = index["tfidf"]
        # Share corpus IDF (and the same char feature extraction) for query scores.
        padded = f"  {query.lower()}  "
        features = {padded[i:i + n] for n in range(2, 6) for i in range(max(0, len(padded) - n + 1))}
        weights = {feature: 1 + math.log((len(frame) + 1) / (document_frequency.get(feature, 0) + 1))
                   for feature in features if feature in document_frequency}
        norm = sum(value * value for value in weights.values()) ** 0.5 or 1.0
        query_map = {feature: value / norm for feature, value in weights.items()}
        scores = [sum(value * vectors[pos].get(feature, 0.0) for feature, value in query_map.items()) for pos in positions]
    ranked = sorted(zip(positions, scores), key=lambda pair: (-pair[1], pair[0]))[:int(k)]
    return [frame.iloc[pos].to_dict() for pos, _score in ranked]


def generic_request_query(request):
    """Build a non-ID query from request text plus fixed financial terms."""
    description = " ".join(str(request.get(key, "")) for key in ("request_type", "request_text"))
    description = re.sub(r"\brequest[_ -]?\d+\b", " ", description, flags=re.I)
    return f"{description} income payout refund salary pending"
