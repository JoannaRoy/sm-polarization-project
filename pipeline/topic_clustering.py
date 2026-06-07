"""Stage 2: BERTopic clustering on extracted claims (text + topic sentence)."""

import json
import logging
from pathlib import Path

from bertopic import BERTopic
from hdbscan import HDBSCAN
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    TEST_DATA_PATH,
    TOPIC_EMBEDDING_MODEL,
    TOPIC_MODEL_PATH,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)
from db import ArgumentInstance, Field, Topic, connect
from db.writes import (
    refresh_post_primary_topics,
    reset_topic_dependent_state,
    write_topic_assignments_for_claims,
)
from pipeline.utils.llm import chat_completion

OUTLIER_TOPIC = -1
logger = logging.getLogger(__name__)
_topic_embedding_model = None

TOPIC_FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
    },
    "required": ["label"],
    "additionalProperties": False,
}

TOPIC_FRAME_PROMPT = """You are naming a topic cluster from BERTopic keywords.

Rules:
- Use 2-5 words for the label.
- Use Title Case.
- Do not include topic numbers, post numbers, underscores, hashtags, or quoted text.
- Name the subject, not a stance toward it.

Examples:
- BERTopic label: electric_vehicle_pricing_strategy -> {{"label": "Electric Vehicle Pricing"}}
- BERTopic label: cats_dogs_pet_choice -> {{"label": "Cats vs Dogs"}}

Return only a JSON object with the label field."""


def build_model():
    logger.debug(
        "Building BERTopic model with embedding model %s",
        TOPIC_EMBEDDING_MODEL,
    )
    return BERTopic(
        embedding_model=SentenceTransformer(TOPIC_EMBEDDING_MODEL),
        umap_model=UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            n_components=UMAP_N_COMPONENTS,
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=UMAP_RANDOM_STATE,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(stop_words="english"),
    )


def topic_embedder():
    global _topic_embedding_model
    if _topic_embedding_model is None:
        logger.info("Loading topic embedding model %s", TOPIC_EMBEDDING_MODEL)
        _topic_embedding_model = SentenceTransformer(TOPIC_EMBEDDING_MODEL)
    return _topic_embedding_model


def embed_topic_texts(texts):
    return np.asarray(
        topic_embedder().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def claim_embedding_text(claim_text, topic_sentence):
    """Combine claim text and topic sentence so the topic signal isn't lost."""
    if topic_sentence:
        return f"{claim_text} | {topic_sentence}"
    return claim_text


def build_topic_centroids(topics, embeddings):
    centroids = {}
    for topic_id in sorted(set(topics)):
        topic_embeddings = embeddings[np.asarray(topics) == topic_id]
        centroid = topic_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[topic_id] = centroid.astype(np.float32)
    return centroids


def train(claims, model_path=TOPIC_MODEL_PATH):
    """Fit BERTopic on a list of claim dicts and save the model."""
    texts = [
        claim_embedding_text(claim[Field.TEXT], claim[Field.TOPIC_SENTENCE])
        for claim in claims
    ]
    claim_ids = [claim[Field.ID] for claim in claims]
    logger.info("Training topic model on %d claims", len(claims))

    model = build_model()
    text_embeddings = embed_topic_texts(texts)
    topics, _ = model.fit_transform(texts, embeddings=text_embeddings)
    logger.info("Initial topic assignment produced %d topics", len(set(topics)))

    if OUTLIER_TOPIC in topics and any(t != OUTLIER_TOPIC for t in topics):
        logger.info("Reducing topic outliers")
        topics = model.reduce_outliers(
            texts,
            topics,
            strategy="embeddings",
            embeddings=text_embeddings,
        )
        model.update_topics(texts, topics=topics)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving topic model to %s", model_path)
    model.save(
        model_path,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=TOPIC_EMBEDDING_MODEL,
    )

    frames = generate_topic_frames(model, topics)
    centroids = build_topic_centroids(topics, text_embeddings)
    return _build_clusters(topics, claim_ids, frames, centroids)


def assign_topics_for_claims(claims):
    """Assign new claims to the nearest persisted topic centroid."""
    logger.info("Assigning %d claims with stored topic centroids", len(claims))
    texts = [
        claim_embedding_text(claim[Field.TEXT], claim[Field.TOPIC_SENTENCE])
        for claim in claims
    ]
    claim_ids = [claim[Field.ID] for claim in claims]
    topics = assign_topics_by_centroid(texts)
    logger.info("Assigned claims across %d topics", len(set(topics)))
    return _build_clusters(topics, claim_ids)


def load_topic_centroids():
    conn = connect()
    rows = (
        conn.query(Topic)
        .filter(Topic.centroid != None)  # noqa: E711
        .order_by(Topic.id)
        .all()
    )
    topic_ids = [
        int(topic.id) if topic.id.lstrip("-").isdigit() else topic.id for topic in rows
    ]
    centroids = [
        np.frombuffer(topic.centroid, dtype=np.float32).copy() for topic in rows
    ]
    conn.close()
    if not centroids:
        raise ValueError("No topic centroids found. Rerun topic clustering first.")
    return topic_ids, np.vstack(centroids)


def assign_topics_by_centroid(texts):
    topic_ids, centroids = load_topic_centroids()
    embeddings = embed_topic_texts(texts)
    scores = embeddings @ centroids.T
    return [topic_ids[index] for index in np.argmax(scores, axis=1)]


def top_topics_by_centroid(texts, k):
    """For each text, return up to ``k`` (topic_id, score) pairs sorted by
    descending cosine similarity to the persisted topic centroids."""
    topic_ids, centroids = load_topic_centroids()
    embeddings = embed_topic_texts(texts)
    scores = embeddings @ centroids.T
    limit = min(k, len(topic_ids))
    results = []
    for row in scores:
        order = np.argsort(-row)[:limit]
        results.append([(topic_ids[i], float(row[i])) for i in order])
    return results


def generate_topic_frame(bertopic_label):
    frame = chat_completion(
        messages=[
            {"role": "system", "content": TOPIC_FRAME_PROMPT},
            {"role": "user", "content": f"BERTopic label: {bertopic_label}"},
        ],
        schema=TOPIC_FRAME_SCHEMA,
    )
    frame[Field.LABEL] = frame[Field.LABEL].strip()
    return frame


def generate_topic_frames(model, topics):
    bertopic_labels = {
        row["Topic"]: row["Name"] for _, row in model.get_topic_info().iterrows()
    }
    frames = {}
    for topic_id in sorted(set(topics)):
        if topic_id == OUTLIER_TOPIC:
            frames[topic_id] = {Field.LABEL: "Outlier"}
            continue

        bertopic_label = bertopic_labels.get(topic_id, f"Topic {topic_id}")
        logger.info("Generating label for topic %s from %s", topic_id, bertopic_label)
        frames[topic_id] = generate_topic_frame(bertopic_label)
    return frames


def _build_clusters(topics, claim_ids, frames=None, centroids=None):
    frames = frames or {}
    centroids = centroids or {}
    clusters = {}
    for claim_id, topic_id in zip(claim_ids, topics):
        if topic_id not in clusters:
            frame = frames.get(topic_id, {})
            label = frame.get(Field.LABEL, f"Topic {topic_id}")
            clusters[topic_id] = {
                Field.LABEL: label,
                Field.CLAIM_IDS: [],
            }
            if topic_id in centroids:
                clusters[topic_id][Field.CENTROID] = centroids[topic_id]
        clusters[topic_id][Field.CLAIM_IDS].append(claim_id)
    logger.debug("Built cluster map: %s", clusters)
    return clusters


def load_posts(path=TEST_DATA_PATH):
    """Load Mastodon-style status fixtures."""
    logger.info("Loading posts from %s", path)
    data = json.loads(Path(path).read_text())
    return data["statuses"]


def load_unassigned_claims(conn):
    """Return [{id, text, topic_sentence}] for claims that need a topic."""
    rows = (
        conn.query(ArgumentInstance)
        .filter(ArgumentInstance.topic_id == None)  # noqa: E711
        .order_by(ArgumentInstance.id)
        .all()
    )
    return [
        {
            Field.ID: row.id,
            Field.TEXT: row.text,
            Field.TOPIC_SENTENCE: row.topic_sentence or "",
        }
        for row in rows
    ]


def run_batch():
    logger.info("Starting topic clustering batch")
    conn = connect()
    reset_topic_dependent_state(conn)
    claims = load_unassigned_claims(conn)
    if not claims:
        conn.close()
        logger.warning(
            "No claims to cluster. Run claim-extraction first to populate "
            "argument_instances."
        )
        return

    clusters = train(claims)
    assignment_count = sum(len(c[Field.CLAIM_IDS]) for c in clusters.values())
    logger.info(
        "Writing %d claim assignments across %d topics",
        assignment_count,
        len(clusters),
    )
    write_topic_assignments_for_claims(conn, clusters)
    refresh_post_primary_topics(conn)
    conn.close()
    logger.info("Finished topic clustering batch")
