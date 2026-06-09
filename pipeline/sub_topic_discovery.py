"""Stage 3: cluster claims within each topic into sub-topics, then have the
LLM name + frame each sub-cluster from a sample of its actual claims.

Sub-topics whose claims do not share a real axis of disagreement get
``polarity_target = NULL`` (descriptive) and skip polarity assignment +
slate generation downstream.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import monotonic
import uuid

from hdbscan import HDBSCAN
import numpy as np
from umap import UMAP

from config import (
    LLM_CONCURRENCY,
    SUBTOPIC_HDBSCAN_MIN_CLUSTER_SIZE,
    SUBTOPIC_HDBSCAN_MIN_SAMPLES,
    SUBTOPIC_LLM_SAMPLE_SIZE,
    SUBTOPIC_UMAP_METRIC,
    SUBTOPIC_UMAP_MIN_DIST,
    SUBTOPIC_UMAP_N_COMPONENTS,
    SUBTOPIC_UMAP_N_NEIGHBORS,
    SUBTOPIC_UMAP_RANDOM_STATE,
)
from db import ArgumentInstance, Field, connect
from db.reads import get_topics
from db.writes import (
    assign_instance_to_sub_topic,
    create_sub_topic,
    reset_sub_topic_state,
    update_sub_topic_centroids,
)
from tqdm import tqdm

from pipeline.topic_clustering import OUTLIER_TOPIC
from pipeline.utils.embeddings import embed_preference_texts
from pipeline.utils.llm import chat_completion

logger = logging.getLogger(__name__)

SUB_TOPIC_FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "polarity_target": {"type": ["string", "null"]},
    },
    "required": ["label", "polarity_target"],
    "additionalProperties": False,
}

SUB_TOPIC_FRAME_PROMPT = """You are analyzing one sub-cluster of claims that all came from the broader topic "{topic_label}".

Below is a sample of representative claims from this sub-cluster:
{claim_list}

Your task is in two steps.

Step 1 - Identify the *dominant proposition* of this sub-cluster.
Skim the sample and ask: "What is the single proposition that the MOST claims in this sample take a position on?" Take-a-position means each claim would either agree or disagree with the proposition - it does not matter which, and the cluster does NOT need to be split. A cluster where 90% of claims agree (and few or none disagree) is perfectly valid, as is the opposite. What matters is that claims are *on-axis* to the proposition - i.e. they engage with it rather than talk about something else.

The proposition has to fit the majority of these claims, not a minority subgroup that happens to be the most rhetorically charged. If two unrelated themes appear (e.g. half the claims are about vegan recipes, half are about meat-industry studies), pick the larger group's proposition - do NOT pick the smaller, more controversial theme.

Step 2 - Decide whether to assign that proposition as ``polarity_target``.
Assign it if AT LEAST HALF of the sample claims would clearly agree or disagree with it (either way is fine, unanimous is fine). Return ``polarity_target: null`` only if most of the claims are *off-axis* from any single proposition - descriptive content (recipes, news, announcements), or a mix of disconnected sub-themes where no proposition fits most of them.

Output formats:

If at least half of the sample takes a position on a dominant proposition:
  {{"label": "2-5 word title-case label", "polarity_target": "complete proposition stated affirmatively"}}

If most claims are off-axis from any single proposition:
  {{"label": "2-5 word title-case label", "polarity_target": null}}

Rules:
- The label must be 2-5 words in Title Case, with no quotation marks or hashtags. It describes what the cluster is about (e.g. "Vegan Alternatives", "Climate Policy Debate"), regardless of whether a polarity_target is set.
- A polarity_target must be a single affirmative declarative sentence one could agree or disagree with - not a question, not a topic name.
- An imbalanced cluster (mostly agreeing or mostly disagreeing on the polarity_target) is a valid and honest outcome. Do not return null just because the cluster is one-sided.
- Only return null when the cluster is genuinely off-axis or descriptive - no shared proposition fits the majority.

Return only the JSON object."""


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def reassign_outliers(labels, embeddings):
    """Reassign HDBSCAN outliers to their nearest non-outlier centroid (cosine)."""
    labels = np.asarray(labels)
    cluster_labels = sorted({int(label) for label in labels} - {-1})
    if not cluster_labels:
        return labels
    centroids = np.vstack(
        [normalize(embeddings[labels == cl].mean(axis=0)) for cl in cluster_labels]
    )
    outlier_mask = labels == -1
    if not np.any(outlier_mask):
        return labels
    sims = embeddings[outlier_mask] @ centroids.T
    new_labels = labels.copy()
    new_labels[outlier_mask] = np.asarray(cluster_labels)[np.argmax(sims, axis=1)]
    return new_labels


def cluster_instances(instances, embeddings):
    """Run UMAP + HDBSCAN over the embeddings of one topic's claims.

    Returns a list of clusters, each ``{"instances": [...], "centroid": np.array}``.
    """
    n = len(instances)

    if n < SUBTOPIC_HDBSCAN_MIN_CLUSTER_SIZE * 2:
        logger.debug(
            "Topic claim pool too small (%d) for sub-clustering; using one sub-cluster",
            n,
        )
        return [
            {
                "instances": list(instances),
                "centroid": normalize(embeddings.mean(axis=0)),
            }
        ]

    n_neighbors = max(2, min(SUBTOPIC_UMAP_N_NEIGHBORS, n - 1))
    n_components = max(2, min(SUBTOPIC_UMAP_N_COMPONENTS, n - 2))
    reduced = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=SUBTOPIC_UMAP_MIN_DIST,
        metric=SUBTOPIC_UMAP_METRIC,
        random_state=SUBTOPIC_UMAP_RANDOM_STATE,
    ).fit_transform(embeddings)

    labels = HDBSCAN(
        min_cluster_size=SUBTOPIC_HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=SUBTOPIC_HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    ).fit_predict(reduced)

    if all(label == -1 for label in labels):
        logger.debug("HDBSCAN found no structure in %d claims; using one sub-cluster", n)
        return [
            {
                "instances": list(instances),
                "centroid": normalize(embeddings.mean(axis=0)),
            }
        ]

    labels = reassign_outliers(labels, embeddings)
    clusters = []
    for cluster_label in sorted({int(label) for label in labels}):
        mask = labels == cluster_label
        members = [inst for inst, keep in zip(instances, mask) if keep]
        clusters.append(
            {
                "instances": members,
                "centroid": normalize(embeddings[mask].mean(axis=0)),
            }
        )
    return clusters


def central_sample(instances, embeddings, centroid, k):
    """Pick up to k instances closest to the centroid (cosine similarity)."""
    if len(instances) <= k:
        return list(instances)
    sims = embeddings @ centroid
    order = np.argsort(-sims)[:k]
    return [instances[int(i)] for i in order]


def frame_sub_cluster(topic_label, sample_instances):
    """Ask the LLM to name + frame a sub-cluster from a sample of its claims."""
    claim_list = "\n".join(
        f"{i + 1}. {inst.text}" for i, inst in enumerate(sample_instances)
    )
    payload = chat_completion(
        messages=[
            {
                "role": "system",
                "content": SUB_TOPIC_FRAME_PROMPT.format(
                    topic_label=topic_label, claim_list=claim_list
                ),
            },
            {"role": "user", "content": "Return only the JSON object."},
        ],
        schema=SUB_TOPIC_FRAME_SCHEMA,
    )
    label = (payload.get(Field.LABEL) or "").strip()
    target = payload.get(Field.POLARITY_TARGET)
    if isinstance(target, str):
        target = target.strip() or None
    return label, target


def discover_sub_topics_for_topic(conn, topic):
    """Cluster the topic's claims, frame each sub-cluster via the LLM, persist.

    Sub-cluster framing LLM calls run concurrently up to ``LLM_CONCURRENCY``.
    """
    instances = (
        conn.query(ArgumentInstance)
        .filter(
            ArgumentInstance.topic_id == topic.id,
            ArgumentInstance.sub_topic_id == None,  # noqa: E711
        )
        .order_by(ArgumentInstance.id)
        .all()
    )
    if not instances:
        logger.debug("Topic %s has no unassigned claims", topic.id)
        return 0

    logger.info(
        "Sub-clustering %d claims for topic %s (%s)",
        len(instances),
        topic.id,
        topic.label,
    )
    embeddings = embed_preference_texts([inst.text for inst in instances])
    sub_clusters = cluster_instances(instances, embeddings)

    samples = [
        central_sample(
            sc["instances"],
            embed_preference_texts([inst.text for inst in sc["instances"]]),
            sc["centroid"],
            SUBTOPIC_LLM_SAMPLE_SIZE,
        )
        for sc in sub_clusters
    ]

    topic_label = topic.label
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        future_to_index = {
            pool.submit(frame_sub_cluster, topic_label, samples[idx]): idx
            for idx in range(len(sub_clusters))
        }
        frames = [None] * len(sub_clusters)
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            label, polarity_target = future.result()
            frames[idx] = (label, polarity_target)

    centroids = {}
    for idx, sub_cluster in enumerate(sub_clusters):
        label, polarity_target = frames[idx]
        sub_topic_id = f"st_{uuid.uuid4().hex[:12]}"
        create_sub_topic(conn, sub_topic_id, topic.id, label, polarity_target)
        for instance in sub_cluster["instances"]:
            assign_instance_to_sub_topic(conn, instance.id, sub_topic_id)
        centroids[sub_topic_id] = {
            Field.CENTROID: sub_cluster["centroid"],
            Field.COUNT: len(sub_cluster["instances"]),
        }
        logger.info(
            "Topic %s sub-topic %s '%s' (%s) -> %d claims",
            topic.id,
            sub_topic_id,
            label,
            "descriptive" if polarity_target is None else f"target: {polarity_target}",
            len(sub_cluster["instances"]),
        )
    update_sub_topic_centroids(conn, centroids)
    return len(sub_clusters)


def is_outlier_topic(topic):
    return topic.id == str(OUTLIER_TOPIC)


def run_batch():
    start = monotonic()
    logger.info("Starting sub-topic discovery batch")
    conn = connect()
    topics = [topic for topic in get_topics(conn) if not is_outlier_topic(topic)]
    topic_ids = [topic.id for topic in topics]
    logger.info("Resetting sub-topic state for %d topics", len(topic_ids))
    reset_sub_topic_state(conn, topic_ids)

    logger.info(
        "Sub-topic discovery progress: %d topics, concurrency=%d",
        len(topics),
        LLM_CONCURRENCY,
    )
    for topic in tqdm(topics, desc="topics", unit="topic"):
        logger.info("Topic: %s", topic.label)
        discover_sub_topics_for_topic(conn, topic)
    conn.close()
    logger.info(
        "Finished sub-topic discovery batch in %s",
        tqdm.format_interval(monotonic() - start),
    )
