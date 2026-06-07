"""Stage 3: assign per-claim polarity, then cluster arguments per topic."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from time import monotonic
import uuid

from hdbscan import HDBSCAN
import numpy as np
from umap import UMAP

from config import (
    ARGUMENT_HDBSCAN_MIN_CLUSTER_SIZE,
    ARGUMENT_HDBSCAN_MIN_SAMPLES,
    ARGUMENT_UMAP_METRIC,
    ARGUMENT_UMAP_MIN_DIST,
    ARGUMENT_UMAP_N_COMPONENTS,
    ARGUMENT_UMAP_N_NEIGHBORS,
    ARGUMENT_UMAP_RANDOM_STATE,
    LLM_CONCURRENCY,
)
from db import ArgumentInstance, Field, Polarity, connect
from db.reads import get_topics, get_unclustered_instances
from db.writes import (
    assign_instance_to_cluster,
    clear_argument_clusters,
    create_argument_cluster,
    reset_argument_graph_state,
    set_claim_polarity,
    update_cluster_centroids,
)
from pipeline.utils.embeddings import embed_preference_texts
from pipeline.utils.llm import chat_completion
from pipeline.topic_clustering import OUTLIER_TOPIC

logger = logging.getLogger(__name__)

STANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "polarity": {"type": "string", "enum": ["for", "against"]},
    },
    "required": ["polarity"],
    "additionalProperties": False,
}

STANCE_PROMPT_TEMPLATE = """You are detecting the stance of a single claim toward {topic}.

Polarity frame:
- "for" means the claim supports this position: {polarity_target}
- "against" means the claim opposes this position: {polarity_target}

Return your response as JSON: {{"polarity": "for"}} or {{"polarity": "against"}}"""


@dataclass
class BatchProgress:
    total: int
    start_time: float
    completed: int = 0


def format_duration(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def progress_bar(completed, total, width=24):
    if total == 0:
        return "[" + "-" * width + "]"
    filled = round(width * completed / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def log_progress(progress, label):
    progress.completed += 1
    elapsed = monotonic() - progress.start_time
    avg = elapsed / progress.completed
    remaining = avg * (progress.total - progress.completed)
    percent = 100 * progress.completed / progress.total if progress.total else 100
    logger.info(
        "%s %d/%d %.1f%% | elapsed %s | eta %s | %s",
        progress_bar(progress.completed, progress.total),
        progress.completed,
        progress.total,
        percent,
        format_duration(elapsed),
        format_duration(remaining),
        label,
    )


def assign_claim_polarity(claim_text, topic_label, polarity_target):
    """Classify a single claim as ``for`` or ``against`` the topic frame."""
    logger.debug("Assigning polarity for claim under topic %s", topic_label)
    payload = chat_completion(
        messages=[
            {
                "role": "system",
                "content": STANCE_PROMPT_TEMPLATE.format(
                    topic=topic_label,
                    polarity_target=polarity_target,
                ),
            },
            {"role": "user", "content": claim_text},
        ],
        schema=STANCE_SCHEMA,
    )
    return Polarity(payload[Field.POLARITY])


def assign_polarity_for_topic(conn, topic, progress=None):
    """Set polarity on every claim under this topic that doesn't have one.

    LLM calls run concurrently up to ``LLM_CONCURRENCY``; the DB write for each
    result happens on the main thread so the shared connection stays safe.
    """
    pending = (
        conn.query(ArgumentInstance)
        .filter(
            ArgumentInstance.topic_id == topic.id,
            ArgumentInstance.polarity == None,  # noqa: E711
        )
        .order_by(ArgumentInstance.id)
        .all()
    )
    if not pending:
        logger.debug("Topic %s has no claims missing polarity", topic.id)
        return 0

    logger.info(
        "Assigning polarity for %d claims under topic %s (%s) concurrency=%d",
        len(pending),
        topic.id,
        topic.label,
        LLM_CONCURRENCY,
    )
    pending_payload = [(instance.id, instance.text) for instance in pending]
    topic_label = topic.label
    polarity_target = topic.polarity_target
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        future_to_id = {
            pool.submit(
                assign_claim_polarity, text, topic_label, polarity_target
            ): instance_id
            for instance_id, text in pending_payload
        }
        for future in as_completed(future_to_id):
            instance_id = future_to_id[future]
            polarity = future.result()
            set_claim_polarity(conn, instance_id, polarity)
            if progress is not None:
                log_progress(
                    progress,
                    f"polarity {topic.id} {instance_id} -> {polarity}",
                )
    return len(pending)


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


def cluster_polarity_bucket(instances):
    """Run UMAP + HDBSCAN over one (topic, polarity) bucket of argument instances.

    Returns a list of clusters, each as ``{"instances": [...], "centroid": np.array}``.
    """
    embeddings = embed_preference_texts([inst.text for inst in instances])
    n = len(instances)

    if n < ARGUMENT_HDBSCAN_MIN_CLUSTER_SIZE * 2:
        logger.debug(
            "Bucket too small for UMAP+HDBSCAN (%d < %d); using a single cluster",
            n,
            ARGUMENT_HDBSCAN_MIN_CLUSTER_SIZE * 2,
        )
        return [
            {
                "instances": list(instances),
                "centroid": normalize(embeddings.mean(axis=0)),
            }
        ]

    n_neighbors = max(2, min(ARGUMENT_UMAP_N_NEIGHBORS, n - 1))
    n_components = max(2, min(ARGUMENT_UMAP_N_COMPONENTS, n - 2))
    reduced = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=ARGUMENT_UMAP_MIN_DIST,
        metric=ARGUMENT_UMAP_METRIC,
        random_state=ARGUMENT_UMAP_RANDOM_STATE,
    ).fit_transform(embeddings)

    labels = HDBSCAN(
        min_cluster_size=ARGUMENT_HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=ARGUMENT_HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    ).fit_predict(reduced)

    if all(label == -1 for label in labels):
        logger.debug(
            "HDBSCAN found no structure in bucket of %d; using a single cluster", n
        )
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


def cluster_topic_arguments(conn, topic_id):
    """Cluster polarized argument instances for a topic with UMAP + HDBSCAN per polarity."""
    unclustered = [
        instance
        for instance in get_unclustered_instances(conn, topic_id)
        if instance.polarity is not None
    ]
    if not unclustered:
        logger.debug("No unclustered arguments for topic %s", topic_id)
        return

    by_polarity = {Polarity.FOR: [], Polarity.AGAINST: []}
    for instance in unclustered:
        by_polarity[instance.polarity].append(instance)

    centroids = {}
    for polarity, polarity_instances in by_polarity.items():
        if not polarity_instances:
            continue
        logger.debug(
            "Clustering %d %s arguments for topic %s",
            len(polarity_instances),
            polarity,
            topic_id,
        )
        clusters = cluster_polarity_bucket(polarity_instances)
        for cluster in clusters:
            new_id = f"ac_{uuid.uuid4().hex[:12]}"
            create_argument_cluster(conn, new_id, polarity, topic_id)
            for instance in cluster["instances"]:
                assign_instance_to_cluster(conn, instance.id, new_id)
            centroids[new_id] = {
                Field.POLARITY: polarity,
                Field.CENTROID: cluster["centroid"],
                Field.COUNT: len(cluster["instances"]),
            }
        logger.debug(
            "Topic %s polarity %s produced %d clusters",
            topic_id,
            polarity,
            len(clusters),
        )

    update_cluster_centroids(conn, centroids)


def is_outlier_topic(topic):
    return topic.id == str(OUTLIER_TOPIC)


def run_batch():
    """Assign per-claim polarity, then cluster arguments per topic."""
    start = monotonic()
    logger.info("Starting argument graph batch")
    conn = connect()
    topics = [topic for topic in get_topics(conn) if not is_outlier_topic(topic)]
    topic_ids = [topic.id for topic in topics]
    logger.info(
        "Resetting argument cluster state for %d topics (keeping claims)",
        len(topic_ids),
    )
    reset_argument_graph_state(conn, topic_ids)

    pending_total = (
        conn.query(ArgumentInstance)
        .filter(
            ArgumentInstance.topic_id.in_(topic_ids),
            ArgumentInstance.polarity == None,  # noqa: E711
        )
        .count()
    )
    progress = BatchProgress(
        total=pending_total + len(topics),
        start_time=start,
    )
    logger.info(
        "Argument graph progress: %d topics, %d claims missing polarity",
        len(topics),
        pending_total,
    )
    for index, topic in enumerate(topics, start=1):
        logger.info(
            "Topic %d/%d: %s",
            index,
            len(topics),
            topic.label,
        )
        assign_polarity_for_topic(conn, topic, progress=progress)
        cluster_topic_arguments(conn, topic.id)
        log_progress(
            progress,
            f"topic {index}/{len(topics)} cluster arguments for topic {topic.id}",
        )
    conn.close()
    logger.info("Finished argument graph batch in %s", format_duration(monotonic() - start))


def run_recluster():
    """Rebuild argument clusters from existing ArgumentInstance rows."""
    start = monotonic()
    logger.info("Starting argument reclustering batch")
    conn = connect()
    topics = [topic for topic in get_topics(conn) if not is_outlier_topic(topic)]
    topic_ids = [topic.id for topic in topics]
    logger.info("Clearing existing argument clusters for %d topics", len(topic_ids))
    clear_argument_clusters(conn, topic_ids)
    progress = BatchProgress(total=len(topics), start_time=start)

    for index, topic in enumerate(topics, start=1):
        logger.info(
            "Topic %d/%d: reclustering existing arguments (%s)",
            index,
            len(topics),
            topic.label,
        )
        cluster_topic_arguments(conn, topic.id)
        log_progress(
            progress,
            f"topic {index}/{len(topics)} recluster arguments for topic {topic.id}",
        )

    conn.close()
    logger.info(
        "Finished argument reclustering batch in %s",
        format_duration(monotonic() - start),
    )
