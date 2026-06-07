"""Stage 3: generate representative statements for argument clusters."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from time import monotonic

import numpy as np
from sentence_transformers.util import cos_sim

from config import (
    GSC_CLUSTER_SLATE_SIZE,
    GSC_GEN_QUERY_ITEMS,
    GSC_POLARITY_SLATE_SIZE,
    LLM_CONCURRENCY,
)
from db import Field, Polarity, RepresentativeStatement, StatementLayer, connect
from db.reads import get_topics
from db.writes import replace_representative_statements
from pipeline.utils.embeddings import embed_preference_texts, preference_embedder
from pipeline.utils.llm import chat_completion

logger = logging.getLogger(__name__)

STATEMENT_SCHEMA = {
    "type": "object",
    "properties": {"statement": {"type": "string"}},
    "required": ["statement"],
    "additionalProperties": False,
}

CLUSTER_GEN_PROMPT = """You are generating a representative statement about {topic}.

You will be given arguments from social media posts that may form a coherent subgroup inside a larger argument cluster. Generate one concise statement (1-2 sentences) that best represents the viewpoint shared by this subgroup.

The statement should be neutral, self-contained, and faithful to the arguments.

Return your response as JSON: {{"statement": "your statement here"}}"""

POLARITY_GEN_PROMPT = """You are generating a representative statement that argues {stance} the position: {polarity_target}.

You will be given cluster-level slate statements that all share this stance. Generate one concise statement (1-2 sentences) that best represents the viewpoint shared by a likely-to-agree subgroup of these slate statements.

The statement should read as a natural argument on this side, be self-contained, and stay faithful to the source slate statements.

Return your response as JSON: {{"statement": "your statement here"}}"""

@dataclass(frozen=True)
class GscItem:
    id: str
    text: str


@dataclass(frozen=True)
class GscRunConfig:
    layer: StatementLayer
    scope_id: str
    topic_id: str
    topic_label: str
    polarity: Polarity | None
    slate_size: int
    gen_prompt: str
    item_label: str


@dataclass(frozen=True)
class ClusterJob:
    """Pre-materialized inputs for one cluster's GSC run, safe to pass between
    threads. Pulled off the SQLAlchemy session on the main thread so workers
    never touch ORM objects."""
    cluster_id: str
    topic_id: str
    topic_label: str
    polarity: Polarity
    items: tuple[GscItem, ...]


@dataclass(frozen=True)
class PolarityJob:
    topic_id: str
    topic_label: str
    polarity_target: str
    polarity: Polarity
    items: tuple[GscItem, ...]


@dataclass(frozen=True)
class GscResult:
    """Output of a single cluster or polarity job; the orchestrator persists it.

    The slate IS the output. There is no final synthesized statement; collapsing
    a slate into one sentence would discard the proportional structure that the
    GEN/DISC loop is designed to produce.
    """
    config: GscRunConfig
    slate_rows: tuple[dict, ...]


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


def ask_llm(system_prompt, user_content, key=Field.STATEMENT):
    start = monotonic()
    logger.debug("Requesting %s from LLM", key)
    payload = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        schema=STATEMENT_SCHEMA,
    )
    result = payload[key]
    logger.debug(
        "Generated %s with %d characters in %.1fs",
        key,
        len(result),
        monotonic() - start,
    )
    return result


def format_items(label, texts):
    return f"{label}:\n" + "\n".join(f"- {text}" for text in texts)


def select_gen_context(items, embeddings, max_items=GSC_GEN_QUERY_ITEMS):
    """GEN context S: central remaining items used to generate a candidate."""
    if len(items) <= max_items:
        return list(range(len(items)))

    centroid = embeddings.mean(axis=0)
    if np.linalg.norm(centroid) == 0:
        return list(range(max_items))

    scores = cos_sim(embeddings, centroid).flatten().numpy()
    return np.argsort(-scores)[:max_items].tolist()


def run_gen_query(config, context_items):
    """GEN(S, r): generate one statement for a likely-to-agree subgroup."""
    logger.debug(
        "Running GEN for %s scope %s with %d context items",
        config.layer,
        config.scope_id,
        len(context_items),
    )
    user_content = format_items(config.item_label, [item.text for item in context_items])
    return ask_llm(config.gen_prompt, user_content)


def run_disc_proxy(statement, remaining_embeddings):
    """DISC(i, alpha): proxy representation by preference-space similarity."""
    statement_embedding = embed_preference_texts([statement])[0]
    return cos_sim(remaining_embeddings, statement_embedding).flatten().numpy()


def select_represented_items(scores, count):
    """Remove the best-represented ~n/k remaining items."""
    represented_count = min(count, len(scores))
    return np.argsort(-scores)[:represented_count].tolist()


def run_gsc_slate(config, items):
    """Run the repeated GEN/DISC selection loop for one layer and scope.

    Pure compute: returns the slate rows. The caller persists them, so this is
    safe to invoke from a worker thread.
    """
    if not items:
        return []

    logger.debug(
        "Starting %s GSC slate for scope %s with %d items and up to %d rounds",
        config.layer,
        config.scope_id,
        len(items),
        config.slate_size,
    )
    item_embeddings = embed_preference_texts([item.text for item in items])
    remaining_indices = list(range(len(items)))
    rounds = min(config.slate_size, len(items))
    rows = []

    for round_index in range(1, rounds + 1):
        rounds_left = rounds - round_index + 1
        remaining_items = [items[index] for index in remaining_indices]
        remaining_embeddings = item_embeddings[remaining_indices]

        context_indices = select_gen_context(remaining_items, remaining_embeddings)
        context_items = [remaining_items[index] for index in context_indices]
        statement = run_gen_query(config, context_items)

        disc_scores = run_disc_proxy(statement, remaining_embeddings)
        quota = int(np.ceil(len(remaining_items) / rounds_left))
        represented_indices = select_represented_items(disc_scores, quota)
        represented_ids = [remaining_items[index].id for index in represented_indices]

        rows.append(
            {
                Field.ROUND_INDEX: round_index,
                Field.STATEMENT: statement,
                Field.REPRESENTED_IDS: represented_ids,
            }
        )

        represented_absolute = {
            remaining_indices[index] for index in represented_indices
        }
        remaining_indices = [
            index for index in remaining_indices if index not in represented_absolute
        ]

    return rows


def cluster_run_config(job):
    return GscRunConfig(
        layer=StatementLayer.ARGUMENT_CLUSTER,
        scope_id=job.cluster_id,
        topic_id=job.topic_id,
        topic_label=job.topic_label,
        polarity=job.polarity,
        slate_size=GSC_CLUSTER_SLATE_SIZE,
        gen_prompt=CLUSTER_GEN_PROMPT.format(topic=job.topic_label),
        item_label="Arguments",
    )


def polarity_run_config(job):
    stance = "for" if job.polarity == Polarity.FOR else "against"
    return GscRunConfig(
        layer=StatementLayer.POLARITY,
        scope_id=f"{job.topic_id}:{job.polarity}",
        topic_id=job.topic_id,
        topic_label=job.topic_label,
        polarity=job.polarity,
        slate_size=GSC_POLARITY_SLATE_SIZE,
        gen_prompt=POLARITY_GEN_PROMPT.format(
            stance=stance, polarity_target=job.polarity_target
        ),
        item_label="Cluster slate statements",
    )


def run_gsc_job(config, items):
    """Run one GSC job: produce the GEN/DISC slate. Pure compute, safe to thread."""
    rows = run_gsc_slate(config, list(items))
    return GscResult(config=config, slate_rows=tuple(rows))


def cluster_slates_by_id(conn, cluster_ids):
    """Return {cluster_id: [round-ordered slate statements]} for these clusters."""
    if not cluster_ids:
        return {}
    rows = (
        conn.query(RepresentativeStatement)
        .filter(
            RepresentativeStatement.layer == StatementLayer.ARGUMENT_CLUSTER,
            RepresentativeStatement.scope_id.in_(cluster_ids),
        )
        .order_by(RepresentativeStatement.round_index)
        .all()
    )
    slates: dict[str, list[RepresentativeStatement]] = {}
    for row in rows:
        slates.setdefault(row.scope_id, []).append(row)
    return slates


def collect_cluster_jobs(conn, topics):
    """Walk topics on the main thread (where the ORM session is bound) and pull
    out plain-data jobs for clusters that still need a slate. A cluster is
    skipped if it already has any representative statements stored."""
    cluster_ids = [cluster.id for topic in topics for cluster in topic.clusters]
    existing_slates = cluster_slates_by_id(conn, cluster_ids)
    jobs = []
    for topic in topics:
        for cluster in topic.clusters:
            if existing_slates.get(cluster.id):
                logger.debug(
                    "Skipping cluster %s: slate already exists", cluster.id
                )
                continue
            items = tuple(
                GscItem(instance.id, instance.text) for instance in cluster.instances
            )
            if not items:
                logger.debug(
                    "Skipping cluster %s: no arguments", cluster.id
                )
                continue
            jobs.append(
                ClusterJob(
                    cluster_id=cluster.id,
                    topic_id=cluster.topic_id,
                    topic_label=topic.label,
                    polarity=cluster.polarity,
                    items=items,
                )
            )
    return jobs


def collect_polarity_jobs(conn, topics):
    """Build polarity-layer jobs whose inputs are every cluster's slate
    statements (3 per cluster) pooled across clusters of the same stance.
    Pooling the slate instead of one synthesized cluster statement is the
    point of dropping the cluster-level collapse: the polarity GEN/DISC sees
    the within-cluster variation directly."""
    cluster_ids = [
        cluster.id
        for topic in topics
        for cluster in topic.clusters
    ]
    slates = cluster_slates_by_id(conn, cluster_ids)
    jobs = []
    for topic in topics:
        for polarity in Polarity:
            items: list[GscItem] = []
            for cluster in topic.clusters:
                if cluster.polarity != polarity:
                    continue
                for row in slates.get(cluster.id, []):
                    items.append(GscItem(row.id, row.statement))
            if not items:
                logger.info(
                    "Skipping %s stance for topic %s: no cluster slate statements",
                    polarity,
                    topic.id,
                )
                continue
            jobs.append(
                PolarityJob(
                    topic_id=topic.id,
                    topic_label=topic.label,
                    polarity_target=topic.polarity_target,
                    polarity=polarity,
                    items=tuple(items),
                )
            )
    return jobs


def persist_result(conn, result):
    config = result.config
    replace_representative_statements(
        conn,
        config.layer,
        config.scope_id,
        config.topic_id,
        config.polarity,
        list(result.slate_rows),
    )


def run_cluster_batch():
    """Generate the GEN/DISC slate for each argument cluster in every topic.

    LLM calls run concurrently across clusters up to ``LLM_CONCURRENCY``; the
    GEN/DISC rounds inside a single cluster stay sequential by design. DB
    writes happen on the main thread as each job completes.
    """
    start = monotonic()
    logger.info("Starting cluster slate generation batch")
    preference_embedder()
    conn = connect()
    topics = get_topics(conn)
    jobs = collect_cluster_jobs(conn, topics)
    progress = BatchProgress(total=len(jobs), start_time=start)
    logger.info(
        "Cluster slate progress: %d topics, %d pending clusters, concurrency=%d",
        len(topics),
        len(jobs),
        LLM_CONCURRENCY,
    )
    if not jobs:
        conn.close()
        logger.info("No clusters need new slates")
        return

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        future_to_job = {
            pool.submit(run_gsc_job, cluster_run_config(job), job.items): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            persist_result(conn, result)
            log_progress(
                progress,
                f"topic {job.topic_id} cluster {job.cluster_id} {job.polarity}",
            )

    conn.close()
    logger.info(
        "Finished cluster slate generation batch in %s",
        format_duration(monotonic() - start),
    )


def run_polarity_batch():
    """Generate the polarity-layer slate for each (topic, stance) from every
    cluster's slate statements. Requires the cluster-layer slates to exist."""
    start = monotonic()
    logger.info("Starting polarity slate generation batch")
    preference_embedder()
    conn = connect()
    topics = get_topics(conn)
    jobs = collect_polarity_jobs(conn, topics)
    progress = BatchProgress(total=len(jobs), start_time=start)
    logger.info(
        "Polarity slate progress: %d topics, %d stance slates, concurrency=%d",
        len(topics),
        len(jobs),
        LLM_CONCURRENCY,
    )
    if not jobs:
        conn.close()
        logger.info("No polarity slates to generate")
        return

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        future_to_job = {
            pool.submit(run_gsc_job, polarity_run_config(job), job.items): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            persist_result(conn, result)
            log_progress(
                progress,
                f"topic {job.topic_id} {job.polarity} stance slate",
            )

    conn.close()
    logger.info(
        "Finished polarity slate generation batch in %s",
        format_duration(monotonic() - start),
    )
