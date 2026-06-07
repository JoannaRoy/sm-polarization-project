"""Stage 5: generate one representative slate per ``(sub_topic, polarity)``
bucket. GSC GEN/DISC runs directly on the raw claim texts now (no intermediate
argument-cluster slate layer).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from time import monotonic

import numpy as np
from sentence_transformers.util import cos_sim
from sqlalchemy import select

from config import (
    GSC_GEN_QUERY_ITEMS,
    GSC_SLATE_SIZE,
    LLM_CONCURRENCY,
)
from db import ArgumentInstance, Field, Polarity, RepresentativeStatement, connect
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

GEN_PROMPT = """You are generating a representative statement that {stance}s with this position: {polarity_target}.

Sub-topic: {sub_topic_label} (within the broader topic: {topic_label}).

You will be given social-media claims that all {stance} with the position above. Generate one concise statement (1-2 sentences) that best represents the viewpoint shared by a likely-to-agree subgroup of these claims.

The statement should read as a natural opinion on this side, be self-contained, and stay faithful to the source claims.

Return your response as JSON: {{"statement": "your statement here"}}"""


@dataclass(frozen=True)
class GscItem:
    id: str
    text: str


@dataclass(frozen=True)
class SlateJob:
    sub_topic_id: str
    sub_topic_label: str
    topic_id: str
    topic_label: str
    polarity_target: str
    polarity: Polarity
    items: tuple[GscItem, ...]


@dataclass(frozen=True)
class SlateResult:
    job: SlateJob
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


def ask_llm(system_prompt, user_content):
    start = monotonic()
    payload = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        schema=STATEMENT_SCHEMA,
    )
    result = payload[Field.STATEMENT]
    logger.debug(
        "Generated statement with %d characters in %.1fs",
        len(result),
        monotonic() - start,
    )
    return result


def format_items(label, texts):
    return f"{label}:\n" + "\n".join(f"- {text}" for text in texts)


def select_gen_context(items, embeddings, max_items=GSC_GEN_QUERY_ITEMS):
    """GEN context: central remaining items used to generate a candidate."""
    if len(items) <= max_items:
        return list(range(len(items)))

    centroid = embeddings.mean(axis=0)
    if np.linalg.norm(centroid) == 0:
        return list(range(max_items))

    scores = cos_sim(embeddings, centroid).flatten().numpy()
    return np.argsort(-scores)[:max_items].tolist()


def run_disc_proxy(statement, remaining_embeddings):
    """DISC: proxy representation by preference-space similarity."""
    statement_embedding = embed_preference_texts([statement])[0]
    return cos_sim(remaining_embeddings, statement_embedding).flatten().numpy()


def select_represented_items(scores, count):
    represented_count = min(count, len(scores))
    return np.argsort(-scores)[:represented_count].tolist()


def run_gsc_slate(system_prompt, items, slate_size):
    """Run the repeated GEN/DISC selection loop for one (sub_topic, polarity).

    Pure compute; the caller persists the rows, so this is safe in a worker thread.
    """
    if not items:
        return []

    item_embeddings = embed_preference_texts([item.text for item in items])
    remaining_indices = list(range(len(items)))
    rounds = min(slate_size, len(items))
    rows = []

    for round_index in range(1, rounds + 1):
        rounds_left = rounds - round_index + 1
        remaining_items = [items[index] for index in remaining_indices]
        remaining_embeddings = item_embeddings[remaining_indices]

        context_indices = select_gen_context(remaining_items, remaining_embeddings)
        context_items = [remaining_items[index] for index in context_indices]
        user_content = format_items("Claims", [item.text for item in context_items])
        statement = ask_llm(system_prompt, user_content)

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


def slate_prompt(job):
    stance = "agree" if job.polarity == Polarity.AGREE else "disagree"
    return GEN_PROMPT.format(
        stance=stance,
        polarity_target=job.polarity_target,
        sub_topic_label=job.sub_topic_label,
        topic_label=job.topic_label,
    )


def collect_jobs(conn, topics):
    """Build one job per (sub_topic, polarity) with at least one claim.

    Descriptive sub-topics (``polarity_target IS NULL``) are skipped.
    A sub-topic+polarity bucket that already has any persisted slate row is
    skipped so reruns are idempotent.
    """
    existing_scopes = {
        (sub_topic_id, Polarity(polarity).value)
        for sub_topic_id, polarity in conn.execute(
            select(
                RepresentativeStatement.sub_topic_id,
                RepresentativeStatement.polarity,
            ).distinct()
        ).all()
    }

    jobs = []
    for topic in topics:
        for sub_topic in topic.sub_topics:
            if sub_topic.polarity_target is None:
                continue
            by_polarity: dict[Polarity, list[ArgumentInstance]] = {
                Polarity.AGREE: [],
                Polarity.DISAGREE: [],
            }
            for instance in sub_topic.instances:
                if instance.polarity in by_polarity:
                    by_polarity[instance.polarity].append(instance)
            for polarity, instances in by_polarity.items():
                if not instances:
                    continue
                if (sub_topic.id, polarity.value) in existing_scopes:
                    logger.debug(
                        "Skipping (%s, %s): slate already exists",
                        sub_topic.id,
                        polarity,
                    )
                    continue
                items = tuple(GscItem(inst.id, inst.text) for inst in instances)
                jobs.append(
                    SlateJob(
                        sub_topic_id=sub_topic.id,
                        sub_topic_label=sub_topic.label,
                        topic_id=topic.id,
                        topic_label=topic.label,
                        polarity_target=sub_topic.polarity_target,
                        polarity=polarity,
                        items=items,
                    )
                )
    return jobs


def run_slate_job(job):
    rows = run_gsc_slate(slate_prompt(job), list(job.items), GSC_SLATE_SIZE)
    return SlateResult(job=job, slate_rows=tuple(rows))


def persist_result(conn, result):
    replace_representative_statements(
        conn,
        result.job.sub_topic_id,
        result.job.polarity,
        list(result.slate_rows),
    )


def run_batch():
    """Generate the GEN/DISC slate for every (sub_topic, polarity) bucket."""
    start = monotonic()
    logger.info("Starting slate generation batch")
    preference_embedder()
    conn = connect()
    topics = get_topics(conn)
    jobs = collect_jobs(conn, topics)
    progress = BatchProgress(total=len(jobs), start_time=start)
    logger.info(
        "Slate generation progress: %d topics, %d slate jobs, concurrency=%d",
        len(topics),
        len(jobs),
        LLM_CONCURRENCY,
    )
    if not jobs:
        conn.close()
        logger.info("No slate jobs to run")
        return

    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        future_to_job = {pool.submit(run_slate_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            persist_result(conn, result)
            log_progress(
                progress,
                f"sub-topic {job.sub_topic_id} {job.polarity} slate",
            )

    conn.close()
    logger.info(
        "Finished slate generation batch in %s",
        format_duration(monotonic() - start),
    )
