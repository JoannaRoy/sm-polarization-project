"""Stage 4: classify every claim in each non-descriptive sub-topic as agree
or disagree relative to that sub-topic's ``polarity_target`` statement.

Sub-topics with ``polarity_target IS NULL`` (descriptive) are skipped entirely;
their claims keep ``polarity = NULL``.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from time import monotonic

from tqdm import tqdm

from config import LLM_CONCURRENCY
from db import ArgumentInstance, Field, Polarity, connect
from db.reads import get_sub_topics_for_topic, get_topics
from db.writes import set_claim_polarity
from pipeline.sub_topic_discovery import is_outlier_topic
from pipeline.utils.llm import chat_completion

logger = logging.getLogger(__name__)

STANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "polarity": {"type": "string", "enum": ["agree", "disagree"]},
    },
    "required": ["polarity"],
    "additionalProperties": False,
}

STANCE_PROMPT_TEMPLATE = """You are detecting whether a single claim agrees or disagrees with a sub-topic statement.

Sub-topic: {sub_topic_label} (within the broader topic: {topic_label}).

Statement: {polarity_target}

Decide:
- "agree" if the claim expresses agreement with the statement above.
- "disagree" if the claim expresses disagreement with the statement above.

Return your response as JSON: {{"polarity": "agree"}} or {{"polarity": "disagree"}}"""


def assign_claim_polarity(claim_text, topic_label, sub_topic_label, polarity_target):
    payload = chat_completion(
        messages=[
            {
                "role": "system",
                "content": STANCE_PROMPT_TEMPLATE.format(
                    topic_label=topic_label,
                    sub_topic_label=sub_topic_label,
                    polarity_target=polarity_target,
                ),
            },
            {"role": "user", "content": claim_text},
        ],
        schema=STANCE_SCHEMA,
    )
    return Polarity(payload[Field.POLARITY])


def assign_polarity_for_sub_topic(conn, topic, sub_topic, pbar=None):
    """Set polarity on every claim under this sub-topic that doesn't have one.

    LLM calls run concurrently up to ``LLM_CONCURRENCY``; DB writes happen on
    the main thread so the shared connection stays safe.
    """
    pending = (
        conn.query(ArgumentInstance)
        .filter(
            ArgumentInstance.sub_topic_id == sub_topic.id,
            ArgumentInstance.polarity == None,  # noqa: E711
        )
        .order_by(ArgumentInstance.id)
        .all()
    )
    if not pending:
        logger.debug("Sub-topic %s has no claims missing polarity", sub_topic.id)
        return 0

    logger.info(
        "Assigning polarity for %d claims under sub-topic %s '%s' (topic %s) concurrency=%d",
        len(pending),
        sub_topic.id,
        sub_topic.label,
        topic.id,
        LLM_CONCURRENCY,
    )
    pending_payload = [(inst.id, inst.text) for inst in pending]
    topic_label = topic.label
    sub_topic_label = sub_topic.label
    polarity_target = sub_topic.polarity_target
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        future_to_id = {
            pool.submit(
                assign_claim_polarity,
                text,
                topic_label,
                sub_topic_label,
                polarity_target,
            ): instance_id
            for instance_id, text in pending_payload
        }
        for future in as_completed(future_to_id):
            instance_id = future_to_id[future]
            polarity = future.result()
            set_claim_polarity(conn, instance_id, polarity)
            if pbar is not None:
                pbar.update(1)
    return len(pending)


def run_batch():
    """Assign polarity per claim, for every non-descriptive sub-topic."""
    start = monotonic()
    logger.info("Starting polarity assignment batch")
    conn = connect()
    topics = [topic for topic in get_topics(conn) if not is_outlier_topic(topic)]

    work_items = []
    for topic in topics:
        for sub_topic in get_sub_topics_for_topic(conn, topic.id):
            if sub_topic.polarity_target is None:
                logger.info(
                    "Skipping descriptive sub-topic %s '%s' (topic %s)",
                    sub_topic.id,
                    sub_topic.label,
                    topic.id,
                )
                continue
            work_items.append((topic, sub_topic))

    pending_total = 0
    for _, sub_topic in work_items:
        pending_total += (
            conn.query(ArgumentInstance)
            .filter(
                ArgumentInstance.sub_topic_id == sub_topic.id,
                ArgumentInstance.polarity == None,  # noqa: E711
            )
            .count()
        )
    logger.info(
        "Polarity assignment progress: %d sub-topics, %d claims missing polarity",
        len(work_items),
        pending_total,
    )
    with tqdm(total=pending_total, desc="claims", unit="claim") as pbar:
        for topic, sub_topic in work_items:
            assign_polarity_for_sub_topic(conn, topic, sub_topic, pbar=pbar)
    conn.close()
    logger.info(
        "Finished polarity assignment batch in %s",
        tqdm.format_interval(monotonic() - start),
    )
