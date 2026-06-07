"""Stage 1: extract opinion claims and topic sentences from posts (topic-agnostic)."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic

from config import DB_PATH, LLM_CONCURRENCY
from db import Field, Post, connect
from db.writes import (
    clear_claim_extraction_markers,
    clear_claim_extractions,
    insert_posts,
    mark_post_claims_extracted,
    store_claims,
)
from pipeline.utils.llm import LLMResponseError, chat_completion
from pipeline.utils.post_parsing import normalize_posts
from pipeline.topic_clustering import load_posts

logger = logging.getLogger(__name__)

FAILURES_PATH = Path(DB_PATH).parent / "claim_extraction.failures.json"

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "post_topic": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["post_topic", "claims"],
    "additionalProperties": False,
}

CLAIM_PROMPT = """You are extracting opinion claims from social media posts.

A claim is a position the author takes that another reader could agree or disagree with: an opinion, preference, recommendation, judgment, or stance.

Skip posts that do not express any opinion. This includes factual updates, status updates, greetings, jokes without stance, neutral observations, and pure questions. For those, return an empty list of claims.

For each post, return:
- "post_topic": a short phrase (1-5 words) naming the broad subject the post discusses. This should be a category or topic name, not a stance. Use the same phrase for every post about the same subject so similar posts cluster together.
- "claims": a list of objects with "text" set to the claim as a concise, self-contained sentence that names its subject.

Examples:

Post: "Carbon taxes just hurt working families. The rich can afford it, everyone else suffers."
Output: {"post_topic": "carbon taxes", "claims": [{"text": "Carbon taxes disproportionately burden lower-income families"}]}

Post: "Cats are way better than dogs. Lower maintenance and they don't bark."
Output: {"post_topic": "cats versus dogs", "claims": [{"text": "Cats are preferable to dogs because they require less maintenance"}, {"text": "Cats are preferable to dogs because dogs are noisy"}]}

Post: "Remote work is great for focus but I miss whiteboard sessions."
Output: {"post_topic": "remote work", "claims": [{"text": "Remote work improves focus"}, {"text": "Remote work makes whiteboard collaboration harder"}]}

Post: "Had a great lunch today"
Output: {"post_topic": "lunch", "claims": []}

Post: "Just deployed v3.2 to production"
Output: {"post_topic": "deployment", "claims": []}

Post: "Should I get a cat or a dog?"
Output: {"post_topic": "cats versus dogs", "claims": []}"""


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


def record_failed_post(post_id, message):
    """Append a failure record so it can be inspected and retried by hand."""
    entries = []
    if FAILURES_PATH.exists():
        entries = json.loads(FAILURES_PATH.read_text())
    if any(entry["post_id"] == post_id for entry in entries):
        return
    entries.append(
        {
            "post_id": post_id,
            "error": message,
            "at": datetime.utcnow().isoformat() + "Z",
        }
    )
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_PATH.write_text(json.dumps(entries, indent=2))


def extract_claims(post_text):
    """Send a post to the local LLM and return ``[{text, topic_sentence}]``.

    The model emits one ``post_topic`` for the post; every claim from the post
    inherits it as its ``topic_sentence`` so all claims about the same subject
    cluster together regardless of how each one is phrased.
    """
    logger.debug("Extracting claims from post")
    payload = chat_completion(
        messages=[
            {"role": "system", "content": CLAIM_PROMPT},
            {"role": "user", "content": post_text},
        ],
        schema=CLAIM_SCHEMA,
    )
    post_topic = payload.get(Field.POST_TOPIC, "").strip()
    cleaned = []
    for claim in payload.get(Field.CLAIMS, []):
        text = claim[Field.TEXT].strip()
        if text and post_topic:
            cleaned.append({Field.TEXT: text, Field.TOPIC_SENTENCE: post_topic})
    logger.debug(
        "Extracted %d claims with post topic %r",
        len(cleaned),
        post_topic,
    )
    return cleaned


def run_batch(data_path=None, resume=False):
    """Extract claims from every post in the test data and persist them.

    ``data_path`` overrides the default fixture file in ``config.TEST_DATA_PATH``;
    use it to point the pipeline at a different ``{"version": 1, "statuses": [...]}``
    JSON file (e.g. one produced by ``fetch_mastodon.py``).

    When ``resume=True``, prior claims and per-post markers are kept, and only
    posts whose ``claims_extracted_at`` is NULL are processed. A malformed LLM
    response for a single post is logged, recorded to
    ``data/claim_extraction.failures.json``, and treated as zero claims so the
    rest of the batch can finish.
    """
    start = monotonic()
    logger.info("Starting claim extraction batch (resume=%s)", resume)
    conn = connect()
    statuses = load_posts(data_path) if data_path else load_posts()
    posts = normalize_posts(statuses)

    if not resume:
        logger.info("Clearing prior claim, topic, cluster, and statement data")
        clear_claim_extractions(conn)
        clear_claim_extraction_markers(conn)

    logger.info("Inserting %d posts", len(posts))
    insert_posts(conn, posts)

    if resume:
        done_ids = {
            row[0]
            for row in conn.query(Post.id)
            .filter(Post.claims_extracted_at.isnot(None))
            .all()
        }
        pending = [post for post in posts if post[Field.ID] not in done_ids]
        logger.info(
            "Resuming: %d/%d posts already done, %d remaining",
            len(done_ids),
            len(posts),
            len(pending),
        )
    else:
        pending = list(posts)

    if not pending:
        conn.close()
        logger.info("Nothing to do; all posts already have claims_extracted_at set")
        return

    progress = BatchProgress(total=len(pending), start_time=start)
    total_claims = 0
    posts_with_claims = 0
    failed_posts = 0
    logger.info(
        "Extracting claims for %d posts with concurrency=%d",
        len(pending),
        LLM_CONCURRENCY,
    )
    with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as pool:
        future_to_post = {
            pool.submit(extract_claims, post[Field.TEXT]): post for post in pending
        }
        for index, future in enumerate(as_completed(future_to_post), start=1):
            post = future_to_post[future]
            try:
                claims = future.result()
            except LLMResponseError as e:
                logger.warning(
                    "post %s: %s; recording 0 claims and continuing",
                    post[Field.ID],
                    e,
                )
                record_failed_post(post[Field.ID], str(e))
                failed_posts += 1
                claims = []
            if claims:
                posts_with_claims += 1
                total_claims += len(claims)
                store_claims(conn, post[Field.ID], claims)
            mark_post_claims_extracted(conn, post[Field.ID])
            log_progress(
                progress,
                f"post {index}/{len(pending)} {post[Field.ID]} ({len(claims)} claims)",
            )

    conn.close()
    logger.info(
        "Finished claim extraction batch in %s: %d posts, %d with claims, %d total claims, %d failed",
        format_duration(monotonic() - start),
        len(pending),
        posts_with_claims,
        total_claims,
        failed_posts,
    )
