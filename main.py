"""Run the full batch pipeline, or process a single post in real time."""

import argparse
import json
import logging

from config import LOG_LEVEL
from db import Field, Polarity, connect
from db.reads import get_polarity_slate, get_sub_topics_for_topic, get_topic
from db.views import MAX_SUB_TOPICS_PER_TOPIC
from pipeline import (
    claim_extraction,
    polarity_assignment,
    statement_generation,
    sub_topic_discovery,
    topic_clustering,
)
from pipeline.topic_clustering import OUTLIER_TOPIC, top_topics_by_centroid
from pipeline.utils.post_parsing import post_to_pipeline_post

DEFAULT_TOPIC_CANDIDATES = 5

logger = logging.getLogger(__name__)


PIPELINE_STAGES = (
    (
        "claim-extraction",
        "Stage 1/5: claim extraction",
        claim_extraction.run_batch,
    ),
    ("topic-clustering", "Stage 2/5: topic clustering", topic_clustering.run_batch),
    (
        "sub-topic-discovery",
        "Stage 3/5: sub-topic discovery",
        sub_topic_discovery.run_batch,
    ),
    (
        "polarity-assignment",
        "Stage 4/5: polarity assignment",
        polarity_assignment.run_batch,
    ),
    (
        "slate-generation",
        "Stage 5/5: slate generation",
        statement_generation.run_batch,
    ),
)
STAGE_NAMES = tuple(name for name, _, _ in PIPELINE_STAGES)


def configure_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def selected_stages(only=None, start_at=None, skip=None):
    stage_lookup = {
        name: (label, run_stage) for name, label, run_stage in PIPELINE_STAGES
    }
    stages = PIPELINE_STAGES
    if only is not None:
        label, run_stage = stage_lookup[only]
        stages = ((only, label, run_stage),)
    if start_at is not None:
        pipeline_stage_names = tuple(name for name, _, _ in PIPELINE_STAGES)
        start_index = pipeline_stage_names.index(start_at)
        stages = PIPELINE_STAGES[start_index:]
    if skip:
        skipped = set(skip)
        stages = [stage for stage in stages if stage[0] not in skipped]
    return stages


def run_batch(only=None, start_at=None, skip=None, data_path=None, resume=False):
    """Preprocessing run: process all test data through every stage.

    ``data_path`` (optional) overrides the default fixture path; only
    consumed by ``claim-extraction`` (the one stage that reads source JSON).

    ``resume`` (optional) is forwarded only to ``claim-extraction``.
    """
    stages = selected_stages(only=only, start_at=start_at, skip=skip)
    if not stages:
        logger.warning("No batch pipeline stages selected")
        return
    stage_names = ", ".join(name for name, _, _ in stages)
    logger.info("Starting batch pipeline stages: %s", stage_names)
    for name, label, run_stage in stages:
        logger.info(label)
        if name == "claim-extraction":
            kwargs = {"resume": resume}
            if data_path is not None:
                kwargs["data_path"] = data_path
            run_stage(**kwargs)
        else:
            run_stage()
    logger.info("Finished batch pipeline stages: %s", stage_names)


def _format_slate(statements):
    return "\n".join(f"- {statement}" for statement in statements)


def format_topic_response(topic, sub_topic_views):
    """Pure-template response: walk every sub-topic of the matched topic and
    render its agree/disagree slates as nested bulleted lists. Descriptive
    sub-topics are skipped."""
    sections = [f"Your post is about {topic.label}."]
    rendered_any = False
    for sub_topic, agree_slate, disagree_slate in sub_topic_views:
        if sub_topic.polarity_target is None:
            continue
        if not agree_slate and not disagree_slate:
            continue
        rendered_any = True
        sections.append(
            f'Sub-topic: {sub_topic.label} -- "{sub_topic.polarity_target}"'
        )
        if agree_slate:
            sections.append(f"Agree:\n{_format_slate(agree_slate)}")
        if disagree_slate:
            sections.append(f"Disagree:\n{_format_slate(disagree_slate)}")
    if not rendered_any:
        sections.append(
            "We do not yet have slates for any sub-topic under this topic."
        )
    return "\n\n".join(sections)


def match_post_topics(post, k=DEFAULT_TOPIC_CANDIDATES):
    """Real-time: rank the top ``k`` candidate topics for the post by centroid
    similarity. Returns a list of {id, label, score} dicts so the caller (UI)
    can ask the user which one fits best."""
    logger.info("Matching single post against top %d topics", k)
    pipeline_post = post_to_pipeline_post(post)
    text = pipeline_post[Field.TEXT]

    matches = top_topics_by_centroid([text], k=k)[0]
    conn = connect()
    candidates = []
    for topic_id, score in matches:
        if topic_id == OUTLIER_TOPIC:
            continue
        topic = get_topic(conn, str(topic_id))
        if topic is None:
            continue
        candidates.append(
            {
                "id": topic.id,
                "label": topic.label,
                "score": score,
            }
        )
    conn.close()

    if not candidates:
        raise ValueError("Post does not match any known topic")
    logger.info(
        "Returning %d topic candidates for post (top match: %s)",
        len(candidates),
        candidates[0]["label"],
    )
    return candidates


def topic_response(topic_id):
    """Return the templated paragraph for a chosen topic, walking every sub-topic."""
    conn = connect()
    topic = get_topic(conn, topic_id)
    if topic is None:
        conn.close()
        raise ValueError(f"Topic {topic_id} not found")
    sub_topics = get_sub_topics_for_topic(conn, topic_id)
    sub_topics = sorted(
        sub_topics,
        key=lambda s: (s.polarity_target is None, -s.count, s.id),
    )[:MAX_SUB_TOPICS_PER_TOPIC]
    sub_topic_views = [
        (
            sub_topic,
            get_polarity_slate(conn, sub_topic.id, Polarity.AGREE),
            get_polarity_slate(conn, sub_topic.id, Polarity.DISAGREE),
        )
        for sub_topic in sub_topics
    ]
    conn.close()
    return format_topic_response(topic, sub_topic_views)


def process_post(post):
    """One-shot: pick the single best-matching topic and return its templated
    paragraph. Used by the CLI; the API exposes the two halves separately so
    the UI can present alternatives."""
    candidates = match_post_topics(post, k=1)
    return topic_response(candidates[0]["id"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    batch_parser = subparsers.add_parser("batch", help="Run batch pipeline stages")
    batch_group = batch_parser.add_mutually_exclusive_group()
    batch_group.add_argument("--only", choices=STAGE_NAMES)
    batch_group.add_argument("--from-stage", choices=STAGE_NAMES)
    batch_parser.add_argument("--skip", choices=STAGE_NAMES, nargs="*", default=[])
    batch_parser.add_argument(
        "--data-path",
        help="Override the source JSON fixture for claim-extraction.",
    )
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip claim-extraction posts already marked done.",
    )

    stage_parser = subparsers.add_parser("stage", help="Run one batch pipeline stage")
    stage_parser.add_argument("name", choices=STAGE_NAMES)
    stage_parser.add_argument(
        "--data-path",
        help="Override the source JSON fixture (only used by claim-extraction).",
    )
    stage_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a crashed claim-extraction run; ignored by other stages.",
    )

    process_parser = subparsers.add_parser("process", help="Process one post JSON")
    process_parser.add_argument("post_json")

    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    if args.command == "process":
        print(process_post(json.loads(args.post_json)))
    elif args.command == "batch":
        run_batch(
            only=args.only,
            start_at=args.from_stage,
            skip=args.skip,
            data_path=args.data_path,
            resume=args.resume,
        )
    elif args.command == "stage":
        run_batch(only=args.name, data_path=args.data_path, resume=args.resume)
    else:
        run_batch()
