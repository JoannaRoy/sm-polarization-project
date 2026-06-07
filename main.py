"""Run the full batch pipeline, or process a single post in real time."""

import argparse
import json
import logging

from config import LOG_LEVEL
from db import Field, Polarity, connect
from db.reads import get_polarity_slate, get_topic
from pipeline import argument_graph, claim_extraction, statement_generation, topic_clustering
from pipeline.utils.post_parsing import post_to_pipeline_post
from pipeline.topic_clustering import OUTLIER_TOPIC, top_topics_by_centroid

DEFAULT_TOPIC_CANDIDATES = 5

logger = logging.getLogger(__name__)


PIPELINE_STAGES = (
    (
        "claim-extraction",
        "Stage 1/5: claim extraction",
        claim_extraction.run_batch,
    ),
    ("topic-clustering", "Stage 2/5: topic clustering", topic_clustering.run_batch),
    ("argument-graph", "Stage 3/5: argument graph", argument_graph.run_batch),
    (
        "cluster-slate-generation",
        "Stage 4/5: cluster slate generation",
        statement_generation.run_cluster_batch,
    ),
    (
        "polarity-slate-generation",
        "Stage 5/5: polarity slate generation",
        statement_generation.run_polarity_batch,
    ),
)
RECLUSTERING_STAGE = (
    "argument-reclustering",
    "Stage 3b/5: argument reclustering",
    argument_graph.run_recluster,
)
STAGE_NAMES = tuple(name for name, _, _ in (*PIPELINE_STAGES, RECLUSTERING_STAGE))


def configure_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def selected_stages(only=None, start_at=None, skip=None):
    stage_lookup = {
        name: (label, run_stage)
        for name, label, run_stage in (*PIPELINE_STAGES, RECLUSTERING_STAGE)
    }
    stages = PIPELINE_STAGES
    if only is not None:
        label, run_stage = stage_lookup[only]
        stages = ((only, label, run_stage),)
    if start_at is not None:
        if start_at == RECLUSTERING_STAGE[0]:
            statement_stages = tuple(
                stage
                for stage in PIPELINE_STAGES
                if stage[0]
                in ("cluster-slate-generation", "polarity-slate-generation")
            )
            stages = (RECLUSTERING_STAGE, *statement_stages)
        else:
            pipeline_stage_names = tuple(name for name, _, _ in PIPELINE_STAGES)
            start_index = pipeline_stage_names.index(start_at)
            stages = PIPELINE_STAGES[start_index:]
    if skip:
        skipped = set(skip)
        stages = [stage for stage in stages if stage[0] not in skipped]
    return stages


def run_batch(only=None, start_at=None, skip=None, data_path=None):
    """Preprocessing run: process all test data through every stage.

    ``data_path`` (optional) overrides the default fixture path; it is only
    consumed by ``claim-extraction``, which is the one stage that reads the
    source JSON. Downstream stages operate on rows already in the DB.
    """
    stages = selected_stages(only=only, start_at=start_at, skip=skip)
    if not stages:
        logger.warning("No batch pipeline stages selected")
        return
    stage_names = ", ".join(name for name, _, _ in stages)
    logger.info("Starting batch pipeline stages: %s", stage_names)
    for name, label, run_stage in stages:
        logger.info(label)
        if name == "claim-extraction" and data_path is not None:
            run_stage(data_path=data_path)
        else:
            run_stage()
    logger.info("Finished batch pipeline stages: %s", stage_names)


def _format_slate(statements):
    return "\n".join(f"- {statement}" for statement in statements)


def format_topic_response(topic, polarity_slates):
    """Pure-template response: pair the matched topic with whichever
    precomputed FOR/AGAINST polarity slates exist for it. Each side renders
    as a bulleted list of slate statements so the proportional structure
    from GEN/DISC is preserved instead of collapsed into a single sentence."""
    sections = [f"Your post is about {topic.label}."]
    for_slate = polarity_slates.get(Polarity.FOR) or []
    against_slate = polarity_slates.get(Polarity.AGAINST) or []
    if for_slate:
        sections.append(
            f'Representative arguments in favor of "{topic.polarity_target}":\n'
            f"{_format_slate(for_slate)}"
        )
    if against_slate:
        sections.append(
            "Representative arguments against:\n"
            f"{_format_slate(against_slate)}"
        )
    if not for_slate and not against_slate:
        sections.append(
            "We do not yet have slates for the arguments on either side of this topic."
        )
    return "\n\n".join(sections)


def match_post_topics(post, k=DEFAULT_TOPIC_CANDIDATES):
    """Real-time: rank the top ``k`` candidate topics for the post by centroid
    similarity. Returns a list of {id, label, polarity_target, score} dicts so
    the caller (UI) can ask the user which one fits best."""
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
                "polarity_target": topic.polarity_target,
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
    """Return the templated FOR/AGAINST paragraph for a chosen topic."""
    conn = connect()
    topic = get_topic(conn, topic_id)
    if topic is None:
        conn.close()
        raise ValueError(f"Topic {topic_id} not found")
    polarity_slates = {
        polarity: get_polarity_slate(conn, topic_id, polarity)
        for polarity in Polarity
    }
    conn.close()
    return format_topic_response(topic, polarity_slates)


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
        help="Override the source JSON fixture for claim-extraction (e.g. test_data/mastodon_real.json).",
    )

    stage_parser = subparsers.add_parser("stage", help="Run one batch pipeline stage")
    stage_parser.add_argument("name", choices=STAGE_NAMES)
    stage_parser.add_argument(
        "--data-path",
        help="Override the source JSON fixture (only used by claim-extraction).",
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
        )
    elif args.command == "stage":
        run_batch(only=args.name, data_path=args.data_path)
    else:
        run_batch()
