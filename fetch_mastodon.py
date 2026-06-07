"""Fetch real Mastodon hashtag-timeline statuses into a fixture-shaped JSON file.

The public hashtag-timeline endpoint
(`GET /api/v1/timelines/tag/{hashtag}`) is unauthenticated, returns up to 40
statuses per request, and emits the same status shape that
`pipeline/utils/post_parsing.py` already understands. The script pages via
`max_id`, drops reblogs/replies/non-English/empty posts, and writes the
survivors to a JSON file in the `{"version": 1, "statuses": [...]}` layout
that `pipeline/topic_clustering.load_posts` expects.

Usage:

    python fetch_mastodon.py --tag climate --per-tag 50
    python fetch_mastodon.py --tag climate --tag housing --output test_data/mastodon_real.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

from config import LOG_LEVEL, MASTODON_FIXTURE_BASE_URL
from pipeline.utils.post_parsing import mastodon_content_to_text

DEFAULT_INSTANCE = MASTODON_FIXTURE_BASE_URL
DEFAULT_TAGS = ("climate",)
DEFAULT_PER_TAG = 50
DEFAULT_OUTPUT_PATH = "test_data/mastodon_real.json"
DEFAULT_LANGUAGE = "en"

PAGE_LIMIT = 40
MIN_TEXT_CHARS = 20
MAX_PAGES_PER_TAG = 20
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.5
USER_AGENT = "sm-polarization-project/0.3 (+https://github.com/local)"

logger = logging.getLogger(__name__)


def keep_status(status, language):
    if status.get("reblog"):
        return False
    if status.get("in_reply_to_id"):
        return False
    status_lang = status.get("language")
    if language and status_lang and status_lang != language:
        return False
    content = status.get("content") or ""
    if not content.strip():
        return False
    if len(mastodon_content_to_text(content)) < MIN_TEXT_CHARS:
        return False
    return True


def fetch_tag(session, instance, tag, target_count, language):
    url = f"{instance.rstrip('/')}/api/v1/timelines/tag/{tag}"
    kept = []
    seen_ids = set()
    max_id = None
    for page in range(1, MAX_PAGES_PER_TAG + 1):
        params = {"limit": PAGE_LIMIT}
        if max_id is not None:
            params["max_id"] = max_id
        resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            logger.info("tag=%s page=%d empty batch, stopping", tag, page)
            break
        for status in batch:
            sid = status.get("id")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            if keep_status(status, language):
                kept.append(status)
                if len(kept) >= target_count:
                    break
        logger.info(
            "tag=%s page=%d kept=%d/%d (batch=%d)",
            tag, page, len(kept), target_count, len(batch),
        )
        if len(kept) >= target_count:
            break
        max_id = batch[-1].get("id")
        if max_id is None:
            break
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return kept


def fetch_all(instance, tags, per_tag, language):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    all_statuses = []
    seen_ids = set()
    for tag in tags:
        logger.info("Fetching #%s (target %d)", tag, per_tag)
        for status in fetch_tag(session, instance, tag, per_tag, language):
            sid = status.get("id")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            all_statuses.append(status)
    return all_statuses


def load_existing(output_path):
    path = Path(output_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("statuses", [])


def merge_statuses(existing, new):
    seen = {s["id"] for s in existing}
    merged = list(existing)
    added = 0
    for status in new:
        sid = status.get("id")
        if sid in seen:
            continue
        seen.add(sid)
        merged.append(status)
        added += 1
    return merged, added


def write_fixture(statuses, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "statuses": statuses}, indent=2))
    logger.info("Wrote %d statuses to %s", len(statuses), path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        default=DEFAULT_INSTANCE,
        help=f"Mastodon instance base URL (default: {DEFAULT_INSTANCE})",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Hashtag to fetch (omit the #). Repeat to add more.",
    )
    parser.add_argument(
        "--per-tag",
        type=int,
        default=DEFAULT_PER_TAG,
        help=f"Max statuses to keep per tag after filtering (default: {DEFAULT_PER_TAG})",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language filter, blank to disable (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge newly fetched statuses into the existing output file (dedupe by id) instead of overwriting it.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    tags = tuple(args.tags) if args.tags else DEFAULT_TAGS
    language = args.language or None
    fetched = fetch_all(args.instance, tags, args.per_tag, language)
    if args.append:
        existing = load_existing(args.output)
        merged, added = merge_statuses(existing, fetched)
        logger.info(
            "Appending: %d existing + %d new (deduped) = %d total",
            len(existing), added, len(merged),
        )
        write_fixture(merged, args.output)
    else:
        write_fixture(fetched, args.output)


if __name__ == "__main__":
    main()
