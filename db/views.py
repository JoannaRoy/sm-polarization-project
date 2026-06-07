"""Pydantic response schemas. Each schema auto-maps from the matching ORM row
via `from_attributes=True`; cross-cutting reshape (bucketing representative
statements by polarity, grouping claims by post) lives in `topic_detail`.

The pipeline does not collapse the GEN/DISC slate into a single statement, so
each sub-topic exposes its agree/disagree slates directly via
``representative_statements``. There is no synthesized statement string."""

import json
from collections import defaultdict
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict

from db.models import (
    ArgumentInstance,
    Polarity,
    Post as PostRow,
    SubTopic,
    Topic,
)


class _Schema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def _decode_ids(value):
    return json.loads(value) if isinstance(value, str) else value


class Argument(_Schema):
    id: str
    text: str
    post_id: str
    topic_sentence: str | None
    polarity: Polarity | None


class RepresentativeStatement(_Schema):
    id: str
    round_index: int
    statement: str
    represented_count: int
    represented_ids: Annotated[list[str], BeforeValidator(_decode_ids)]


class Claim(_Schema):
    id: str
    text: str
    topic_sentence: str | None
    polarity: Polarity | None
    sub_topic_id: str | None


class Post(_Schema):
    id: str
    text: str
    claims: list[Claim]


class SubTopicDetail(_Schema):
    id: str
    label: str
    polarity_target: str | None
    count: int
    arguments: list[Argument]
    agree_slate: list[RepresentativeStatement]
    disagree_slate: list[RepresentativeStatement]


MAX_SUB_TOPICS_PER_TOPIC = 8


class TopicDetail(_Schema):
    id: str
    label: str
    opinion_post_count: int
    argument_count: int
    posts: list[Post]
    sub_topics: list[SubTopicDetail]
    total_sub_topic_count: int


class TopicSummary(_Schema):
    id: str
    label: str
    sub_topic_count: int
    argument_count: int
    has_any_slates: bool


def _has_any_slates(topic: Topic) -> bool:
    return any(
        sub.representative_statements
        for sub in topic.sub_topics
    )


def topic_summary(topic: Topic) -> TopicSummary:
    return TopicSummary(
        id=topic.id,
        label=topic.label,
        sub_topic_count=len(topic.sub_topics),
        argument_count=len(topic.arguments),
        has_any_slates=_has_any_slates(topic),
    )


def _post_with_claims(post: PostRow, claims: list[ArgumentInstance]) -> Post:
    return Post(
        id=post.id,
        text=post.text,
        claims=[Claim.model_validate(c) for c in sorted(claims, key=lambda c: c.id)],
    )


def _sub_topic_detail(sub_topic: SubTopic) -> SubTopicDetail:
    by_polarity: dict[str, list] = defaultdict(list)
    for row in sorted(
        sub_topic.representative_statements, key=lambda r: r.round_index
    ):
        by_polarity[row.polarity].append(row)
    return SubTopicDetail(
        id=sub_topic.id,
        label=sub_topic.label,
        polarity_target=sub_topic.polarity_target,
        count=sub_topic.count,
        arguments=[
            Argument.model_validate(inst)
            for inst in sorted(sub_topic.instances, key=lambda i: i.id)
        ],
        agree_slate=[
            RepresentativeStatement.model_validate(r)
            for r in by_polarity.get(Polarity.AGREE, [])
        ],
        disagree_slate=[
            RepresentativeStatement.model_validate(r)
            for r in by_polarity.get(Polarity.DISAGREE, [])
        ],
    )


def topic_detail(topic: Topic) -> TopicDetail:
    claims_by_post: dict[str, list[ArgumentInstance]] = defaultdict(list)
    posts_by_id: dict[str, PostRow] = {}
    for claim in topic.arguments:
        if claim.post is None:
            continue
        posts_by_id.setdefault(claim.post.id, claim.post)
        claims_by_post[claim.post.id].append(claim)

    sub_topics = sorted(
        topic.sub_topics,
        key=lambda s: (s.polarity_target is None, -s.count, s.id),
    )

    return TopicDetail(
        id=topic.id,
        label=topic.label,
        opinion_post_count=len(posts_by_id),
        argument_count=len(topic.arguments),
        posts=[
            _post_with_claims(post, claims_by_post[post.id])
            for post in sorted(posts_by_id.values(), key=lambda p: p.id)
        ],
        sub_topics=[_sub_topic_detail(s) for s in sub_topics[:MAX_SUB_TOPICS_PER_TOPIC]],
        total_sub_topic_count=len(sub_topics),
    )
