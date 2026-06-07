"""Pydantic response schemas. Each schema auto-maps from the matching ORM row
via `from_attributes=True`; cross-cutting reshape (bucketing representative
statements by scope, grouping claims by post) lives in `topic_detail`.

The pipeline does not collapse the GEN/DISC slate into a single statement, so
clusters and polarity sides expose their slate directly via
``representative_statements``. There is no synthesized statement string."""

import json
from collections import defaultdict
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict

from db.models import (
    ArgumentCluster,
    ArgumentInstance,
    Polarity,
    Post as PostRow,
    StatementLayer,
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


class Post(_Schema):
    id: str
    text: str
    claims: list[Claim]


class Cluster(_Schema):
    id: str
    polarity: Polarity
    count: int
    arguments: list[Argument]
    representative_statements: list[RepresentativeStatement]


class PolaritySlate(_Schema):
    polarity: Polarity
    representative_statements: list[RepresentativeStatement]


class TopicDetail(_Schema):
    id: str
    label: str
    polarity_target: str
    opinion_post_count: int
    argument_count: int
    posts: list[Post]
    clusters: list[Cluster]
    polarity_slates: list[PolaritySlate]


class TopicSummary(_Schema):
    id: str
    label: str
    polarity_target: str
    cluster_count: int
    argument_count: int
    has_polarity_slates: bool


def _bucket_statements(topic: Topic):
    by_cluster: dict[str, list] = defaultdict(list)
    by_polarity: dict[str, list] = defaultdict(list)
    for row in sorted(topic.representative_statements, key=lambda r: r.round_index):
        if row.layer == StatementLayer.ARGUMENT_CLUSTER:
            by_cluster[row.scope_id].append(row)
        elif row.layer == StatementLayer.POLARITY:
            _, polarity = row.scope_id.split(":", 1)
            by_polarity[polarity].append(row)
    return by_cluster, by_polarity


def _has_polarity_slate(topic: Topic) -> bool:
    return any(
        row.layer == StatementLayer.POLARITY
        for row in topic.representative_statements
    )


def topic_summary(topic: Topic) -> TopicSummary:
    return TopicSummary(
        id=topic.id,
        label=topic.label,
        polarity_target=topic.polarity_target,
        cluster_count=len(topic.clusters),
        argument_count=len(topic.arguments),
        has_polarity_slates=_has_polarity_slate(topic),
    )


def _post_with_claims(post: PostRow, claims: list[ArgumentInstance]) -> Post:
    return Post(
        id=post.id,
        text=post.text,
        claims=[Claim.model_validate(c) for c in sorted(claims, key=lambda c: c.id)],
    )


def _cluster_view(cluster: ArgumentCluster, slate) -> Cluster:
    return Cluster(
        id=cluster.id,
        polarity=cluster.polarity,
        count=cluster.count,
        arguments=[
            Argument.model_validate(a)
            for a in sorted(cluster.instances, key=lambda i: i.id)
        ],
        representative_statements=[
            RepresentativeStatement.model_validate(r) for r in slate
        ],
    )


def _polarity_slate_view(polarity: Polarity, slate) -> PolaritySlate:
    return PolaritySlate(
        polarity=polarity,
        representative_statements=[
            RepresentativeStatement.model_validate(r) for r in slate
        ],
    )


def topic_detail(topic: Topic) -> TopicDetail:
    by_cluster, by_polarity = _bucket_statements(topic)

    claims_by_post: dict[str, list[ArgumentInstance]] = defaultdict(list)
    posts_by_id: dict[str, PostRow] = {}
    for claim in topic.arguments:
        if claim.post is None:
            continue
        posts_by_id.setdefault(claim.post.id, claim.post)
        claims_by_post[claim.post.id].append(claim)

    polarity_slates = [
        _polarity_slate_view(Polarity(polarity), by_polarity[polarity])
        for polarity in sorted(by_polarity.keys())
    ]

    return TopicDetail(
        id=topic.id,
        label=topic.label,
        polarity_target=topic.polarity_target,
        opinion_post_count=len(posts_by_id),
        argument_count=len(topic.arguments),
        posts=[
            _post_with_claims(post, claims_by_post[post.id])
            for post in sorted(posts_by_id.values(), key=lambda p: p.id)
        ],
        clusters=[
            _cluster_view(c, by_cluster[c.id])
            for c in sorted(
                topic.clusters,
                key=lambda x: (x.polarity, -len(x.instances), x.id),
            )
        ],
        polarity_slates=polarity_slates,
    )
