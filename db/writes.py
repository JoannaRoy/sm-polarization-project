"""Inserts and updates against the pipeline database."""

import json
import uuid
from datetime import datetime

import numpy as np
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db.models import (
    ArgumentInstance,
    Field,
    Post,
    RepresentativeStatement,
    SubTopic,
    Topic,
)

OUTLIER_TOPIC_ID = -1


# --- Posts / topics ---


def insert_posts(conn, posts):
    """Insert or refresh posts by id.

    Existing rows have their text refreshed but keep any prior topic
    assignments and claim-extraction markers, so re-running claim extraction
    in resume mode does not undo work from later stages.
    """
    rows = [{"id": p[Field.ID], "text": p[Field.TEXT]} for p in posts]
    if rows:
        stmt = sqlite_insert(Post).values(rows)
        conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[Post.id],
                set_={Post.text: stmt.excluded.text},
            )
        )
        conn.commit()


def mark_post_claims_extracted(conn, post_id):
    """Stamp a post as having been processed by claim extraction."""
    post = conn.get(Post, post_id)
    if post is not None:
        post.claims_extracted_at = datetime.utcnow()
        conn.commit()


def clear_claim_extraction_markers(conn):
    conn.query(Post).update(
        {Post.claims_extracted_at: None},
        synchronize_session=False,
    )
    conn.commit()


def write_topic_assignments_for_claims(conn, clusters):
    """Write Topic rows and update each claim's topic_id (outliers stay unassigned)."""
    for topic_id, info in clusters.items():
        if topic_id == OUTLIER_TOPIC_ID:
            continue
        tid = str(topic_id)
        label = info[Field.LABEL]
        centroid = info.get(Field.CENTROID)
        centroid_bytes = (
            np.asarray(centroid, dtype=np.float32).tobytes()
            if centroid is not None
            else None
        )
        topic = conn.get(Topic, tid)
        if topic is None:
            conn.add(Topic(id=tid, label=label, centroid=centroid_bytes))
        else:
            topic.label = label
            topic.centroid = centroid_bytes
        for claim_id in info[Field.CLAIM_IDS]:
            instance = conn.get(ArgumentInstance, claim_id)
            if instance is not None:
                instance.topic_id = tid
    conn.commit()


def refresh_post_primary_topics(conn):
    """Set each post's topic_id/topic_label to its most-frequent claim topic."""
    posts = conn.query(Post).all()
    for post in posts:
        topic_counts = {}
        for instance in post.arguments:
            if instance.topic_id is None:
                continue
            topic_counts[instance.topic_id] = topic_counts.get(instance.topic_id, 0) + 1
        if not topic_counts:
            post.topic_id = None
            post.topic_label = None
            continue
        primary_topic_id = max(topic_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        topic = conn.get(Topic, primary_topic_id)
        post.topic_id = primary_topic_id
        post.topic_label = topic.label if topic is not None else None
    conn.commit()


def reset_topic_dependent_state(conn):
    """Drop topics, sub-topics, statements; clear topic/sub-topic/polarity on claims."""
    for model in (RepresentativeStatement, SubTopic):
        conn.query(model).delete(synchronize_session=False)
    conn.query(ArgumentInstance).update(
        {
            ArgumentInstance.topic_id: None,
            ArgumentInstance.sub_topic_id: None,
            ArgumentInstance.polarity: None,
        },
        synchronize_session=False,
    )
    conn.query(Post).update(
        {Post.topic_id: None, Post.topic_label: None},
        synchronize_session=False,
    )
    conn.query(Topic).delete(synchronize_session=False)
    conn.commit()


# --- Argument instances ---


def reset_sub_topic_state(conn, topic_ids):
    """Drop sub-topics and statements for these topics; reset claim sub_topic_id and polarity."""
    if not topic_ids:
        return

    conn.query(ArgumentInstance).filter(ArgumentInstance.topic_id.in_(topic_ids)).update(
        {ArgumentInstance.sub_topic_id: None, ArgumentInstance.polarity: None},
        synchronize_session=False,
    )
    sub_topic_ids = [
        st.id
        for st in conn.query(SubTopic).filter(SubTopic.topic_id.in_(topic_ids)).all()
    ]
    if sub_topic_ids:
        conn.query(RepresentativeStatement).filter(
            RepresentativeStatement.sub_topic_id.in_(sub_topic_ids)
        ).delete(synchronize_session=False)
    conn.query(SubTopic).filter(SubTopic.topic_id.in_(topic_ids)).delete(
        synchronize_session=False
    )
    conn.commit()


def store_claims(conn, post_id, claims):
    """Persist topic-agnostic extracted claims as ArgumentInstance rows."""
    for claim in claims:
        conn.add(
            ArgumentInstance(
                id=f"arg_{uuid.uuid4().hex[:12]}",
                text=claim[Field.TEXT],
                topic_sentence=claim[Field.TOPIC_SENTENCE],
                post_id=post_id,
            )
        )
    conn.commit()


def clear_claim_extractions(conn):
    """Wipe everything that depends on claims before re-extracting them."""
    for model in (
        RepresentativeStatement,
        ArgumentInstance,
        SubTopic,
        Topic,
    ):
        conn.query(model).delete(synchronize_session=False)
    conn.query(Post).update(
        {Post.topic_id: None, Post.topic_label: None},
        synchronize_session=False,
    )
    conn.commit()


def set_claim_polarity(conn, instance_id, polarity):
    instance = conn.get(ArgumentInstance, instance_id)
    if instance is not None:
        instance.polarity = polarity
        conn.commit()


def assign_instance_to_sub_topic(conn, instance_id, sub_topic_id):
    instance = conn.get(ArgumentInstance, instance_id)
    if instance is not None:
        instance.sub_topic_id = sub_topic_id


# --- Sub-topics ---


def create_sub_topic(conn, sub_topic_id, topic_id, label, polarity_target):
    """Create a SubTopic row. ``polarity_target=None`` marks it descriptive."""
    conn.add(
        SubTopic(
            id=sub_topic_id,
            topic_id=topic_id,
            label=label,
            polarity_target=polarity_target,
            count=0,
        )
    )
    conn.flush()


def update_sub_topic_centroids(conn, centroids):
    """Persist centroid arrays + counts for sub-topics."""
    for sid, cdata in centroids.items():
        sub_topic = conn.get(SubTopic, sid)
        if sub_topic is not None:
            sub_topic.centroid = np.asarray(
                cdata[Field.CENTROID], dtype=np.float32
            ).tobytes()
            sub_topic.count = cdata[Field.COUNT]
    conn.commit()


def replace_representative_statements(conn, sub_topic_id, polarity, rows):
    conn.query(RepresentativeStatement).filter(
        RepresentativeStatement.sub_topic_id == sub_topic_id,
        RepresentativeStatement.polarity == polarity,
    ).delete()
    for row in rows:
        represented_ids = row[Field.REPRESENTED_IDS]
        conn.add(
            RepresentativeStatement(
                id=f"rs_{uuid.uuid4().hex[:12]}",
                sub_topic_id=sub_topic_id,
                polarity=polarity,
                round_index=row[Field.ROUND_INDEX],
                statement=row[Field.STATEMENT],
                represented_ids=json.dumps(represented_ids),
                represented_count=len(represented_ids),
            )
        )
    conn.commit()
