"""Inserts and updates against the pipeline database."""

import json
import uuid

import numpy as np
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db.models import (
    ArgumentCluster,
    ArgumentInstance,
    Field,
    Post,
    RepresentativeStatement,
    Topic,
)

OUTLIER_TOPIC_ID = -1


# --- Posts / topics ---


def insert_posts(conn, posts):
    """Insert or refresh posts by id."""
    rows = [{"id": p[Field.ID], "text": p[Field.TEXT]} for p in posts]
    if rows:
        stmt = sqlite_insert(Post).values(rows)
        conn.execute(
            stmt.on_conflict_do_update(
                index_elements=[Post.id],
                set_={
                    Post.text: stmt.excluded.text,
                    Post.topic_id: None,
                    Post.topic_label: None,
                },
            )
        )
        conn.commit()


def write_topic_assignments_for_claims(conn, clusters):
    """Write Topic rows and update each claim's topic_id (outliers stay unassigned)."""
    for topic_id, info in clusters.items():
        if topic_id == OUTLIER_TOPIC_ID:
            continue
        tid = str(topic_id)
        label = info[Field.LABEL]
        polarity_target = info[Field.POLARITY_TARGET]
        centroid = info.get(Field.CENTROID)
        centroid_bytes = (
            np.asarray(centroid, dtype=np.float32).tobytes()
            if centroid is not None
            else None
        )
        topic = conn.get(Topic, tid)
        if topic is None:
            conn.add(
                Topic(
                    id=tid,
                    label=label,
                    polarity_target=polarity_target,
                    centroid=centroid_bytes,
                )
            )
        else:
            topic.label = label
            topic.polarity_target = polarity_target
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
    """Drop topics, clusters, and statements; clear topic ids on existing claims."""
    for model in (
        RepresentativeStatement,
        ArgumentCluster,
    ):
        conn.query(model).delete(synchronize_session=False)
    conn.query(ArgumentInstance).update(
        {
            ArgumentInstance.topic_id: None,
            ArgumentInstance.cluster_id: None,
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


def reset_argument_graph_state(conn, topic_ids):
    """Drop clusters and statements for these topics; reset claim cluster_id and polarity."""
    if not topic_ids:
        return

    conn.query(ArgumentInstance).filter(ArgumentInstance.topic_id.in_(topic_ids)).update(
        {ArgumentInstance.cluster_id: None, ArgumentInstance.polarity: None},
        synchronize_session=False,
    )
    for model in (RepresentativeStatement, ArgumentCluster):
        conn.query(model).filter(model.topic_id.in_(topic_ids)).delete(
            synchronize_session=False
        )
    conn.commit()


def clear_argument_clusters(conn, topic_ids):
    """Remove cluster-dependent data while preserving extracted arguments."""
    if not topic_ids:
        return

    conn.query(ArgumentInstance).filter(ArgumentInstance.topic_id.in_(topic_ids)).update(
        {ArgumentInstance.cluster_id: None},
        synchronize_session=False,
    )
    for model in (RepresentativeStatement, ArgumentCluster):
        conn.query(model).filter(model.topic_id.in_(topic_ids)).delete(
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
        ArgumentCluster,
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


def assign_instance_to_cluster(conn, instance_id, cluster_id):
    instance = conn.get(ArgumentInstance, instance_id)
    if instance is not None:
        instance.cluster_id = cluster_id


# --- Argument clusters ---


def create_argument_cluster(conn, cluster_id, polarity, topic_id):
    conn.add(
        ArgumentCluster(id=cluster_id, polarity=polarity, topic_id=topic_id, count=1)
    )
    conn.flush()


def update_cluster_centroids(conn, centroids):
    """Persist centroid arrays + counts back to the DB."""
    for cid, cdata in centroids.items():
        cluster = conn.get(ArgumentCluster, cid)
        if cluster is not None:
            cluster.centroid = np.asarray(
                cdata[Field.CENTROID], dtype=np.float32
            ).tobytes()
            cluster.count = cdata[Field.COUNT]
    conn.commit()


def replace_representative_statements(conn, layer, scope_id, topic_id, polarity, rows):
    conn.query(RepresentativeStatement).filter(
        RepresentativeStatement.layer == layer,
        RepresentativeStatement.scope_id == scope_id,
    ).delete()
    for row in rows:
        represented_ids = row[Field.REPRESENTED_IDS]
        conn.add(
            RepresentativeStatement(
                id=f"rs_{uuid.uuid4().hex[:12]}",
                layer=layer,
                scope_id=scope_id,
                topic_id=topic_id,
                polarity=polarity,
                round_index=row[Field.ROUND_INDEX],
                statement=row[Field.STATEMENT],
                represented_ids=json.dumps(represented_ids),
                represented_count=len(represented_ids),
            )
        )
    conn.commit()
