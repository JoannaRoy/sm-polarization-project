"""Read-only queries against the pipeline database."""

from db.models import ArgumentInstance, RepresentativeStatement, StatementLayer, Topic


def get_topics(conn):
    return conn.query(Topic).all()


def get_topic(conn, topic_id):
    return conn.get(Topic, topic_id)


def get_unclustered_instances(conn, topic_id):
    return (
        conn.query(ArgumentInstance)
        .filter(
            ArgumentInstance.cluster_id == None,  # noqa: E711
            ArgumentInstance.topic_id == topic_id,
        )
        .all()
    )


def get_polarity_slate(conn, topic_id, polarity):
    """Return the polarity-layer slate statements (round-ordered) for a topic side."""
    rows = (
        conn.query(RepresentativeStatement)
        .filter(
            RepresentativeStatement.layer == StatementLayer.POLARITY,
            RepresentativeStatement.topic_id == topic_id,
            RepresentativeStatement.polarity == polarity,
        )
        .order_by(RepresentativeStatement.round_index)
        .all()
    )
    return [row.statement for row in rows]


def get_cluster_slate(conn, cluster_id):
    """Return the cluster-layer slate statements (round-ordered) for one cluster."""
    rows = (
        conn.query(RepresentativeStatement)
        .filter(
            RepresentativeStatement.layer == StatementLayer.ARGUMENT_CLUSTER,
            RepresentativeStatement.scope_id == cluster_id,
        )
        .order_by(RepresentativeStatement.round_index)
        .all()
    )
    return [row.statement for row in rows]
