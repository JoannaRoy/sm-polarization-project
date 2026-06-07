"""Read-only queries against the pipeline database."""

from db.models import ArgumentInstance, RepresentativeStatement, SubTopic, Topic


def get_topics(conn):
    return conn.query(Topic).all()


def get_topic(conn, topic_id):
    return conn.get(Topic, topic_id)


def get_sub_topics_for_topic(conn, topic_id):
    return (
        conn.query(SubTopic)
        .filter(SubTopic.topic_id == topic_id)
        .order_by(SubTopic.id)
        .all()
    )


def get_unassigned_instances_for_topic(conn, topic_id):
    return (
        conn.query(ArgumentInstance)
        .filter(
            ArgumentInstance.sub_topic_id == None,  # noqa: E711
            ArgumentInstance.topic_id == topic_id,
        )
        .all()
    )


def get_polarity_slate(conn, sub_topic_id, polarity):
    """Return the slate statements (round-ordered) for a (sub_topic, polarity) bucket."""
    rows = (
        conn.query(RepresentativeStatement)
        .filter(
            RepresentativeStatement.sub_topic_id == sub_topic_id,
            RepresentativeStatement.polarity == polarity,
        )
        .order_by(RepresentativeStatement.round_index)
        .all()
    )
    return [row.statement for row in rows]
