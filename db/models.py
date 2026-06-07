"""ORM models and field enums."""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Field(StrEnum):
    ID = "id"
    TEXT = "text"
    LABEL = "label"
    POLARITY_TARGET = "polarity_target"
    LAYER = "layer"
    SCOPE_ID = "scope_id"
    POLARITY = "polarity"
    TOPIC_ID = "topic_id"
    TOPIC_LABEL = "topic_label"
    TOPIC_SENTENCE = "topic_sentence"
    POST_TOPIC = "post_topic"
    CLAIMS = "claims"
    CLAIM_IDS = "claim_ids"
    CLUSTER_ID = "cluster_id"
    POST_ID = "post_id"
    POST_IDS = "post_ids"
    CENTROID = "centroid"
    COUNT = "count"
    ROUND_INDEX = "round_index"
    STATEMENT = "statement"
    PARAGRAPH = "paragraph"
    REPRESENTED_IDS = "represented_ids"
    REPRESENTED_COUNT = "represented_count"
    POST_ARGUMENTS = "post_arguments"
    OTHER_ARGUMENTS = "other_arguments"


class Polarity(StrEnum):
    FOR = "for"
    AGAINST = "against"


class StatementLayer(StrEnum):
    ARGUMENT_CLUSTER = "argument_cluster"
    POLARITY = "polarity"


PolarityType = Enum(
    Polarity,
    name="polarity",
    values_callable=lambda e: [m.value for m in e],
)

StatementLayerType = Enum(
    StatementLayer,
    name="statement_layer",
    values_callable=lambda e: [m.value for m in e],
)


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    polarity_target: Mapped[str] = mapped_column(Text, nullable=False)
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)

    posts: Mapped[list["Post"]] = relationship(back_populates="topic")
    arguments: Mapped[list["ArgumentInstance"]] = relationship(back_populates="topic")
    clusters: Mapped[list["ArgumentCluster"]] = relationship(back_populates="topic")
    representative_statements: Mapped[list["RepresentativeStatement"]] = relationship(
        back_populates="topic"
    )


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    topic_id: Mapped[str | None] = mapped_column(ForeignKey("topics.id"))
    topic_label: Mapped[str | None] = mapped_column(String)
    claims_extracted_at: Mapped[datetime | None] = mapped_column(DateTime)

    topic: Mapped["Topic | None"] = relationship(back_populates="posts")
    arguments: Mapped[list["ArgumentInstance"]] = relationship(back_populates="post")


class ArgumentInstance(Base):
    __tablename__ = "argument_instances"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    topic_sentence: Mapped[str | None] = mapped_column(Text)
    polarity: Mapped[Polarity | None] = mapped_column(PolarityType)
    cluster_id: Mapped[str | None] = mapped_column(ForeignKey("argument_clusters.id"))
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.id"), nullable=False)
    topic_id: Mapped[str | None] = mapped_column(ForeignKey("topics.id"))

    topic: Mapped["Topic | None"] = relationship(back_populates="arguments")
    post: Mapped["Post"] = relationship(back_populates="arguments")
    cluster: Mapped["ArgumentCluster | None"] = relationship(back_populates="instances")


class ArgumentCluster(Base):
    __tablename__ = "argument_clusters"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    polarity: Mapped[Polarity] = mapped_column(PolarityType, nullable=False)
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id"), nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=0)
    centroid: Mapped[bytes | None] = mapped_column(LargeBinary)

    topic: Mapped["Topic"] = relationship(back_populates="clusters")
    instances: Mapped[list["ArgumentInstance"]] = relationship(back_populates="cluster")


class RepresentativeStatement(Base):
    """One generated GEN/DISC slate statement for a topic-scoped layer."""

    __tablename__ = "representative_statements"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    layer: Mapped[StatementLayer] = mapped_column(StatementLayerType, nullable=False)
    scope_id: Mapped[str] = mapped_column(String, nullable=False)
    topic_id: Mapped[str] = mapped_column(ForeignKey("topics.id"), nullable=False)
    polarity: Mapped[Polarity | None] = mapped_column(PolarityType)
    round_index: Mapped[int] = mapped_column(Integer, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    represented_ids: Mapped[str] = mapped_column(Text, nullable=False)
    represented_count: Mapped[int] = mapped_column(Integer, nullable=False)

    topic: Mapped["Topic"] = relationship(back_populates="representative_statements")
