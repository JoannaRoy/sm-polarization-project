import { use, useState } from "react";

import {
  fetchTopic,
  fetchTopics,
  type TopicSummary,
} from "../api";
import { EmptyState } from "../components/EmptyState";
import {
  GraphView,
  type DetailState,
  type NodeOffset,
} from "../components/GraphView";

let topicsPromise: Promise<TopicSummary[]> = fetchTopics();

export function Graph() {
  const [, bumpVersion] = useState(0);
  const topics = use(topicsPromise);

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [details, setDetails] = useState<Map<string, DetailState>>(new Map());
  const [offsets, setOffsets] = useState<Map<string, NodeOffset>>(new Map());
  const [highlightId, setHighlightId] = useState<string | null>(null);

  const ensureTopic = async (topicId: string): Promise<DetailState> => {
    const existing = details.get(topicId);
    if (existing && existing.kind !== "error") return existing;
    setDetails((prev) => new Map(prev).set(topicId, { kind: "loading" }));
    try {
      const data = await fetchTopic(topicId);
      const ready: DetailState = { kind: "ready", data };
      setDetails((prev) => new Map(prev).set(topicId, ready));
      return ready;
    } catch (err) {
      const error: DetailState = {
        kind: "error",
        message: (err as Error).message,
      };
      setDetails((prev) => new Map(prev).set(topicId, error));
      return error;
    }
  };

  const toggle = async (nodeId: string, topicId?: string) => {
    setHighlightId(nodeId);
    if (topicId) {
      const result = await ensureTopic(topicId);
      if (result.kind !== "ready") return;
    }
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
  };

  const collapseAll = () => {
    setExpanded(new Set());
    setHighlightId(null);
  };

  const resetLayout = () => {
    setOffsets(new Map());
  };

  const moveNode = (nodeId: string, offset: NodeOffset) => {
    setOffsets((prev) => new Map(prev).set(nodeId, offset));
  };

  const refresh = () => {
    topicsPromise = fetchTopics();
    setDetails(new Map());
    setExpanded(new Set());
    setOffsets(new Map());
    setHighlightId(null);
    bumpVersion((v) => v + 1);
  };

  if (topics.length === 0) {
    return (
      <div className="page">
        <section className="panel">
          <EmptyState>
            No topics in the database yet. Run the topic-clustering stage.
          </EmptyState>
        </section>
      </div>
    );
  }

  const totalArguments = topics.reduce((sum, t) => sum + t.argument_count, 0);
  const totalClusters = topics.reduce((sum, t) => sum + t.cluster_count, 0);
  const readyTopics = topics.filter(
    (t) => t.cluster_count > 0 && t.has_polarity_slates,
  ).length;

  return (
    <div className="page">
      <section className="panel">
        <div className="graph-header">
          <div>
            <h2 style={{ margin: 0 }}>Argument graph</h2>
            <p className="muted" style={{ margin: "4px 0 0" }}>
              Click any node to expand the next layer. Click again to collapse.
              Posts that match none of these topics get an "outlier" response.
            </p>
          </div>
          <div className="row">
            <button onClick={resetLayout} disabled={offsets.size === 0}>
              Reset layout
            </button>
            <button onClick={collapseAll} disabled={expanded.size === 0}>
              Collapse all
            </button>
            <button onClick={refresh}>Refresh</button>
          </div>
        </div>
        <div className="summary-banner" style={{ marginTop: 14 }}>
          <div className="stat">
            <strong>
              {readyTopics}/{topics.length}
            </strong>
            <span>topics ready</span>
          </div>
          <div className="stat">
            <strong>{totalClusters}</strong>
            <span>clusters</span>
          </div>
          <div className="stat">
            <strong>{totalArguments}</strong>
            <span>arguments</span>
          </div>
        </div>
        <div className="legend">
          <span className="legend-item">
            <span className="legend-dot topic" /> topic
          </span>
          <span className="legend-item">
            <span className="legend-dot polarity for" /> for
          </span>
          <span className="legend-item">
            <span className="legend-dot polarity against" /> against
          </span>
          <span className="legend-item">
            <span className="legend-dot cluster" /> cluster
          </span>
          <span className="legend-item">
            <span className="legend-dot argument" /> argument
          </span>
        </div>
      </section>

      <section className="panel" style={{ padding: 0 }}>
        <GraphView
          topics={topics}
          details={details}
          expanded={expanded}
          offsets={offsets}
          highlightId={highlightId}
          onToggle={(nodeId, _kind, topicId) => toggle(nodeId, topicId)}
          onMove={moveNode}
        />
      </section>
    </div>
  );
}
