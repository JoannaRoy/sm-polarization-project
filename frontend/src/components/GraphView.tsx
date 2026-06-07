import { useRef } from "react";

import type {
  Argument,
  Polarity,
  RepresentativeStatement,
  SubTopicDetail,
  TopicDetail,
  TopicSummary,
} from "../api";

type NodeKind = "root" | "topic" | "subtopic" | "polarity" | "argument";

interface VNode {
  id: string;
  kind: NodeKind;
  label: string;
  sub?: string;
  polarity?: Polarity;
  descriptive?: boolean;
  loading?: boolean;
  error?: string;
  children: () => VNode[];
}

interface Pos {
  node: VNode;
  x: number;
  y: number;
  depth: number;
}

interface Edge {
  from: Pos;
  to: Pos;
}

export type DetailState =
  | { kind: "loading" }
  | { kind: "ready"; data: TopicDetail }
  | { kind: "error"; message: string };

// Heights are conservative minimums; CSS enforces min-height so the visible
// rect is always >= these values. Edge endpoints will then always fall inside
// the rendered card.
const NODE_SIZES: Record<NodeKind, { width: number; height: number }> = {
  root: { width: 220, height: 60 },
  topic: { width: 260, height: 60 },
  subtopic: { width: 380, height: 140 },
  polarity: { width: 380, height: 160 },
  argument: { width: 320, height: 80 },
};
const TOPIC_RADIUS = 380;
const DEPTH_SPACING = 520;
// Sub-topics under a topic alternate their radial distance from the topic
// node by this amount. The zig-zag means adjacent sub-topics are spread
// along the topic's ray instead of stacked perpendicular, so we can pack
// them tighter along the perpendicular axis without nodes overlapping.
const SUBTOPIC_RADIAL_STAGGER = 400;
const PERP_SPACING = 120;
const MAX_RAY_DEPTH = 3;
const CANVAS_SIZE = 12000;
const CENTER = CANVAS_SIZE / 2;
const ROOT_ID = "root";
const DRAG_THRESHOLD = 4;
const MIN_SCALE = 0.15;
const MAX_SCALE = 2.5;
const SCALE_STEP = 1.15;
const WHEEL_ZOOM_SENSITIVITY = 0.01;

export type NodeOffset = { dx: number; dy: number };

function buildArgumentNode(parentId: string, arg: Argument): VNode {
  // Namespace the node id by parent so the same arg id can never collide if
  // it ever shows up under two parents (defensive — collectEdges keys by
  // node.id, and collisions would orphan edges).
  return {
    id: `${parentId}/arg:${arg.id}`,
    kind: "argument",
    label: arg.text,
    sub: `post ${arg.post_id}`,
    children: () => [],
  };
}

function formatSlate(slate: RepresentativeStatement[], emptyLabel: string): string {
  if (slate.length === 0) return emptyLabel;
  const ordered = [...slate].sort((a, b) => a.round_index - b.round_index);
  return ordered.map((row, i) => `${i + 1}. ${row.statement}`).join("\n");
}

const POLARITY_LABEL: Record<Polarity, string> = {
  agree: "AGREE",
  disagree: "DISAGREE",
};

function buildPolarityNode(
  subTopicId: string,
  polarity: Polarity,
  subTopic: SubTopicDetail,
): VNode {
  const slate =
    polarity === "agree" ? subTopic.agree_slate : subTopic.disagree_slate;
  const claims = subTopic.arguments.filter((a) => a.polarity === polarity);
  const polarityId = `pol:${subTopicId}:${polarity}`;
  return {
    id: polarityId,
    kind: "polarity",
    polarity,
    label: POLARITY_LABEL[polarity],
    sub: formatSlate(slate, "(no slate yet)"),
    children: () => claims.map((c) => buildArgumentNode(polarityId, c)),
  };
}

function buildSubTopicNode(topicId: string, subTopic: SubTopicDetail): VNode {
  const descriptive = subTopic.polarity_target === null;
  const claimCount = `${subTopic.count} claim${subTopic.count === 1 ? "" : "s"}`;
  const sub = descriptive
    ? `descriptive · ${claimCount}`
    : `${subTopic.polarity_target}\n${claimCount}`;
  return {
    id: `sub:${topicId}:${subTopic.id}`,
    kind: "subtopic",
    label: subTopic.label,
    sub,
    descriptive,
    children: () =>
      descriptive
        ? []
        : [
            buildPolarityNode(subTopic.id, "agree", subTopic),
            buildPolarityNode(subTopic.id, "disagree", subTopic),
          ],
  };
}

function subTopicSubtitle(
  topic: TopicSummary,
  detail: DetailState | undefined,
): string {
  const args = `${topic.argument_count} argument${topic.argument_count === 1 ? "" : "s"}`;
  if (detail?.kind !== "ready") {
    const n = topic.sub_topic_count;
    return `${n} sub-topic${n === 1 ? "" : "s"} · ${args}`;
  }
  const shown = detail.data.sub_topics.length;
  const total = detail.data.total_sub_topic_count;
  if (shown < total) {
    return `showing top ${shown} of ${total} sub-topics · ${args}`;
  }
  return `${total} sub-topic${total === 1 ? "" : "s"} · ${args}`;
}

function buildTopicNode(
  topic: TopicSummary,
  detail: DetailState | undefined,
): VNode {
  return {
    id: `topic:${topic.id}`,
    kind: "topic",
    label: topic.label,
    sub: subTopicSubtitle(topic, detail),
    loading: detail?.kind === "loading",
    error: detail?.kind === "error" ? detail.message : undefined,
    children: () => {
      if (detail?.kind !== "ready") return [];
      return detail.data.sub_topics.map((s) => buildSubTopicNode(topic.id, s));
    },
  };
}

function buildRoot(
  topics: TopicSummary[],
  details: Map<string, DetailState>,
): VNode {
  return {
    id: ROOT_ID,
    kind: "root",
    label: "Argument Graph",
    sub: `${topics.length} topic${topics.length === 1 ? "" : "s"}`,
    children: () => topics.map((t) => buildTopicNode(t, details.get(t.id))),
  };
}

interface LocalPos {
  node: VNode;
  lx: number;
  ly: number;
  depth: number;
}

function layoutRay(
  node: VNode,
  localDepth: number,
  expanded: Set<string>,
): { positions: LocalPos[]; height: number } {
  const lx = localDepth * DEPTH_SPACING;
  const children = node.children();
  if (!expanded.has(node.id) || children.length === 0 || localDepth >= MAX_RAY_DEPTH) {
    return { positions: [{ node, lx, ly: 0, depth: localDepth }], height: 1 };
  }
  const subtrees = children.map((c) => layoutRay(c, localDepth + 1, expanded));
  const totalHeight = subtrees.reduce((s, t) => s + t.height, 0);
  let cursor = -totalHeight / 2;
  const positions: LocalPos[] = [];
  const staggerChildren = localDepth === 0;
  for (let i = 0; i < children.length; i++) {
    const subtree = subtrees[i];
    const offsetY = cursor + subtree.height / 2;
    cursor += subtree.height;
    const radialBump = staggerChildren && i % 2 === 1 ? SUBTOPIC_RADIAL_STAGGER : 0;
    for (const p of subtree.positions) {
      positions.push({
        node: p.node,
        lx: p.lx + radialBump,
        ly: p.ly + offsetY,
        depth: p.depth,
      });
    }
  }
  positions.push({ node, lx, ly: 0, depth: localDepth });
  return { positions, height: totalHeight };
}

function buildPositions(root: VNode, expanded: Set<string>): Pos[] {
  const positions: Pos[] = [
    { node: root, x: CENTER, y: CENTER, depth: 0 },
  ];
  const topicNodes = root.children();
  const n = topicNodes.length;
  if (n === 0) return positions;
  for (let i = 0; i < n; i++) {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const subtree = layoutRay(topicNodes[i], 0, expanded);
    for (const p of subtree.positions) {
      const r = TOPIC_RADIUS + p.lx;
      const perp = p.ly * PERP_SPACING;
      positions.push({
        node: p.node,
        x: CENTER + r * cos - perp * sin,
        y: CENTER + r * sin + perp * cos,
        depth: p.depth + 1,
      });
    }
  }
  return positions;
}

function collectEdges(positions: Pos[], expanded: Set<string>): Edge[] {
  const byId = new Map(positions.map((p) => [p.node.id, p]));
  const edges: Edge[] = [];
  for (const p of positions) {
    const isExpanded = p.node.kind === "root" || expanded.has(p.node.id);
    if (!isExpanded) continue;
    for (const child of p.node.children()) {
      const cp = byId.get(child.id);
      if (cp) edges.push({ from: p, to: cp });
    }
  }
  return edges;
}

function cumulativeOffset(
  nodeId: string,
  offsets: Map<string, NodeOffset>,
  parentById: Map<string, string>,
): NodeOffset {
  let dx = 0;
  let dy = 0;
  let id: string | undefined = nodeId;
  while (id) {
    const o = offsets.get(id);
    if (o) {
      dx += o.dx;
      dy += o.dy;
    }
    id = parentById.get(id);
  }
  return { dx, dy };
}

function effectivePos(
  pos: Pos,
  offsets: Map<string, NodeOffset>,
  parentById: Map<string, string>,
): { x: number; y: number } {
  const o = cumulativeOffset(pos.node.id, offsets, parentById);
  return { x: pos.x + o.dx, y: pos.y + o.dy };
}

function rectEdgePoint(
  inside: { x: number; y: number },
  toward: { x: number; y: number },
  halfW: number,
  halfH: number,
): { x: number; y: number } {
  const dx = toward.x - inside.x;
  const dy = toward.y - inside.y;
  if (dx === 0 && dy === 0) return inside;
  const tx = Math.abs(dx) > 1e-6 ? halfW / Math.abs(dx) : Infinity;
  const ty = Math.abs(dy) > 1e-6 ? halfH / Math.abs(dy) : Infinity;
  const t = Math.min(tx, ty);
  return { x: inside.x + t * dx, y: inside.y + t * dy };
}

function edgePath(
  edge: Edge,
  offsets: Map<string, NodeOffset>,
  parentById: Map<string, string>,
): string {
  const from = effectivePos(edge.from, offsets, parentById);
  const to = effectivePos(edge.to, offsets, parentById);
  const fromSize = NODE_SIZES[edge.from.node.kind];
  const toSize = NODE_SIZES[edge.to.node.kind];
  const start = rectEdgePoint(from, to, fromSize.width / 2, fromSize.height / 2);
  const end = rectEdgePoint(to, from, toSize.width / 2, toSize.height / 2);
  return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
}

function edgeClass(edge: Edge): string {
  const polarity = edge.to.node.polarity;
  if (polarity === "agree") return "graph-edge agree";
  if (polarity === "disagree") return "graph-edge disagree";
  return "graph-edge";
}

interface DragState {
  pointerId: number;
  startX: number;
  startY: number;
  scrollLeft: number;
  scrollTop: number;
}

export function GraphView({
  topics,
  details,
  expanded,
  offsets,
  onToggle,
  onMove,
  highlightId,
}: {
  topics: TopicSummary[];
  details: Map<string, DetailState>;
  expanded: Set<string>;
  offsets: Map<string, NodeOffset>;
  onToggle: (nodeId: string, kind: NodeKind, topicId?: string) => void;
  onMove: (nodeId: string, offset: NodeOffset) => void;
  highlightId: string | null;
}) {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const sizerRef = useRef<HTMLDivElement | null>(null);
  const innerRef = useRef<HTMLDivElement | null>(null);
  const percentRef = useRef<HTMLButtonElement | null>(null);
  const didScroll = useRef(false);
  const dragState = useRef<DragState | null>(null);
  const scaleRef = useRef(0.35);

  const root = buildRoot(topics, details);
  const positions = buildPositions(root, expanded);
  const edges = collectEdges(positions, expanded);
  
  const parentById = new Map<string, string>();
  for (const e of edges) parentById.set(e.to.node.id, e.from.node.id);

  const writeScaleToDom = (next: number) => {
    const sizer = sizerRef.current;
    const inner = innerRef.current;
    if (sizer) {
      sizer.style.width = `${CANVAS_SIZE * next}px`;
      sizer.style.height = `${CANVAS_SIZE * next}px`;
    }
    if (inner) {
      inner.style.transform = `scale(${next})`;
    }
    if (percentRef.current) {
      percentRef.current.textContent = `${Math.round(next * 100)}%`;
    }
  };

  const applyZoom = (factor: number, anchorX?: number, anchorY?: number) => {
    const el = canvasRef.current;
    if (!el) return;
    const current = scaleRef.current;
    const next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, current * factor));
    if (next === current) return;
    const ax = anchorX ?? el.clientWidth / 2;
    const ay = anchorY ?? el.clientHeight / 2;
    const worldX = (el.scrollLeft + ax) / current;
    const worldY = (el.scrollTop + ay) / current;
    scaleRef.current = next;
    writeScaleToDom(next);
    el.scrollLeft = worldX * next - ax;
    el.scrollTop = worldY * next - ay;
  };

  const resetZoom = () => {
    const el = canvasRef.current;
    scaleRef.current = 0.35;
    writeScaleToDom(0.35);
    if (!el) return;
    el.scrollLeft = (CANVAS_SIZE * 0.35 - el.clientWidth) / 2;
    el.scrollTop = (CANVAS_SIZE * 0.35 - el.clientHeight) / 2;
  };

  const wheelListener = useRef<((e: WheelEvent) => void) | null>(null);
  if (!wheelListener.current) {
    wheelListener.current = (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const el = canvasRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const factor = Math.exp(-event.deltaY * WHEEL_ZOOM_SENSITIVITY);
      applyZoomRef.current(factor, event.clientX - rect.left, event.clientY - rect.top);
    };
  }
  const applyZoomRef = useRef(applyZoom);
  applyZoomRef.current = applyZoom;

  const setCanvasRef = (el: HTMLDivElement | null) => {
    const prev = canvasRef.current;
    const handler = wheelListener.current!;
    if (prev && prev !== el) {
      prev.removeEventListener("wheel", handler);
    }
    canvasRef.current = el;
    if (el && el !== prev) {
      el.addEventListener("wheel", handler, { passive: false });
    }
    if (el && !didScroll.current) {
      el.scrollLeft = (el.scrollWidth - el.clientWidth) / 2;
      el.scrollTop = (el.scrollHeight - el.clientHeight) / 2;
      didScroll.current = true;
    }
  };

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest(".graph-node")) return;
    if (target.closest(".graph-zoom-controls")) return;
    if (event.button !== 0 && event.pointerType === "mouse") return;
    const el = event.currentTarget;
    el.setPointerCapture(event.pointerId);
    el.classList.add("dragging");
    dragState.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: el.scrollLeft,
      scrollTop: el.scrollTop,
    };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const el = event.currentTarget;
    el.scrollLeft = drag.scrollLeft - (event.clientX - drag.startX);
    el.scrollTop = drag.scrollTop - (event.clientY - drag.startY);
  };

  const endDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const el = event.currentTarget;
    if (el.hasPointerCapture(event.pointerId)) {
      el.releasePointerCapture(event.pointerId);
    }
    el.classList.remove("dragging");
    dragState.current = null;
  };

  return (
    <div className="graph-wrap">
      <div
        className="graph-canvas"
        ref={setCanvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <div
          ref={sizerRef}
          className="graph-canvas-sizer"
          style={{
            width: CANVAS_SIZE * scaleRef.current,
            height: CANVAS_SIZE * scaleRef.current,
          }}
        >
          <div
            ref={innerRef}
            className="graph-canvas-inner"
            style={{
              width: CANVAS_SIZE,
              height: CANVAS_SIZE,
              transform: `scale(${scaleRef.current})`,
            }}
          >
            <svg
              className="graph-edges"
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              viewBox={`0 0 ${CANVAS_SIZE} ${CANVAS_SIZE}`}
            >
              {edges.map((edge, i) => (
                <path
                  key={i}
                  className={edgeClass(edge)}
                  d={edgePath(edge, offsets, parentById)}
                />
              ))}
            </svg>
            {positions.map((p) => {
              const topicId =
                p.node.kind === "topic" ? p.node.id.slice("topic:".length) : undefined;
              // Own offset drives drag deltas; cumulative offset (own + all
              // ancestors) drives where the node is actually drawn so children
              // follow when an ancestor is dragged.
              const ownOffset = offsets.get(p.node.id) ?? { dx: 0, dy: 0 };
              const cum = cumulativeOffset(p.node.id, offsets, parentById);
              return (
                <NodeCard
                  key={p.node.id}
                  node={p.node}
                  x={p.x + cum.dx}
                  y={p.y + cum.dy}
                  width={NODE_SIZES[p.node.kind].width}
                  baseOffset={ownOffset}
                  open={expanded.has(p.node.id)}
                  highlighted={p.node.id === highlightId}
                  scaleRef={scaleRef}
                  onToggle={() => onToggle(p.node.id, p.node.kind, topicId)}
                  onMove={(next) => onMove(p.node.id, next)}
                />
              );
            })}
          </div>
        </div>
      </div>
      <div className="graph-zoom-controls" onPointerDown={(e) => e.stopPropagation()}>
        <button type="button" onClick={() => applyZoom(1 / SCALE_STEP)} title="Zoom out">
          −
        </button>
        <button
          type="button"
          ref={percentRef}
          onClick={resetZoom}
          title="Reset zoom"
        >
          {Math.round(scaleRef.current * 100)}%
        </button>
        <button type="button" onClick={() => applyZoom(SCALE_STEP)} title="Zoom in">
          +
        </button>
      </div>
    </div>
  );
}

interface NodeDragState {
  pointerId: number;
  startClientX: number;
  startClientY: number;
  baseDx: number;
  baseDy: number;
  moved: boolean;
}

function NodeCard({
  node,
  x,
  y,
  width,
  baseOffset,
  open,
  highlighted,
  scaleRef,
  onToggle,
  onMove,
}: {
  node: VNode;
  x: number;
  y: number;
  width: number;
  baseOffset: NodeOffset;
  open: boolean;
  highlighted: boolean;
  scaleRef: React.MutableRefObject<number>;
  onToggle: () => void;
  onMove: (offset: NodeOffset) => void;
}) {
  const dragRef = useRef<NodeDragState | null>(null);
  const polarityClass = node.polarity ? ` ${node.polarity}` : "";
  const descriptiveClass = node.descriptive ? " descriptive" : "";
  // Descriptive sub-topics have no children, so do not show a toggle.
  const isToggleable =
    node.kind !== "root" &&
    node.kind !== "argument" &&
    !(node.kind === "subtopic" && node.descriptive);
  const chevron = node.loading
    ? "…"
    : isToggleable
      ? open
        ? "−"
        : "+"
      : "";

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      baseDx: baseOffset.dx,
      baseDy: baseOffset.dy,
      moved: false,
    };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.startClientX;
    const dy = event.clientY - drag.startClientY;
    if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    drag.moved = true;
    const s = scaleRef.current || 1;
    onMove({ dx: drag.baseDx + dx / s, dy: drag.baseDy + dy / s });
  };

  const endDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const wasClick = !drag.moved;
    dragRef.current = null;
    if (wasClick && isToggleable) onToggle();
  };

  return (
    <div
      role={isToggleable ? "button" : undefined}
      tabIndex={isToggleable ? 0 : undefined}
      className={`graph-node ${node.kind}${polarityClass}${descriptiveClass}${open ? " open" : ""}${highlighted ? " highlighted" : ""}${isToggleable ? "" : " leaf"}`}
      style={{ left: x, top: y, width }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onKeyDown={(event) => {
        if (!isToggleable) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onToggle();
        }
      }}
      aria-expanded={isToggleable ? open : undefined}
      title={node.label}
    >
      <div className="graph-node-header">
        {isToggleable && <span className="graph-node-toggle">{chevron}</span>}
        <span className="graph-node-kind">{node.kind}</span>
      </div>
      <div className="graph-node-label">{node.label}</div>
      {node.sub && <div className="graph-node-sub">{node.sub}</div>}
      {node.error && <div className="graph-node-error">{node.error}</div>}
    </div>
  );
}
