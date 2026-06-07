import { useRef } from "react";

import type {
  Argument,
  Cluster,
  Polarity,
  RepresentativeStatement,
  TopicDetail,
  TopicSummary,
} from "../api";

type NodeKind = "root" | "topic" | "polarity" | "cluster" | "argument";

interface VNode {
  id: string;
  kind: NodeKind;
  label: string;
  sub?: string;
  polarity?: Polarity;
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
// (or exactly on) the rendered card, where the opaque background hides any
// overshoot. If text wraps and the card grows taller, the line just ends
// slightly inside the box, which still looks like it touches.
const NODE_SIZES: Record<NodeKind, { width: number; height: number }> = {
  root: { width: 220, height: 60 },
  topic: { width: 260, height: 60 },
  polarity: { width: 380, height: 140 },
  cluster: { width: 380, height: 160 },
  argument: { width: 320, height: 80 },
};
const TOPIC_RADIUS = 380;
const DEPTH_SPACING = 520;
const PERP_SPACING = 280;
const MAX_RAY_DEPTH = 3;
const CANVAS_SIZE = 5400;
const CENTER = CANVAS_SIZE / 2;
const ROOT_ID = "root";
const DRAG_THRESHOLD = 4;
const MIN_SCALE = 0.15;
const MAX_SCALE = 2.5;
const SCALE_STEP = 1.15;
const WHEEL_ZOOM_SENSITIVITY = 0.01;

export type NodeOffset = { dx: number; dy: number };

function buildArgumentNode(arg: Argument): VNode {
  return {
    id: `arg:${arg.id}`,
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

function buildClusterNode(cluster: Cluster): VNode {
  return {
    id: `cluster:${cluster.id}`,
    kind: "cluster",
    polarity: cluster.polarity,
    label: formatSlate(
      cluster.representative_statements,
      "(no cluster slate yet)",
    ),
    sub: `${cluster.arguments.length} argument${cluster.arguments.length === 1 ? "" : "s"}`,
    children: () => cluster.arguments.map(buildArgumentNode),
  };
}

function buildPolarityNode(
  topicId: string,
  polarity: Polarity,
  data: TopicDetail,
): VNode {
  const clusters = data.clusters.filter((c) => c.polarity === polarity);
  const slate =
    data.polarity_slates.find((s) => s.polarity === polarity)
      ?.representative_statements ?? [];
  const clusterCount = `${clusters.length} cluster${clusters.length === 1 ? "" : "s"}`;
  return {
    id: `pol:${topicId}:${polarity}`,
    kind: "polarity",
    polarity,
    label: polarity.toUpperCase(),
    sub: slate.length > 0 ? formatSlate(slate, clusterCount) : clusterCount,
    children: () => clusters.map(buildClusterNode),
  };
}

function buildTopicNode(
  topic: TopicSummary,
  detail: DetailState | undefined,
): VNode {
  return {
    id: `topic:${topic.id}`,
    kind: "topic",
    label: topic.label,
    sub: `${topic.cluster_count} clusters · ${topic.argument_count} arguments`,
    loading: detail?.kind === "loading",
    error: detail?.kind === "error" ? detail.message : undefined,
    children: () => {
      if (detail?.kind !== "ready") return [];
      return [
        buildPolarityNode(topic.id, "for", detail.data),
        buildPolarityNode(topic.id, "against", detail.data),
      ];
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

// Lay out an expanded topic's subtree as a horizontal tree in local coords:
// lx grows outward along the ray (depth), ly is perpendicular sibling spread.
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
  for (let i = 0; i < children.length; i++) {
    const subtree = subtrees[i];
    const offsetY = cursor + subtree.height / 2;
    cursor += subtree.height;
    for (const p of subtree.positions) {
      positions.push({ node: p.node, lx: p.lx, ly: p.ly + offsetY, depth: p.depth });
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

function effectivePos(
  pos: Pos,
  offsets: Map<string, NodeOffset>,
): { x: number; y: number } {
  const o = offsets.get(pos.node.id);
  return o ? { x: pos.x + o.dx, y: pos.y + o.dy } : { x: pos.x, y: pos.y };
}

// Where a line from `inside` toward `toward` exits the rect centered on `inside`.
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
): string {
  const from = effectivePos(edge.from, offsets);
  const to = effectivePos(edge.to, offsets);
  const fromSize = NODE_SIZES[edge.from.node.kind];
  const toSize = NODE_SIZES[edge.to.node.kind];
  const start = rectEdgePoint(from, to, fromSize.width / 2, fromSize.height / 2);
  const end = rectEdgePoint(to, from, toSize.width / 2, toSize.height / 2);
  return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
}

function edgeClass(edge: Edge): string {
  const polarity = edge.to.node.polarity;
  if (polarity === "for") return "graph-edge for";
  if (polarity === "against") return "graph-edge against";
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
  // Scale lives in a ref, not state: we apply it directly to the DOM in a
  // single synchronous block alongside the scroll-position correction so the
  // zoom anchor stays locked under the cursor with no inter-frame jitter.
  const scaleRef = useRef(1);

  const root = buildRoot(topics, details);
  const positions = buildPositions(root, expanded);
  const edges = collectEdges(positions, expanded);

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
    scaleRef.current = 1;
    writeScaleToDom(1);
    if (!el) return;
    el.scrollLeft = (CANVAS_SIZE - el.clientWidth) / 2;
    el.scrollTop = (CANVAS_SIZE - el.clientHeight) / 2;
  };

  // Stable native wheel listener; reads scale from ref so identity never changes.
  const wheelListener = useRef<((e: WheelEvent) => void) | null>(null);
  if (!wheelListener.current) {
    wheelListener.current = (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const el = canvasRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      // Smooth, delta-proportional zoom so trackpad pinch + mouse wheel both feel calm.
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
                  d={edgePath(edge, offsets)}
                />
              ))}
            </svg>
            {positions.map((p) => {
              const topicId =
                p.node.kind === "topic" ? p.node.id.slice("topic:".length) : undefined;
              const offset = offsets.get(p.node.id) ?? { dx: 0, dy: 0 };
              return (
                <NodeCard
                  key={p.node.id}
                  node={p.node}
                  x={p.x + offset.dx}
                  y={p.y + offset.dy}
                  width={NODE_SIZES[p.node.kind].width}
                  baseOffset={offset}
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
  const isToggleable =
    node.kind !== "root" && node.kind !== "argument";
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
      className={`graph-node ${node.kind}${polarityClass}${open ? " open" : ""}${highlighted ? " highlighted" : ""}${isToggleable ? "" : " leaf"}`}
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
