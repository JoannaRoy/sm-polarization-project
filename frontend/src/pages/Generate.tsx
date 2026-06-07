import { useState } from "react";

import {
  TopicCandidate,
  fetchTopicResponse,
  matchPostTopics,
} from "../api";

type Status =
  | { kind: "idle" }
  | { kind: "matching" }
  | { kind: "candidates"; candidates: TopicCandidate[] }
  | {
      kind: "responding";
      candidates: TopicCandidate[];
      selected: TopicCandidate;
    }
  | {
      kind: "ready";
      candidates: TopicCandidate[];
      selected: TopicCandidate;
      paragraph: string;
    }
  | { kind: "error"; message: string };

export function Generate() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  const reset = () => {
    setText("");
    setStatus({ kind: "idle" });
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;
    setStatus({ kind: "matching" });
    try {
      const candidates = await matchPostTopics(trimmed);
      setStatus({ kind: "candidates", candidates });
    } catch (err) {
      setStatus({ kind: "error", message: (err as Error).message });
    }
  };

  const choose = async (
    candidates: TopicCandidate[],
    selected: TopicCandidate,
  ) => {
    setStatus({ kind: "responding", candidates, selected });
    try {
      const paragraph = await fetchTopicResponse(selected.id);
      setStatus({ kind: "ready", candidates, selected, paragraph });
    } catch (err) {
      setStatus({ kind: "error", message: (err as Error).message });
    }
  };

  const back = (candidates: TopicCandidate[]) => {
    setStatus({ kind: "candidates", candidates });
  };

  const showCandidates =
    status.kind === "candidates" ||
    status.kind === "responding" ||
    status.kind === "ready";

  return (
    <div className="page">
      <section className="panel">
        <h2>Generate response paragraph</h2>
        <p className="muted">
          Paste a post. We rank the closest topics by centroid similarity so
          you can confirm the right framing, then return every sub-topic under
          that topic along with its agree/disagree slates.
        </p>
        <form onSubmit={submit}>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="e.g. should i adopt a cat or a dog"
            aria-label="Post content"
            disabled={status.kind === "matching"}
          />
          <div className="row" style={{ marginTop: 12 }}>
            <button
              type="submit"
              className="active"
              disabled={status.kind === "matching" || !text.trim()}
            >
              {status.kind === "matching" ? "Matching topics..." : "Find topics"}
            </button>
            {status.kind !== "idle" && status.kind !== "matching" && (
              <button type="button" onClick={reset}>
                Start over
              </button>
            )}
          </div>
        </form>
        {status.kind === "error" && (
          <div className="error" style={{ marginTop: 14 }}>
            {status.message}
          </div>
        )}
      </section>

      {showCandidates && (
        <section className="panel">
          <h2>Closest topics</h2>
          <p className="muted">
            Pick the topic that best fits your post. The score is the cosine
            similarity between your post and each topic's centroid.
          </p>
          <ul className="candidate-list">
            {status.candidates.map((candidate) => {
              const isSelected =
                (status.kind === "responding" ||
                  status.kind === "ready") &&
                status.selected.id === candidate.id;
              return (
                <li key={candidate.id}>
                  <button
                    type="button"
                    className={`candidate${isSelected ? " selected" : ""}`}
                    onClick={() => choose(status.candidates, candidate)}
                    disabled={status.kind === "responding"}
                  >
                    <div className="candidate-row">
                      <span className="candidate-label">{candidate.label}</span>
                      <span className="candidate-score">
                        {candidate.score.toFixed(3)}
                      </span>
                    </div>
                    {isSelected && status.kind === "responding" && (
                      <span className="muted" style={{ fontSize: 12 }}>
                        Loading response...
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {status.kind === "ready" && (
        <section className="panel">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2 style={{ margin: 0 }}>Response for {status.selected.label}</h2>
            <button type="button" onClick={() => back(status.candidates)}>
              Try a different topic
            </button>
          </div>
          <ResponseView text={status.paragraph} />
        </section>
      )}
    </div>
  );
}

interface SubTopicSection {
  label: string;
  target: string;
  agree: string[];
  disagree: string[];
}

interface ParsedResponse {
  intro: string;
  sections: SubTopicSection[];
  trailing: string;
}

function parseBullets(block: string): string[] {
  const lines = block.split("\n").slice(1);
  const items: string[] = [];
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("- ")) {
      items.push(line.slice(2).trim());
    } else if (line.startsWith("-")) {
      items.push(line.slice(1).trim());
    } else if (items.length > 0 && line.trim()) {
      items[items.length - 1] += " " + line.trim();
    }
  }
  return items.filter(Boolean);
}

function parseTopicResponse(text: string): ParsedResponse {
  const blocks = text
    .split(/\n\n+/)
    .map((b) => b.trim())
    .filter(Boolean);
  let intro = "";
  let trailing = "";
  const sections: SubTopicSection[] = [];
  let current: SubTopicSection | null = null;
  for (const block of blocks) {
    if (block.startsWith("Your post is about")) {
      intro = block;
      continue;
    }
    if (block.startsWith("Sub-topic:")) {
      const match = block.match(/^Sub-topic:\s*(.*?)\s*--\s*"(.*)"\s*$/s);
      const label = match ? match[1] : block.replace(/^Sub-topic:\s*/, "");
      const target = match ? match[2] : "";
      current = { label, target, agree: [], disagree: [] };
      sections.push(current);
      continue;
    }
    if (block.startsWith("Agree:") && current) {
      current.agree = parseBullets(block);
      continue;
    }
    if (block.startsWith("Disagree:") && current) {
      current.disagree = parseBullets(block);
      continue;
    }
    trailing = trailing ? `${trailing}\n\n${block}` : block;
  }
  return { intro, sections, trailing };
}

function ResponseView({ text }: { text: string }) {
  const parsed = parseTopicResponse(text);
  return (
    <div className="response-body">
      {parsed.intro && <p className="response-intro">{parsed.intro}</p>}
      {parsed.sections.length > 0 && (
        <ul className="subtopic-list">
          {parsed.sections.map((section, i) => (
            <SubTopicDropdown key={i} section={section} defaultOpen={i === 0} />
          ))}
        </ul>
      )}
      {parsed.trailing && <p className="muted response-trailing">{parsed.trailing}</p>}
    </div>
  );
}

function SubTopicDropdown({
  section,
  defaultOpen,
}: {
  section: SubTopicSection;
  defaultOpen: boolean;
}) {
  return (
    <li className="subtopic-item">
      <details open={defaultOpen}>
        <summary className="subtopic-summary">
          <span className="subtopic-chevron" aria-hidden>
            ▸
          </span>
          <div className="subtopic-summary-body">
            <div className="subtopic-label">{section.label}</div>
            {section.target && (
              <div className="subtopic-target">"{section.target}"</div>
            )}
          </div>
          <div className="subtopic-counts">
            {section.agree.length > 0 && (
              <span className="badge agree">{section.agree.length} agree</span>
            )}
            {section.disagree.length > 0 && (
              <span className="badge disagree">
                {section.disagree.length} disagree
              </span>
            )}
          </div>
        </summary>
        <div className="subtopic-body">
          {section.agree.length > 0 && (
            <Slate polarity="agree" items={section.agree} />
          )}
          {section.disagree.length > 0 && (
            <Slate polarity="disagree" items={section.disagree} />
          )}
        </div>
      </details>
    </li>
  );
}

function Slate({
  polarity,
  items,
}: {
  polarity: "agree" | "disagree";
  items: string[];
}) {
  return (
    <div className={`slate slate-${polarity}`}>
      <div className={`slate-header badge ${polarity}`}>{polarity}</div>
      <ul className="slate-list">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
