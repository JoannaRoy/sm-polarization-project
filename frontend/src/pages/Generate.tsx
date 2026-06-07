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
          you can confirm the right framing, then weave the precomputed
          for/against statements into a response paragraph.
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
                    <span className="candidate-target">
                      {candidate.polarity_target}
                    </span>
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
          <p
            style={{
              margin: "12px 0 0",
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
            }}
          >
            {status.paragraph}
          </p>
        </section>
      )}
    </div>
  );
}
