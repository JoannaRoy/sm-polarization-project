import { Suspense, useState } from "react";

import { ErrorBoundary } from "./components/ErrorBoundary";
import { Spinner } from "./components/Spinner";
import { Generate } from "./pages/Generate";
import { Graph } from "./pages/Graph";

type Tab = "generate" | "graph";

export function App() {
  const [tab, setTab] = useState<Tab>("generate");

  return (
    <>
      <header className="app-header">
        <h1>Argument Graph</h1>
        <p>Generate counter-arguments for a post or explore the existing graph.</p>
      </header>

      <nav className="tabbar" aria-label="Pages">
        <button
          className={tab === "generate" ? "active" : ""}
          onClick={() => setTab("generate")}
        >
          Generate
        </button>
        <button
          className={tab === "graph" ? "active" : ""}
          onClick={() => setTab("graph")}
        >
          Graph
        </button>
      </nav>

      {tab === "generate" ? (
        <Generate />
      ) : (
        <ErrorBoundary
          fallback={(error, reset) => (
            <div className="page">
              <section className="panel">
                <div className="error">{error.message}</div>
                <button style={{ marginTop: 12 }} onClick={reset}>
                  Try again
                </button>
              </section>
            </div>
          )}
        >
          <Suspense
            fallback={
              <div className="page">
                <section className="panel">
                  <Spinner label="Loading topics..." />
                </section>
              </div>
            }
          >
            <Graph />
          </Suspense>
        </ErrorBoundary>
      )}
    </>
  );
}
