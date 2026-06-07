# Deployment

## Frontend

The React frontend lives in `frontend/`. In production the FastAPI container
builds it into `frontend/dist/` and serves it at `http://localhost:8000/`
alongside the API endpoints.

For local development against a running API:

```bash
# Terminal 1: start the API (already running inside docker compose).
docker compose up -d

# Terminal 2: run the Vite dev server with hot reload.
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

The dev server proxies `/topics`, `/topics/{id}`, `/match-post-topics`, and
`/topic-response/{id}` to the API on port 8000.

The frontend has two pages:

- **Generate** — paste a post, pick from the 3-5 closest topics (by centroid
  similarity), and get back the precomputed for/against slates for the topic
  you choose. The response is a bulleted list of GEN/DISC slate statements
  per side, not a single synthesized sentence. The two-step flow lets you
  correct the framing when the top match isn't quite right.
- **Graph** — sidebar of available topics with status pills, and a
  click-to-expand view (topic → for/against → cluster → individual arguments).
  Each cluster and each FOR/AGAINST side displays its slate of representative
  statements. Topics without clusters or polarity slates yet are flagged, so
  you can see pipeline progress mid-batch.

## Build

```bash
# Build both Docker images.
docker compose build
```

## Fetch Real Mastodon Data

`fetch_mastodon.py` pulls statuses from one or more public hashtag timelines on a
Mastodon instance (default `mastodon.social`, no auth required) and writes them
to `test_data/mastodon_real.json` (the default source the pipeline reads from).
Reblogs, replies, non-English posts, and very short posts are dropped.

```bash
# One topic, 50 statuses.
uv run python fetch_mastodon.py --tag climate --per-tag 50

# Multiple tags in one shot.
uv run python fetch_mastodon.py --tag climate --tag remotework --per-tag 500

# Add more topics to the existing fixture (dedupes by status id).
uv run python fetch_mastodon.py --tag vegan --per-tag 500 --append
```

## LLM Backend

The pipeline talks to an LLM for claim extraction, topic framing, polarity
assignment, and statement generation. Two backends are supported via the
`LLM_PROVIDER` env var:

- **`llamafile` (default)**: local llama.cpp / llamafile container. Free, fully
  offline, slow on CPU (multi-hour batches even on 150 posts).
- **`openai`**: any OpenAI-compatible endpoint (Together AI, Fireworks, OpenAI,
  Groq, DeepInfra). Much faster, dollars per batch, can parallelize calls.

Configure via env vars (e.g. in an `.env` file next to `docker-compose.yml`):

| Var | Default | Required when |
|-----|---------|---------------|
| `LLM_PROVIDER` | `llamafile` | always |
| `LLM_BASE_URL` | `http://llm:8080/v1` | always |
| `LLM_MODEL` | — | `LLM_PROVIDER=openai` |
| `LLM_API_KEY` | — | `LLM_PROVIDER=openai` |
| `LLM_CONCURRENCY` | `1` | optional (8-16 with hosted) |

For example, to run against Together AI's Llama-3.1-8B-Instruct-Turbo:

```bash
# .env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.together.xyz/v1
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct-Lite
LLM_API_KEY=tgp-...
LLM_CONCURRENCY=8
```

With `LLM_PROVIDER=openai` you do **not** need to start the local `llm`
service. It lives in the `local` Compose profile and is skipped by default.

## First Run

```bash
# Hosted LLM (uses LLM_PROVIDER / LLM_BASE_URL / LLM_MODEL / LLM_API_KEY).
docker compose up -d

# OR: fully local with the bundled llamafile (downloads a 2 GB model on first build).
docker compose --profile local up -d

# Populate data/pipeline.db and data/bertopic_model from
# test_data/mastodon_real.json.
docker compose exec pipeline python main.py
```

To run against a different source JSON (e.g. the curated eval fixtures), pass
`--data-path`. This only affects `claim-extraction`, which is the one stage that
reads the source file; downstream stages read from the DB.

```bash
docker compose exec pipeline python main.py batch \
  --data-path test_data/mastodon_real.json
```

Each topic stores a polarity frame. `for` arguments support the topic's
`polarity_target`; `against` arguments oppose it. This keeps comparison topics
such as "Cats vs Dogs" consistent across all extracted arguments.

## Run a Long Batch in the Background (macOS)

Claim-extraction is per-post LLM-bound and can take a while. To run it without
holding your terminal open and without the Mac dozing off mid-run:

```bash
caffeinate -i docker compose exec -d pipeline bash -c \
  'python main.py > /app/data/pipeline.log 2>&1'
```

- `caffeinate -i` keeps the Mac from idle-sleeping.
- `docker compose exec -d` detaches the python process from your shell, so
  closing the terminal won't kill it.
- Output is redirected to `data/pipeline.log` on the host (the `./data` folder
  is volume-mounted into the container).

Watch progress without blocking the shell:

```bash
tail -f data/pipeline.log
```

If the Mac sleeps (lid-close), Docker Desktop pauses the container rather than
killing it, so the run resumes when you wake the laptop. For uninterrupted
runs, leave the lid open on AC power or use clamshell mode (external monitor,
power, keyboard/mouse all connected).

### Resume a crashed claim-extraction run

`claim-extraction` records a `claims_extracted_at` timestamp on each post as
soon as it finishes processing it (whether the post produced claims or not).
Re-run with `--resume` to skip those and only process the remainder:

```bash
docker compose exec pipeline python main.py stage claim-extraction --resume
# or, to continue with all downstream stages after the resumed extraction:
docker compose exec pipeline python main.py batch --resume
```

If the LLM returns malformed JSON for a post (e.g. truncated output), that
single post is logged as a warning, recorded in
`data/claim_extraction.failures.json`, treated as zero claims, and marked done
so the batch keeps moving. Inspect that file later to retry by hand.

The `claims_extracted_at` column was added in a later revision. If your
`data/pipeline.db` predates it, add the column once before the first
`--resume` run:

```bash
sqlite3 data/pipeline.db "ALTER TABLE posts ADD COLUMN claims_extracted_at DATETIME"
```

A truly fresh DB (`rm -rf data` then a normal run) does not need the manual
ALTER; SQLAlchemy creates the column from the model on first connect.

## Run Individual Pipeline Stages

```bash
# Stages: claim-extraction, topic-clustering, argument-graph,
# argument-reclustering, cluster-slate-generation,
# polarity-slate-generation.

# Run one stage (e.g. regenerate only the polarity slates,
# leaving expensive cluster slates untouched).
docker compose exec pipeline python main.py stage polarity-slate-generation

# Resume from a stage through the end of the pipeline.
docker compose exec pipeline python main.py batch --from-stage argument-graph
```

Rerunning `topic-clustering` clears existing topics, argument clusters, and
generated slates (claims are kept if `claim-extraction` was run separately,
otherwise they are re-extracted by the default batch).

Rerunning `argument-graph` clears existing argument clusters and generated
slates for the current topics, reassigns polarity for every claim using each
topic's polarity frame, and rebuilds the cluster structure.

Rerunning `argument-reclustering` preserves extracted argument instances, clears
argument clusters and generated slates, and rebuilds the cluster structure.

Argument clusters are produced per `(topic, polarity)` bucket using the same
UMAP + HDBSCAN approach as `topic-clustering`. Granularity is controlled by
`ARGUMENT_UMAP_*` and `ARGUMENT_HDBSCAN_*` in `config.py`; lower
`ARGUMENT_HDBSCAN_MIN_CLUSTER_SIZE` produces finer-grained clusters.

The API can start before these exist:

```text
data/pipeline.db
data/bertopic_model
```

`/match-post-topics` and `/topic-response/{id}` need them to be populated first.

## Process A Post

The single-post flow is split into two steps so the user can correct the topic
framing when the top match isn't quite right.

```bash
# Check that the API is ready.
curl http://localhost:8000/health

# Step 1: rank the closest topics for the post (default k=5, max 10).
curl -X POST 'http://localhost:8000/match-post-topics?k=5' \
  -H "Content-Type: application/json" \
  -d '{"id":"post_1","content":"should i adopt a cat or a dog"}'
```

`/match-post-topics` returns the candidates sorted by descending cosine
similarity to each topic's centroid:

```json
[
  {"id": "3", "label": "Cats vs Dogs", "polarity_target": "cats are preferable to dogs", "score": 0.81},
  {"id": "7", "label": "Pet Adoption",  "polarity_target": "people should adopt rather than buy pets", "score": 0.62},
  {"id": "1", "label": "Apartment Living", "polarity_target": "small apartments are fine for pets", "score": 0.41}
]
```

```bash
# Step 2: fetch the templated FOR/AGAINST paragraph for the chosen topic.
curl http://localhost:8000/topic-response/3
```

`/topic-response/{id}` returns plain text built from the topic's polarity
target and the precomputed GSC polarity slate for each side as a bulleted
list (sides without clusters yet are omitted). The slate is the canonical
output: each bullet is one representative statement from the GEN/DISC
loop, kept separate so the proportional structure of the slate is visible.

## Update

```bash
# Pull code changes.
git pull

# Rebuild images after code or dependency changes.
docker compose build

# Restart services with the new images.
docker compose up -d
```

If the data format changed:

```bash
# Delete generated pipeline data.
rm -rf data

# Start the LLM, API, and pipeline containers.
docker compose up -d

# Recreate data/pipeline.db and data/bertopic_model inside the existing image.
docker compose exec pipeline python main.py
```

## Stop

```bash
# Stop all services.
docker compose down
```
