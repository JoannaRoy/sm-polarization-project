"""Central configuration for all pipeline components."""

import os

# --- Shared ---
DB_PATH = "data/pipeline.db"
SOURCE_TEST_DATA_PATH = "pipeline_test_data.json"
TEST_DATA_PATH = "test_data/mastodon_real.json"
GROUND_TRUTH_PATH = "test_data/ground_truth.json"
MASTODON_FIXTURE_BASE_URL = "https://mastodon.social"
FIXTURE_ACCOUNT_CREATED_AT = "2026-01-01T00:00:00.000Z"

# --- LLM ---
# Provider: "llamafile" for a local llama.cpp/llamafile server, or "openai" for
# any OpenAI-compatible chat-completions endpoint (Together AI, Fireworks,
# OpenAI itself, etc.). Both speak the same shape; the client picks the right
# JSON-schema field per provider.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "llamafile").lower()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
# Max in-flight LLM requests. 1 is safe for a single-slot local llamafile;
# 8-16 makes sense for a hosted provider (subject to rate limits).
LLM_CONCURRENCY = max(1, int(os.environ.get("LLM_CONCURRENCY", "1")))

TOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PREFERENCE_EMBEDDING_MODEL = "cartgr/embeddings-for-preferences-st5-xl"
PREFERENCE_EMBEDDING_DEVICE = "cpu"
PIPELINE_CPU_THREADS = 10
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# --- Topic Clustering (1-topic-clustering/) ---
TOPIC_MODEL_PATH = "data/bertopic_model"
UMAP_N_NEIGHBORS = 20
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 3

# --- Argument Graph (2-argument-graph/) ---
# UMAP + HDBSCAN per (topic, polarity) bucket. Buckets are smaller than the
# full topic-clustering corpus, so neighbors and cluster size are smaller too.
ARGUMENT_UMAP_N_NEIGHBORS = 5
ARGUMENT_UMAP_N_COMPONENTS = 5
ARGUMENT_UMAP_MIN_DIST = 0.0
ARGUMENT_UMAP_METRIC = "cosine"
ARGUMENT_UMAP_RANDOM_STATE = 42
ARGUMENT_HDBSCAN_MIN_CLUSTER_SIZE = 3
ARGUMENT_HDBSCAN_MIN_SAMPLES = 1

# --- Statement Generation (3-statement-generation/) ---
GSC_CLUSTER_SLATE_SIZE = 3
GSC_POLARITY_SLATE_SIZE = 3
GSC_GEN_QUERY_ITEMS = 5
