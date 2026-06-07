"""HTTP client for the LLM backend.

Two providers are supported and selected by the ``LLM_PROVIDER`` env var:

- ``llamafile``: a local llama.cpp / llamafile server. Schemas are passed
  through the top-level ``json_schema`` request field.
- ``openai``: any OpenAI-compatible chat-completions endpoint (Together AI,
  Fireworks, OpenAI). Schemas are passed via ``response_format`` with
  ``type: "json_schema"``.

The shared ``requests.Session`` keeps connections warm across calls and retries
transient failures (LLM restart, 429 rate limits, 5xx) so a multi-hour batch
isn't killed by a single hiccup.
"""

import json
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
)

logger = logging.getLogger(__name__)

PROVIDER_LLAMAFILE = "llamafile"
PROVIDER_OPENAI = "openai"

# (connect_timeout, read_timeout). Connect is short so an unreachable backend
# fails fast; read is generous because grammar-constrained CPU generation can
# take a while on a local model.
DEFAULT_LLM_TIMEOUT = (10, 300)

# 1.5 * 2**(n-1) -> sleeps of 1.5s, 3s, 6s, 12s, 24s, 48s = ~95s total. Long
# enough to ride out a llamafile container restart or a hosted-API rate limit.
_retry = Retry(
    total=6,
    connect=6,
    read=2,
    status=4,
    backoff_factor=1.5,
    backoff_max=60,
    status_forcelist=(429, 502, 503, 504),
    allowed_methods=frozenset(["POST"]),
    raise_on_status=False,
)

_session = requests.Session()
_session.mount("http://", HTTPAdapter(max_retries=_retry))
_session.mount("https://", HTTPAdapter(max_retries=_retry))


def _chat_completions_url():
    return f"{LLM_BASE_URL.rstrip('/')}/chat/completions"


def _headers():
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    return headers


def _build_payload(messages, schema):
    if LLM_PROVIDER == PROVIDER_LLAMAFILE:
        return {"messages": messages, "json_schema": schema}
    if LLM_PROVIDER == PROVIDER_OPENAI:
        if not LLM_MODEL:
            raise ValueError(
                "LLM_MODEL must be set when LLM_PROVIDER=openai "
                "(e.g. 'meta-llama/Meta-Llama-3-8B-Instruct-Lite')"
            )
        return {
            "model": LLM_MODEL,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


def chat_completion(messages, schema, timeout=DEFAULT_LLM_TIMEOUT):
    """POST a chat-completions request and return the parsed JSON object the
    model emitted as its single message content. ``schema`` is a JSON schema
    dict; the client translates it to the right wire format per provider."""
    payload = _build_payload(messages, schema)
    response = _session.post(
        _chat_completions_url(),
        json=payload,
        headers=_headers(),
        timeout=timeout,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)
