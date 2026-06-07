FROM debian:bookworm-slim

ARG LLAMAFILE_VERSION
ARG MODEL_FILE
ARG MODEL_URL

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /usr/local/bin/llamafile \
        "https://github.com/mozilla-ai/llamafile/releases/download/${LLAMAFILE_VERSION}/llamafile-${LLAMAFILE_VERSION}" \
    && chmod +x /usr/local/bin/llamafile

RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p /models \
    && if [ -s /run/secrets/hf_token ]; then \
           curl -fsSL -H "Authorization: Bearer $(cat /run/secrets/hf_token)" \
               -o "/models/${MODEL_FILE}" "${MODEL_URL}"; \
       else \
           curl -fsSL -o "/models/${MODEL_FILE}" "${MODEL_URL}"; \
       fi

ENV MODEL_PATH=/models/${MODEL_FILE} \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "exec /usr/local/bin/llamafile --server -m \"$MODEL_PATH\" --host 0.0.0.0 --port \"$PORT\" -c 4096 --cache-ram 256 --ctx-checkpoints 1"]
