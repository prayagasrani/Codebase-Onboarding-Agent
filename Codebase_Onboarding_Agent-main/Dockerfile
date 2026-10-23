FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git is required for cloning repos and incremental git-diff narrowing
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY evals ./evals

RUN pip install ".[faiss,treesitter,postgres]"

EXPOSE 8000
CMD ["onboarding-agent", "serve"]
