# syntax=docker/dockerfile:1.7

# ============================================================
# Stage 1 : builder — installe les dépendances dans un venv
# ============================================================
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copie d'abord les fichiers de dépendances seuls (cache layer)
COPY pyproject.toml uv.lock ./

# Installe les dépendances (sans le projet lui-même)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-group monitoring

# Copie le code applicatif et installe le projet
COPY api/ ./api/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY models/ ./models/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group monitoring


# ============================================================
# Stage 2 : runtime — image finale légère
# ============================================================
FROM python:3.13-slim-bookworm AS runtime

# curl uniquement (pour le healthcheck) ; psycopg[binary] embarque libpq
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# User non-root UID 1000 (convention HF Spaces)
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser

WORKDIR /app

# Copie du venv et du code depuis le builder
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
