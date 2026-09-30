FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/matleniz/datatoolkit"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependency layer: cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra api --no-install-project

# Project layer: only re-run when the source changes.
COPY src ./src
RUN uv sync --frozen --no-dev --extra api

ENV PATH="/app/.venv/bin:$PATH" \
    DTK_HOME=/data

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin dtk \
    && mkdir /data \
    && chown dtk:dtk /data

USER dtk
VOLUME /data
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/keys', timeout=4)"]

CMD ["dtk-api", "--host", "0.0.0.0", "--port", "8765"]
