FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels .


FROM python:3.12-slim AS runtime

ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels /wheels/vllm_agent_gateway-*.whl \
    && rm -rf /wheels \
    && groupadd --gid 10001 gateway \
    && useradd --uid 10001 --gid gateway --no-create-home --shell /usr/sbin/nologin gateway

USER gateway

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3); r.close()"

STOPSIGNAL SIGTERM
ENTRYPOINT ["vllm-agent-gateway"]
