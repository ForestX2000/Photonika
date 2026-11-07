# syntax=docker/dockerfile:1
# BuildKit skips the web stage when --target train is selected.
ARG PYTHON_BASE=python:3.12-slim

FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/src ./src
COPY frontend/public ./public
RUN npm run build

FROM ${PYTHON_BASE} AS scientific
WORKDIR /app
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60

RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Keep the expensive torch layer independent of application requirements/code.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_VERSION=2.5.1
ARG TORCH_PREINSTALLED=0
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TORCH_PREINSTALLED" = "1" ]; then \
        python -c "import os, torch; assert torch.__version__.split('+')[0] == os.environ['TORCH_VERSION']; print('Using preinstalled torch:', torch.__version__, 'CUDA:', torch.version.cuda)"; \
    else \
        python -m pip install --prefer-binary --retries 3 "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"; \
    fi

COPY requirements-core.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary --retries 3 -r requirements-core.txt

FROM scientific AS train
COPY ml ./ml
COPY tools ./tools
COPY scripts ./scripts
COPY requirements*.txt ./
CMD ["python", "-u", "tools/a6000.py", "--minutes", "180"]

FROM scientific AS web-deps
COPY requirements-web.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary --retries 3 -r requirements-web.txt

FROM web-deps AS application
COPY ml ./ml
COPY backend ./backend
COPY tools ./tools
COPY data/testsets ./data/testsets
COPY --from=web /web/dist ./frontend/dist
VOLUME /app/data/weights
VOLUME /app/results
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).close()"
CMD ["python", "-m", "backend", "--host", "0.0.0.0", "--port", "8000"]

FROM application AS streamlit
COPY requirements-demo.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary --retries 3 -r requirements-demo.txt
ENV STREAMLIT_SERVER_HEADLESS=true
EXPOSE 8501
HEALTHCHECK NONE
CMD ["streamlit", "run", "ml/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# Preserve the web application as the default target.
FROM application AS runtime
