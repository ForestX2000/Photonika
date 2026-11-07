# Photonika: синтез фазового ДОЭ под 4 изображения.
# Python 3.12. По умолчанию — CUDA-колёса PyTorch (cu128, драйвер NVIDIA 535+);
# для CPU-образа: docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu ...
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
WORKDIR /app

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1200

# Шрифты с кириллицей нужны генератору датасета (символы из TTF)
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade "pip>=25.2" && \
    pip install --no-cache-dir --timeout 1200 --retries 20 torch --index-url ${TORCH_INDEX}
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

COPY . .
RUN python -m app.cli make-testsets > /dev/null

VOLUME /app/data/weights
VOLUME /app/results
EXPOSE 8501

CMD ["streamlit", "run", "app/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
