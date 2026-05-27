FROM ghcr.io/mlflow/mlflow:v2.18.0

# Зависимости для transformers flavor — не ставить при каждом старте контейнера
RUN pip install --no-cache-dir \
    "protobuf>=4.21,<5" \
    "transformers>=4.36,<5" \
    "accelerate>=1.1.0" \
    "flask" \
    "torch" \
    --extra-index-url https://download.pytorch.org/whl/cpu

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
