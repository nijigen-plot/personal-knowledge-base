FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip

COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /bin/
RUN uv python install 3.11
RUN uv python pin 3.11

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY llm.py log_config.py ./

RUN uv sync --locked
RUN uv run pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

CMD ["uv", "run", "python", "llm.py", "--model-type", "openai-20b"]
