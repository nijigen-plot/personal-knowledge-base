FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ARG TARGETARCH

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get -y install \
    python3.11 python3-pip tzdata curl wget zip unzip git make jq vim direnv \
    && apt-get clean all \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo
ENV LANG=C.UTF-8
ENV HOME=/app

COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /bin/
RUN uv python install 3.11
RUN uv python pin 3.11

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --locked
RUN uv run pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
