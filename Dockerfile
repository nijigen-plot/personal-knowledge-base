# アーキテクチャ別のベースイメージを定義
ARG TARGETPLATFORM TARGETARCH

FROM --platform=$TARGETPLATFORM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:46cb48a4abfbc40c836fe57bc05a07101b6458fffc63bbdfd6a50db98c9358bd AS base-amd64
FROM --platform=$TARGETPLATFORM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:2915b5ef3034f5e5de8cf787d84a2b9b455e34a71f93b2b62bde97faedef99ff AS base-arm64

# アーキテクチャに応じて適切なベースイメージを選択
FROM base-${TARGETARCH} AS base

SHELL ["/bin/bash", "-c"]

# DEBIAN_FRONTEND=noninteractive は 設定しないとtzdataの設定が対話的になりそこから進まなくなる
RUN apt-get update \
    && env DEBIAN_FRONTEND=noninteractive apt-get -y install \
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
