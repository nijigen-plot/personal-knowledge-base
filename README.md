Run the [Gemma3](https://huggingface.co/google/gemma-3-4b-it) Model for local.

# Setup

## .env

1. `cp .env.example .env`
2. .envの各項目に値を記入
3. `sudo apt-get install direnv`
4. `cp .envrc.example .envrc`
5. `sudo apt install nodejs npm`
6. `sudo npm install -g dotenv-cli`
7. `direnv allow`

## command flow

1. install uv
2. run `uv sync`
3. get Hugging Face Write access token https://huggingface.co/docs/hub/security-tokens
4. install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/guides/cli)
5. run `huggingface-cli login`
6. paste Hugging Face access token
7. install [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation)
8. clone llm repository `git clone https://huggingface.co/google/gemma-3-4b-it` or `git clone git@hf.co:google/gemma-3-4b-it`(Need Write Permission Access Token)
9. git embedding model repository `clone git@hf.co:pfnet/plamo-embedding-1b`
10. run `docker compose up -d` (OpenSearch Server 専用のサーバーがあるのでそっちで立ち上げる)
11. run `uv run pytest test_app.py` （単体テスト）
12. run `uv run uvicorn app:app --reload --port $APP_PORT` or `uv run python app.py`(FastAPI立ち上げ)

## 過去データの挿入

FastAPI経由でリクエストを送ってデータ挿入が可能

接続確認
```
$ curl -i http://localhost:8050
HTTP/1.1 200 OK
date: Sun, 29 Jun 2025 14:05:07 GMT
server: uvicorn
content-length: 53
content-type: application/json

{"message":"ナレッジベースAPIへようこそ"}
```

データ挿入
```
url = f"http://localhost:8050/documents"
response = requests.post(
    url,
    json=data,
    headers={"Content-Type": "application/json"},
    timeout=30
)
```
## 構成

- OpenSearchは192.168.0.45でホスト（OpenSearch用サーバー）
- FastAPIは192.168.0.46でホスト（LLM+Embedding用サーバー）
- APIの公開は192.168.0.44がSSL証明書を持っている&プロキシサーバーを立てているのでリバースプロキシしてFastAPIにアクセスさせる

# API

## 入力構成

- Timestamp (自動ではいる)
    - Optionalにして過去文書を入れる時は明示できるようにする
- タグ(話題が何に関連するものなのか。指定しない場合全て。)
    - lifestyle, music, technology
- 質問文(自由入力)

# モデルのメモリ解放について

https://github.com/mjun0812/hf-model-cleanup-experiment

> export MALLOC_TRIM_THRESHOLD_=-1 + del model; gc.collect();で削除

gemma3.py, embedding_model.pyに内容は記載

# CPUメモ

1. Raspberry Pi 5はBCM2712 SoCで、CPUはArm Cortex-A76 (https://eetimes.itmedia.co.jp/ee/articles/2309/28/news177.html)
2. Cortex-A76はARM v8.2-Aアーキテクチャで半精度16bitの計算に対応している (https://en.wikipedia.org/wiki/AArch64#ARMv8.2-A)
3. Intel x86 CPUは4世代Xeonから半精度16bit計算対応で普通の人はまず持ってない (https://zenn.dev/mod_poppo/articles/half-precision-floating-point)
    a. 変換はIvy Bridgeから対応(3世代)

# Benchmarkメモ

4bは簡素で1bは文章量が多いので生成秒数が逆転している
リアルタイムで描画するようにしたらまた評価かわるかも
google gemma 3 1b ggufはめっちゃ速い 3~5秒くらい。Raspberry Pi 5なら7秒くらい

## i9-9980XE

- 4b -> 10秒程
- 1b -> 30秒程
- 1b gguf -> 3秒程
## Cortex-A76

- 4b -> 40~50秒
- 1b -> 40~50秒
- 1b gguf -> 7秒程
