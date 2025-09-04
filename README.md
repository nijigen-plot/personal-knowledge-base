# Knowledge Base

[nijigen-plot](https://x.com/nijigen_plot)([Quarkgabber](https://x.com/Quarkgabber))のナレッジベースです。

要するに私について質問ができます！

2012年からの過去ツイートと、2025年からちょいちょい入れてる日常系データをRAGとしています。

![アーキテクチャ図](./images/architecture.png)

[全体像について記載した記事](https://zenn.dev/nijigen_plot/articles/personal_knowledge_base)

## 使い方

### StreamlitチャットBOT

質問に対する回答(conversation API)をしてくれるアプリ

https://home.quark-hardcore.com/personal-knowledge-base/app/

<img width="998" height="613" alt="Image" src="https://github.com/user-attachments/assets/f0420ba5-3062-456c-9d2b-69a1401a598c" />

## API

以下API Docsを参照してください

- https://home.quark-hardcore.com/personal-knowledge-base/api/v1/docs
- https://home.quark-hardcore.com/personal-knowledge-base/api/v1/redoc



## Remote MCP

### 利用方法

MCP Settingにて以下を追加して再起動で認識されたらOK

```json
{
  "mcpServers": {
    "calculator": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "my-mcp-server.your-account.workers.dev/sse"
      ]
    }
  }
}
```


Claude Desktopの場合は設定→コネクタ→カスタムコネクタを追加で/sse付きURLを入れることで追加できた

### ローカル開発&Deploy

CloudFlareでRemote MCP Serverをホストしています。

`cd ./mcp-server`後`npm run deploy`でローカルMCPサーバーが立ちます。

`npx @modelcontextprotocol/inspector@latest`で[MCP Inspector](https://modelcontextprotocol.io/legacy/tools/inspector)が立ちます。

Transport Type: sse
URL : http://localhost:8787/sse

に設定し接続、List ToolsからToolを選んでRun Tool
問題無ければレスポンスが返ります。

`npm run deploy`でCloudFlareにDeploy(初回のみOAuth認証が入ります)

MCP InspectorでURLをCloudFlareにDeployして発行されたURL/sseに差し替えます。

これでも問題無ければリモートで動いてる

![成功例](./images/screenshot2025-09-02231313.png)

#### 参考

https://azukiazusa.dev/blog/cloudflare-mcp-server/

https://developers.cloudflare.com/agents/guides/remote-mcp-server/

## Setup

### .env

1. `cp .env.example .env`
2. .envの各項目に値を記入
3. `sudo apt-get install direnv`
4. `cp .envrc.example .envrc`
5. `sudo apt install nodejs npm`
6. `sudo npm install -g dotenv-cli`
7. `source ~/.bashrc` (.bashrcに`eval "$(direnv hook bash)"`があること前提)
8. `direnv allow`

### 共通設定

結局LLMはOpenAIが安定という結論になったので、デフォルトOPENAI APIを使っています。
.envに`OPENAPI_API_KEY=`があるのでそれにKEYいれてください。

1. install uv
2. run `uv sync --all-groups`
3. (必須ではない)get Hugging Face Write access token https://huggingface.co/docs/hub/security-tokens
4. (必須ではない)install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/guides/cli)
5. (必須ではない)run `huggingface-cli login`
6. (必須ではない)paste Hugging Face access token
7. install [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation)
8. (必須ではない)clone llm repository `git clone git@hf.co:google/gemma-3-1b-it-qat-q4_0-gguf`(Need Write Permission Access Token)
9. git embedding model repository `git clone git@hf.co:pfnet/plamo-embedding-1b`
10. run `docker compose --profile xxx up -d` (OpenSearch Server 専用のサーバー 192.168.0.45があるのでそっちで立ち上げ済)
  a. Dev環境なら--profile dev
  b. 本番環境は役割毎にPC違うのでそのprofileを指定
11. run `docker compose ps -a` で立ち上がっているかチェック
12. run `curl https://localhost:9200 -ku username:password` で情報が返ってくるかチェック
13. run `uv run pytest test_app.py` （単体テスト）
14. run `uv run uvicorn app:app --reload --port $APP_PORT --host $APP_HOST` or `uv run python app.py`(FastAPI立ち上げ)
15. run `uv run streamlit run streamlit_app.py --server.port $STREAMLIT_APP_PORT` (192.168.0.44 チャットBOT用Streamlit立ち上げ)

### API Server

### LLM Server

1. run `docker build -f llm.Dockerfile . -t llm-server`
2. run `docker run --gpus all -v ./gpt-oss-20b:/app/gpt-oss-20b llm-server`

### OpenSearch Server

セキュリティやらmax_map_countやらで落とし穴多い

- https://docs.opensearch.org/2.19/install-and-configure/install-opensearch/index/#important-settings

OpenSearchを立ち上げる環境では以下の設定が必要

```bash:/etc/sysctl.conf
vm.max_map_count=262144
```

設定を確認
```
$ sudo sysctl -p
$ sysctl vm.max_map_count
```

### Front Server

#### daemon
- 192.168.0.46 : `/etc/systemd/system/personal-knowledge-base-fastapi.service`で起動時立ち上がるように設定済み。HOST,PORTは直接書いてるので要確認
- 192.168.0.44 : `/etc/systemd/system/personal-knowledge-base-streamlit.service`で起動時立ち上がるように設定済み。HOST,PORTは直接書いてるので要確認
    - `start_streamlit.sh`をdaemonで実行している

#### Streamlit Meta

https://discuss.streamlit.io/t/adding-a-meta-description-to-your-streamlit-app/17847/5

`.venv/lib/python3.11/site-packages/streamlit/static/index.html`を以下のように変更すると、URL埋め込みで情報が反映される。
これはgit管理外なので適宜対応


```
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta property="og:title" content="Quarkgabber Knowledge Base">
    <meta property="og:url" content="https://home.quark-hardcore.com/personal-knowledge-base/app/">
    <meta
      name="viewport"
      content="width=device-width, initial-scale=1, shrink-to-fit=no"
    />
    <link rel="shortcut icon" href="./favicon.png" />
    <link
      rel="preload"
      href="./static/media/SourceSansVF-Upright.ttf.BsWL4Kly.woff2"
      as="font"
      type="font/woff2"
      crossorigin
    />

    <title>Quarkgabber Knowledge Base</title>

    <!-- initialize window.prerenderReady to false and then set to true in React app when app is ready for indexing -->
    <script>
      window.prerenderReady = false
    </script>
    <script type="module" crossorigin src="./static/js/index.DvRPFfw6.js"></script>
    <link rel="stylesheet" crossorigin href="./static/css/index.CJVRHjQZ.css">
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
```

### 過去データの挿入

FastAPI経由でリクエストを送ってデータ挿入が可能
ADMIN_API_KEYに設定した値をBearer Tokenとして認証が必要

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
url = f"http://localhost:8050/api/v1/documents"
response = requests.post(
    url,
    json=data,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ADMIN_API_KEY}"
    },
    timeout=30
)
```

### LLMのテスト

普通の会話
```
$ uv run python llm.py "こんにちは～あなたのモデルはなんですか？"
2025-07-09 13:39:51,767 - __main__ - INFO - OpenAI APIクライアントを初期化しました
2025-07-09 13:39:51,767 - __main__ - INFO - history.txt から過去の会話履歴を読み込みました。
こんにちは～私はOpenAIのGPT-4oモデルで動作しています。
2025-07-09 13:39:54,381 - __main__ - INFO - 処理時間: 2.62秒
言語モデルのメモリを解放中...
言語モデルのメモリ解放完了
```


RAGを利用した会話
"debug": trueをbodyにいれるとベクトル検索で引っかかった文書内容も見れるよ
```
$ curl -X POST "http://localhost:8050/api/v1/conversation" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ADMIN_API_KEY" \
    -d '{
      "question": "最近あった出来事は？"
    }'
{"question":"最近あった出来事は？","answer":"最近の出来事といえば、曲の最後の詰めをしようとしていたときに、Wavesのプラグインの認証が急に通らなくなってしまったことがありました。ちょっと焦りましたが、PowerShell周りをいじったらなんとか直りました。ただ、Wavesのプラグインの認証周りは本当に勝手が悪くて、もう少しスムーズにいってほしいなと思っています。","search_results":[{"id":"WMFm5ZcBlg4zycrBZ0lN","score":0.78672117,"content":"曲最後の詰めやろうとしたら急にWavesの認証が通らなくなった・・・PowerShell周り弄ったら直ったけど、プラグイン認証周り勝手が悪いんだよなーWavesは","tag":"music","timestamp":"2025-07-05T18:00:00.000000"}],"search_count":1,"used_knowledge":true,"processing_time":6.18,"model_type":"openai","model_size":"4b"}
```


タグ抽出
```
$ uv run python llm.py "最近あった音楽関係の出来事は？" --extract-tag
2025-07-09 13:30:56,142 - __main__ - INFO - OpenAI APIクライアントを初期化しました
```json
{
  "tag": ["music"]
}
```
2025-07-09 13:30:58,831 - __main__ - INFO - 処理時間: 2.69秒
抽出結果 (試行 1/3): ```json
{
  "tag": ["music"]
}
```
{'content': '最近あった音楽関係の出来事は？', 'tag': 'music', 'timestamp': {'gte': '2025-07-02T22:30:56.142483', 'lte': '2025-07-09T22:30:56.142483'}}
言語モデルのメモリを解放中...
言語モデルのメモリ解放完了
```

### 構成

- OpenSearchは192.168.0.45でホスト（OpenSearch用サーバー）
- FastAPIは192.168.0.46でホスト（LLM+Embedding用サーバー）
- APIの公開は192.168.0.44がSSL証明書を持っている&プロキシサーバーを立てているのでリバースプロキシしてFastAPIにアクセスさせてます
    - 設定内容は192.168.0.44の`/etc/apache2/sites-enabled/`を参照

## 開発

### Formatter/CI

`uv run pre-commit run --all-files`でコミット前にフォーマットチェック

### モデルのメモリ解放について

https://github.com/mjun0812/hf-model-cleanup-experiment

> export MALLOC_TRIM_THRESHOLD_=-1 + del model; gc.collect();で削除

gemma3.py, embedding_model.pyに内容は記載
