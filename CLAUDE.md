# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

基本的なやり取りは日本語で行ってください。

## プロジェクト概要

このプロジェクトは、RAG（Retrieval-Augmented Generation）ベースの個人ナレッジベースシステムです。@Quarkgabberの2012年からのTwitter履歴および日常記録を統合し、日本語特化のAI会話システムを構築しています。

### アーキテクチャの全体像

1. **Embedding Layer**: PlamoEmbedding (pfnet/plamo-embedding-1b) でドキュメントのベクトル化
2. **Vector Store**: OpenSearch でベクトル検索とフィルタリング
3. **LLM Layer**: OpenAI GPT-4o または Gemma 3 モデルで対話生成
4. **API Layer**: FastAPI (app.py) でRESTful APIを提供
5. **Frontend Layer**: Streamlit (streamlit_app.py) でWebチャットインターフェース

### コンポーネント間の連携

- **質問フロー**: ユーザー質問 → Embedding → OpenSearch検索 → LLMコンテキスト生成 → 回答生成
- **タグ抽出**: `llm.py`の`extract_tag()`で質問から検索フィルタ（tag, timestamp）を自動抽出
- **メモリ管理**: 各モジュール（`embedding_model.py`, `llm.py`）が明示的なメモリ解放機能を持つ

## 開発コマンド

### セットアップ
```bash
# 1. 環境設定ファイル作成
cp .env.example .env
# .envファイルを編集してOPENAI_API_KEY等を設定

# 2. 依存関係インストール
uv sync --all-groups

# 3. モデルファイルのダウンロード（オプション）
# git clone git@hf.co:pfnet/plamo-embedding-1b
# git clone git@hf.co:openai/gpt-oss-20b

# 4. GPU版PyTorch（必要に応じて）
uv run pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

### サーバー起動
```bash
# OpenSearchクラスターの起動
docker compose --profile opensearch up -d

# FastAPI開発サーバー（Docker）
docker compose --profile fastapi up -d

# Streamlitアプリ（Docker）
docker compose --profile streamlit up -d

# または、ローカルでサーバー起動
uv run uvicorn app:app --reload --port $APP_PORT --host $APP_HOST
uv run streamlit run streamlit_app.py --server.port $STREAMLIT_APP_PORT
```

### 開発用プロファイル
```bash
# 全サービス起動（開発用）
docker compose --profile dev up -d

# OpenSearchのみ起動
docker compose --profile opensearch up -d

# FastAPIのみ起動
docker compose --profile fastapi up -d
```

### テストとモデル実行
```bash
# 単体テスト実行
uv run pytest test_app.py

# 各モジュールの単体テスト
uv run python embedding_model.py
uv run python opensearch_client.py

# LLMモデル直接テスト
uv run python llm.py "こんにちは～あなたのモデルはなんですか？"

# タグ抽出テスト
uv run python llm.py "最近あった音楽関係の出来事は？" --extract-tag

# RAG会話テスト（APIを使用）
curl -X POST "http://localhost:8050/api/v1/conversation" \
  -H "Content-Type: application/json" \
  -d '{"question": "最近あった出来事は？", "debug": true}'
```

### コード品質管理
```bash
# プリコミットチェック（フォーマット＋リント）
uv run pre-commit run --all-files

# 個別実行
uv run black .
uv run isort .
```

## アーキテクチャの詳細

### コアコンポーネント

1. **llm.py** - 統合言語モデルクラス
   - OpenAI GPT-4o API (`model_type="openai-api"`)
   - ローカルモデル (`model_type="openai-20b"`) - `./gpt-oss-20b`モデル使用
   - 会話履歴管理 (`history.txt`) とタグ・タイムスタンプ抽出機能
   - RAGコンテキスト統合と一人称回答生成

2. **embedding_model.py** - ベクトル化エンジン
   - PlamoEmbedding (pfnet/plamo-embedding-1b) 日本語特化
   - 自動デバイス検出（CUDA/CPU）と適応的dtype選択
   - バッチ処理対応とメモリ効率最適化

3. **opensearch_client.py** - ベクトル検索エンジン
   - OpenSearch KNN検索（HNSW + Faiss）
   - コサイン類似度ベース検索
   - タグ・タイムスタンプフィルタリング機能

4. **app.py** - 統合APIサーバー
   - FastAPI RESTful API
   - ドキュメント管理（CRUD）とRAG会話エンドポイント
   - 認証・バリデーション・エラーハンドリング

### データフロー設計

```
ユーザー質問 → LLM.extract_tag() → タグ・時期抽出
             ↓
PlamoEmbedding.encode() → ベクトル化
             ↓
OpenSearch.search() → 関連ドキュメント検索（フィルタ適用）
             ↓
LLM.generate() → RAGコンテキスト統合 → 一人称回答生成
```

### Docker Compose アーキテクチャ

- **opensearch-node1/node2**: クラスター構成のベクトルデータベース
- **opensearch-dashboards**: 管理UI（ポート5601）
- **fastapi**: API サーバー（app.py）
- **streamlit**: Web UI（streamlit_app.py）
- **profiles**: 開発用（`dev`）、個別サービス用プロファイル

### メモリ管理戦略

- `MALLOC_TRIM_THRESHOLD_=-1` + `gc.collect()` でモデルメモリ解放
- 各クラスに`clear_memory()`メソッドと`__del__()`デストラクタ
- CPUアーキテクチャ別dtype最適化（ARMv8.2-A float16 vs x86変換）

## API エンドポイント設計

### 認証システム
- **管理機能**: `ADMIN_API_KEY` Bearer Token認証必須
- **検索・会話**: 認証不要（パブリックAPI）

### 主要エンドポイント
```
GET    /api/v1/                    - API情報とガイダンス
POST   /api/v1/conversation        - RAG会話（認証不要）
POST   /api/v1/search              - ベクトル検索（認証不要）
POST   /api/v1/documents           - ドキュメント追加（認証必要）
POST   /api/v1/documents/batch     - 一括ドキュメント追加（認証必要）
GET    /api/v1/stats               - インデックス統計（認証必要）
DELETE /api/v1/index               - インデックスリセット（認証必要）
DELETE /api/v1/documents/{id}      - ドキュメント削除（認証必要）
```

### データモデル構造

```python
# 入力
DocumentRequest: content, timestamp, tag (lifestyle|music|technology)
SearchRequest: query, k, tag_filter, timestamp_filter
ConversationRequest: question, use_history, max_tokens, temperature, search_k, debug

# 出力
SearchResult: id, score, content, tag, timestamp
ConversationResponse: question, answer, search_results, search_count, used_knowledge, processing_time, model_type
```

## 環境設定とデプロイ

### 必須環境変数
```bash
# OpenAI API (メインLLM)
OPENAI_API_KEY=your_api_key

# OpenSearch設定
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin

# API設定
APP_HOST=0.0.0.0
APP_PORT=8050
STREAMLIT_APP_PORT=8501
ADMIN_API_KEY=your_admin_key

# Dashboard設定
OPENSEARCH_DASHBOARD_PORT=5601
```

### 本番デプロイメモ
- **192.168.0.45**: OpenSearchクラスター専用サーバー
- **192.168.0.46**: FastAPI + Embedding処理サーバー
- **192.168.0.44**: SSL + Apache リバースプロキシサーバー
- systemd serviceファイルでデーモン化済み

## 開発時の注意点

### モデルファイル管理
- `plamo-embedding-1b/`: PlamoEmbeddingモデル（必須）
- `gpt-oss-20b/`: ローカルGPTモデル（オプション）
- Git LFS使用でモデルファイル管理

### パフォーマンス特性
- **Gemma 3 1B/4B**: ローカル推論可能だがRAG能力限定
- **OpenAI GPT-4o**: RAG統合と一人称回答に最適（現在のデフォルト）
- **PlamoEmbedding**: 日本語ベクトル化に特化

### デバッグとモニタリング
- `debug=true` パラメータで検索結果詳細を表示
- `log_config.py` で統一ログ設定
- 全API呼び出しでprocessing_time計測

## 追加開発ガイド

### Docker Compose での開発
- volume マウント（`./:/app`）でローカルファイル変更がコンテナに即座に反映
- `--reload` フラグで FastAPI/Streamlit の自動リロード対応
- ヘルスチェック設定で OpenSearch 起動待ちを自動化

### ドキュメント挿入パイプライン
```bash
# バッチドキュメント挿入例
url = f"http://localhost:8050/api/v1/documents/batch"
response = requests.post(
    url,
    json=documents_list,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ADMIN_API_KEY}"
    }
)
```

### タグ自動分類システム
- LLM が質問から `lifestyle|music|technology` タグを抽出
- 正規表現ベースのタイムスタンプ抽出（「最近」「昨日」「3日前」等）
- `extract_tag()` でフィルタ条件の自動生成

## 重要な設計原則

- **日本語特化**: 全システムが日本語処理に最適化
- **個人化**: @Quarkgabber の一人称体験として回答生成
- **リアルタイム**: ストリーミング対応で即座な応答
- **スケーラビリティ**: Docker Compose でマルチサービス展開
- **メモリ効率**: 明示的なリソース管理で安定動作
