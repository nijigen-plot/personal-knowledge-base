# CLAUDE.md

基本的なやり取りは日本語で行ってください。

このファイルはこのリポジトリで作業する際にClaude Code (claude.ai/code) への指針を提供します。

## プロジェクト概要

このプロジェクトは、ローカル推論のためのGoogle Gemma 3言語モデルを中心とした個人ナレッジベースです。日本語処理に最適化された、異なる形式（標準PyTorchおよびGGUF量子化）のGemma 3モデル（1Bおよび4Bパラメータ）の様々なバリエーションを実行することに焦点を当てています。

さらに、PlamoEmbeddingモデル（pfnet/plamo-embedding-1b）を使用したドキュメントのベクトル化とOpenSearchを用いた検索機能を提供するナレッジベースパイプラインも含まれています。

## 開発コマンド

### セットアップ
```bash
# 依存関係のインストール
uv sync

# Hugging Face CLIのセットアップ（モデルダウンロードに必要）
huggingface-cli login

# 大きなモデルファイル用のGit LFSのインストール
git lfs install
```

### モデルの実行
```bash
# 標準モデル（対話型会話）
python gemma3-1b.py
python gemma3-4b.py

# GGUF量子化モデル（コマンドラインインターフェース）
python gemma3-1b-q4-gguf.py "あなたのプロンプトをここに"
```

### ナレッジベースAPI
```bash
# ナレッジベースAPIサーバーの起動
python knowledge_base_api.py

# または uvicornで起動
uvicorn knowledge_base_api:app --host 0.0.0.0 --port 8000 --reload

# Embeddingモデル単体テスト
python embedding_model.py

# OpenSearchクライアント単体テスト
python opensearch_client.py
```

### コード品質
```bash
# コードフォーマット
black .

# インポートソート
isort .
```

## アーキテクチャ

### モデル実装
プロジェクトには異なるトレードオフを持つ3つの異なるモデルランナーが含まれています：

1. **gemma3-4b.py** - Transformersを使用した4Bパラメータモデル
   - 最高品質、最も遅い推論（i9-9980XEで約10秒）
   - 自動デバイス検出付きの`transformers.pipeline`を使用

2. **gemma3-1b.py** - Transformersを使用した1Bパラメータモデル
   - 4Bより高速だがGGUFより遅い（i9-9980XEで約30秒）
   - 4Bバージョンと同じアーキテクチャ

3. **gemma3-1b-q4-gguf.py** - llama-cpp-pythonを使用した1B量子化GGUFモデル
   - 最速推論（i9-9980XEで約3秒、Raspberry Pi 5で約7秒）
   - 会話履歴、ストリーミング出力、CLIインターフェースを含む
   - 永続的な会話メモリに`history.txt`を使用

### ナレッジベースパイプライン
新しく追加されたドキュメント処理・検索システム：

1. **embedding_model.py** - PlamoEmbeddingモデル（pfnet/plamo-embedding-1b）
   - 日本語に最適化されたローカルEmbeddingモデル
   - 自動デバイス検出（CUDA/CPU）
   - バッチ処理とベクトル正規化をサポート

2. **opensearch_client.py** - OpenSearchベクトルデータベース
   - KNNベクトル検索用のインデックス管理
   - HNSWアルゴリズムによる高速類似検索
   - バッチドキュメント挿入とメタデータサポート

3. **knowledge_base_api.py** - FastAPI RESTfulサーバー
   - ドキュメント追加・検索API
   - ファイルアップロード（.txt/.md）と自動チャンキング
   - リアルタイムパフォーマンス計測

### 共通パターン
- すべてのモデルで日本語システムプロンプトを使用（"あなたは日本語を話すAIアシスタントです"）
- `torch.cuda.is_available()`による自動CUDA/CPU デバイス検出
- `time.perf_counter()`によるパフォーマンス計測
- 異なるCPUアーキテクチャに最適化された自動dtype選択

### ハードウェア最適化
コードベースはx86とARMアーキテクチャの両方に最適化されています：
- **x86 (i9-9980XE)**: ネイティブfloat16サポートなし、変換に依存
- **ARM Cortex-A76 (Raspberry Pi 5)**: ARMv8.2-A経由でネイティブfloat16サポート

## モデル管理

### ディレクトリ構造
モデルはHugging Faceリポジトリ名と一致するディレクトリにローカルに保存されます：
- `gemma-3-1b-it/` - 1B指示調整モデル
- `gemma-3-4b-it/` - 4B指示調整モデル
- `gemma-3-1b-it-qat-q4_0-gguf/` - 1B GGUF量子化モデル

### モデルダウンロード
モデルはHugging Faceから手動でクローンする必要があります：
```bash
# Gemmaモデル
git clone https://huggingface.co/google/gemma-3-4b-it
git clone https://huggingface.co/google/gemma-3-1b-it

# PlamoEmbeddingモデル（自動ダウンロード）
# embedding_model.pyを実行すると自動的にダウンロードされます
```

## メモリとコンテキスト管理

GGUF実装（`gemma3-1b-q4-gguf.py`）には高度なメモリ管理が含まれています：
- 会話履歴は`history.txt`に保存
- `n_ctx`パラメータによるコンテキストウィンドウ管理（デフォルト：32768）
- コンテキスト制限に合わせた履歴の自動切り詰め
- 連続性のためにメモリファイルを読み取りシステムプロンプトに含める

## 開発ノート

- 自動テストインフラストラクチャなし - モデルは直接実行によってテストされます
- ビルドシステム不要 - 純粋なPython実行
- 開発ツールよりも推論パフォーマンスに重点
- すべてのモデルは日本語入力を期待し、日本語応答を提供します
- 環境変数はHugging Faceトークン用の`.env`ファイルで管理されます

## ナレッジベースAPI設定

### 環境変数
ナレッジベースAPIは以下の環境変数で設定できます：
```bash
# OpenSearch接続設定
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USER=admin
OPENSEARCH_PASS=admin
```

### APIエンドポイント
- **POST /documents** - 単一ドキュメント追加
- **POST /documents/batch** - 複数ドキュメント一括追加
- **POST /search** - ベクトル類似検索
- **POST /upload** - ファイルアップロード（.txt/.md）
- **GET /stats** - インデックス統計
- **DELETE /index** - インデックスリセット

### 使用例
```bash
# ドキュメント追加
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{"content": "これはテストドキュメントです", "metadata": {"type": "test"}}'

# 検索
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "テスト", "k": 5}'

# ファイルアップロード
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.txt"
```
