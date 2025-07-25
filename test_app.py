import os
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from log_config import get_logger

logger = get_logger(__name__)


# fixtureはテストにされる関数で、API連携やDB接続を事事前テストするためのもの。
@pytest.fixture
def mock_embedding_model():
    """エンベディングモデルのモック"""
    mock_model = Mock()
    mock_model.encode.return_value = np.array(
        [[0.1, 0.2, 0.3, 0.4]]
    )  # 4次元のダミーベクトル
    mock_model.get_embedding_dimension.return_value = 4
    return mock_model


@pytest.fixture
def mock_vector_store():
    """ベクトルストアのモック"""
    mock_store = Mock()
    mock_store.create_index.return_value = None
    mock_store.add_documents.return_value = (1, [])  # 成功1件、失敗0件
    mock_store.search.return_value = [
        {
            "id": "test_doc_1",
            "score": 0.9,
            "content": "これはテストドキュメントです",
            "tag": "music",
            "timestamp": "2024-01-01T00:00:00.000000",
        }
    ]
    mock_store.get_index_stats.return_value = {
        "document_count": 1,
        "size_bytes": 1024,
        "size_mb": 0.001,
    }
    mock_store.delete_document.return_value = {
        "message": "ドキュメント test_doc_1 を削除しました",
        "result": "deleted",
    }
    return mock_store


@pytest.fixture
def mock_llm_model():
    """LLMモデルのモック"""
    mock_llm = Mock()
    mock_llm.model_type = "gguf"
    mock_llm.model_size = "1b"
    mock_llm.generate.return_value = "これはLLMからのテスト応答です。"
    mock_llm.extract_tag.return_value = {
        "tag": "music",
        "timestamp": None,
        "content": "テスト質問",
    }
    return mock_llm


@pytest.fixture
def client(mock_embedding_model, mock_vector_store, mock_llm_model):
    """テスト用のFastAPIクライアント"""
    # テスト用のアプリケーションを作成（lifespanなし）
    from fastapi import FastAPI

    test_app = FastAPI(
        title="テスト用ナレッジベースAPI", description="テスト用API", version="1.0.0"
    )

    # 元のアプリからルートをコピー
    for route in app.routes:
        if hasattr(route, "path") and route.path not in [
            "/openapi.json",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
        ]:
            test_app.routes.append(route)

    # モックをグローバル変数に注入
    with (
        patch("app.embedding_model", mock_embedding_model),
        patch("app.vector_store", mock_vector_store),
        patch("app.llm_model", mock_llm_model),
    ):
        # TestClientはresponseコードを自動で返す
        with TestClient(test_app) as test_client:
            # Bearer Token認証に変更
            api_key = os.getenv("ADMIN_API_KEY", "test-api-key")
            test_client.headers.update({"Authorization": f"Bearer {api_key}"})
            # yieldにすることで、テスト後にAPIやDB接続を閉じることができる。今回はFastAPIのTestClientを使っているので、yieldでなくてもいい。
            yield test_client


# Testで始まるClass、test_で始まる関数はpytestが自動で検出して実行する。
class TestRootEndpoint:
    """ルートエンドポイントのテスト"""

    def test_root_endpoint(self, client):
        """ルートエンドポイントが正しく動作することを確認"""
        response = client.get("/api/v1/")
        assert response.status_code == 200
        assert response.json() == {
            "message": "Quarkgabberの個人ナレッジベースAPIへようこそ",
            "description": "このAPIは、Quarkgabberの日常記録、体験談、知識が蓄積された個人ベクトルデータベースです。",
            "features": {
                "embedding_model": "PlamoEmbedding (pfnet/plamo-embedding-1b) による日本語特化ベクトル化",
                "vector_search": "OpenSearch による高速類似検索とRAG（Retrieval-Augmented Generation）",
                "ai_conversation": "OpenAI GPT-4o または Gemma 3 による自然な対話生成",
            },
            "usage": "'/conversation' エンドポイントで質問を送信すると、関連する記録を検索し、Quarkgabberの体験として一人称で回答します。",
            "author": "Quarkgabber",
            "website": "https://quark-hardcore.com/",
        }


class TestSearchEndpoint:
    """検索エンドポイントのテスト"""

    def test_search_success(self, client, mock_embedding_model, mock_vector_store):
        """正常な検索が動作することを確認"""
        response = client.post("/api/v1/search", json={"query": "テスト", "k": 5})

        if response.status_code != 200:
            logger.error(f"Response status: {response.status_code}")
            logger.error(f"Response body: {response.text}")

        assert response.status_code == 200
        data = response.json()

        # レスポンスの構造を確認
        assert isinstance(data, list)
        assert len(data) == 1

        # 最初の結果の内容を確認
        result = data[0]
        assert result["id"] == "test_doc_1"
        assert result["score"] == 0.9
        assert result["content"] == "これはテストドキュメントです"
        assert result["tag"] == "music"

        # モックが正しく呼び出されたことを確認
        mock_embedding_model.encode.assert_called_once_with(["テスト"])
        mock_vector_store.search.assert_called_once()

    def test_search_with_filter_query(self, client, mock_vector_store):
        """フィルタクエリが動作することを確認"""
        # フィルタ付きの検索リクエスト
        response = client.post(
            "/api/v1/search",
            json={"query": "テスト", "k": 10, "tag_filter": "music"},
        )

        assert response.status_code == 200
        data = response.json()

        # モックが正しいフィルタで呼び出されたことを確認
        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["tag_filter"] == "music"

    def test_search_missing_query(self, client):
        """クエリパラメータが不足している場合のエラー処理"""
        response = client.post("/api/v1/search", json={})
        assert response.status_code == 422  # バリデーションエラー

    def test_search_default_parameters(self, client):
        """デフォルトパラメータでの検索が動作することを確認"""
        response = client.post("/api/v1/search", json={"query": "テスト"})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestAddEndpoint:
    """ドキュメント追加エンドポイントのテスト"""

    def test_add_document_success(
        self, client, mock_embedding_model, mock_vector_store
    ):
        """正常なドキュメント追加が動作することを確認"""
        document_data = {
            "content": "新しいテストドキュメント",
            "tag": "technology",
        }

        response = client.post("/api/v1/documents", json=document_data)

        assert response.status_code == 200
        data = response.json()

        # レスポンスの内容を確認
        assert data["message"] == "ドキュメントが正常に追加されました"
        assert "processing_time" in data
        assert data["embedding_dimension"] == 4

        # モックが正しく呼び出されたことを確認
        mock_embedding_model.encode.assert_called_once_with("新しいテストドキュメント")
        mock_vector_store.add_documents.assert_called_once()

    def test_add_document_with_minimal_data(self, client):
        """最小限のデータでドキュメント追加"""
        document_data = {"content": "最小限のドキュメント", "tag": "lifestyle"}

        response = client.post("/api/v1/documents", json=document_data)
        assert response.status_code == 200

    def test_add_document_missing_content(self, client):
        """コンテンツが不足している場合のエラー処理"""
        document_data = {"tag": "music"}

        response = client.post("/api/v1/documents", json=document_data)
        assert response.status_code == 422  # バリデーションエラー

    def test_add_document_storage_failure(self, client, mock_vector_store):
        """ストレージ失敗の場合のエラー処理"""
        # 保存失敗をシミュレート
        mock_vector_store.add_documents.return_value = (0, ["failed_item"])

        document_data = {"content": "失敗するドキュメント", "tag": "music"}

        response = client.post("/api/v1/documents", json=document_data)
        assert response.status_code == 500
        assert "ドキュメントの保存に失敗しました" in response.json()["detail"]


class TestStatsEndpoint:
    """統計エンドポイントのテスト"""

    def test_get_stats_success(self, client, mock_vector_store):
        """正常な統計取得が動作することを確認"""
        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = response.json()

        # レスポンスの構造を確認
        assert data["document_count"] == 1
        assert data["size_bytes"] == 1024
        assert data["size_mb"] == 0.001

        # モックが正しく呼び出されたことを確認
        mock_vector_store.get_index_stats.assert_called_once_with("knowledge-base")

    def test_get_stats_index_not_found(self, client, mock_vector_store):
        """インデックスが見つからない場合のエラー処理"""
        mock_vector_store.get_index_stats.return_value = {
            "error": "インデックスが見つかりません"
        }

        response = client.get("/api/v1/stats")
        assert response.status_code == 404


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_search_embedding_error(self, client, mock_embedding_model):
        """エンベディング生成エラーの処理"""
        mock_embedding_model.encode.side_effect = Exception("エンベディングエラー")

        response = client.post("/api/v1/search", json={"query": "エラーテスト"})
        assert response.status_code == 500
        assert "検索エラー" in response.json()["detail"]

    def test_add_document_embedding_error(self, client, mock_embedding_model):
        """ドキュメント追加時のエンベディングエラー処理"""
        mock_embedding_model.encode.side_effect = Exception("エンベディングエラー")

        document_data = {"content": "エラーテストドキュメント", "tag": "technology"}
        response = client.post("/api/v1/documents", json=document_data)

        assert response.status_code == 500
        assert "エラー" in response.json()["detail"]


class TestConversationEndpoint:
    """会話エンドポイント（RAG）のテスト"""

    def test_conversation_with_rag_success(
        self, client, mock_embedding_model, mock_vector_store, mock_llm_model
    ):
        """正常なRAG会話が動作することを確認"""
        conversation_data = {
            "question": "ナレッジベースについて教えて",
            "max_tokens": 256,
            "temperature": 0.5,
            "search_k": 3,
        }

        response = client.post("/api/v1/conversation", json=conversation_data)

        if response.status_code != 200:
            logger.error(f"Response status: {response.status_code}")
            logger.error(f"Response body: {response.text}")

        assert response.status_code == 200
        data = response.json()

        # レスポンスの構造を確認
        assert data["question"] == "ナレッジベースについて教えて"
        assert data["answer"] == "これはLLMからのテスト応答です。"
        assert data["model_type"] == "gguf"
        assert data["model_size"] == "1b"
        assert "processing_time" in data
        assert "search_results" in data
        assert data["search_count"] == 1  # mock_vector_storeから1件返される
        assert data["used_knowledge"] == True

        # Embeddingモデルが正しく呼び出されたことを確認
        mock_embedding_model.encode.assert_called_once_with(
            ["ナレッジベースについて教えて"]
        )

        # ベクトルストアが正しく呼び出されたことを確認
        mock_vector_store.search.assert_called_once()

        # LLMが正しく呼び出されたことを確認
        mock_llm_model.generate.assert_called_once()
        call_args = mock_llm_model.generate.call_args
        assert call_args[1]["prompt"] == "ナレッジベースについて教えて"
        assert call_args[1]["max_tokens"] == 256
        assert call_args[1]["temperature"] == 0.5
        assert call_args[1]["silent"] == True
        # ナレッジコンテキストが生成されていることを確認
        assert "【参考情報 1】" in call_args[1]["rag_context"]

    def test_conversation_no_relevant_docs(
        self, client, mock_vector_store, mock_llm_model
    ):
        """関連文書が見つからない場合のテスト"""
        # 空の検索結果を返すようにモックを設定
        mock_vector_store.search.return_value = []

        conversation_data = {"question": "関連性のない質問"}

        response = client.post("/api/v1/conversation", json=conversation_data)

        assert response.status_code == 200
        data = response.json()

        assert data["search_count"] == 0  # 検索結果が0件
        assert data["used_knowledge"] == False

        # LLMにはナレッジコンテキストが渡されないことを確認
        call_args = mock_llm_model.generate.call_args
        assert call_args[1]["rag_context"] is None

    def test_conversation_minimal_request(self, client):
        """最小限のリクエストで会話をテスト"""
        conversation_data = {"question": "テスト質問"}

        response = client.post("/api/v1/conversation", json=conversation_data)

        if response.status_code != 200:
            logger.error(f"Response status: {response.status_code}")
            logger.error(f"Response body: {response.text}")

        assert response.status_code == 200

    def test_conversation_missing_question(self, client):
        """質問が不足している場合のエラー処理"""
        conversation_data = {"max_tokens": 256}

        response = client.post("/api/v1/conversation", json=conversation_data)
        assert response.status_code == 422  # バリデーションエラー

    def test_conversation_embedding_error(self, client, mock_embedding_model):
        """Embeddingエラーの処理"""
        mock_embedding_model.encode.side_effect = Exception("Embeddingエラー")

        conversation_data = {"question": "エラーテスト"}
        response = client.post("/api/v1/conversation", json=conversation_data)

        assert response.status_code == 500
        assert "会話生成エラー" in response.json()["detail"]

    def test_conversation_llm_error(self, client, mock_llm_model):
        """LLM生成エラーの処理"""
        mock_llm_model.generate.side_effect = Exception("LLMエラー")

        conversation_data = {"question": "エラーテスト"}
        response = client.post("/api/v1/conversation", json=conversation_data)

        assert response.status_code == 500
        assert "会話生成エラー" in response.json()["detail"]


class TestDeleteDocumentEndpoint:
    """ドキュメント削除エンドポイントのテスト"""

    def test_delete_document_success(self, client, mock_vector_store):
        """正常なドキュメント削除が動作することを確認"""
        document_id = "test_doc_1"
        response = client.delete(f"/api/v1/documents/{document_id}")

        assert response.status_code == 200
        data = response.json()

        # レスポンスの内容を確認
        assert data["message"] == "ドキュメント test_doc_1 を削除しました"
        assert data["result"] == "deleted"

        # モックが正しく呼び出されたことを確認
        mock_vector_store.delete_document.assert_called_once_with(
            "knowledge-base", document_id
        )

    def test_delete_document_not_found(self, client, mock_vector_store):
        """存在しないドキュメントの削除エラー処理"""
        # ドキュメントが見つからない場合をシミュレート
        mock_vector_store.delete_document.return_value = {
            "error": "ドキュメントが見つかりません: nonexistent_doc"
        }

        document_id = "nonexistent_doc"
        response = client.delete(f"/api/v1/documents/{document_id}")

        assert response.status_code == 404
        assert "ドキュメントが見つかりません" in response.json()["detail"]

    def test_delete_document_server_error(self, client, mock_vector_store):
        """削除時のサーバーエラー処理"""
        # サーバーエラーをシミュレート
        mock_vector_store.delete_document.return_value = {
            "error": "削除エラー: サーバー内部エラー"
        }

        document_id = "error_doc"
        response = client.delete(f"/api/v1/documents/{document_id}")

        assert response.status_code == 500
        assert "削除エラー" in response.json()["detail"]

    def test_delete_document_exception(self, client, mock_vector_store):
        """削除時の例外処理"""
        # 例外をシミュレート
        mock_vector_store.delete_document.side_effect = Exception("予期しないエラー")

        document_id = "exception_doc"
        response = client.delete(f"/api/v1/documents/{document_id}")

        assert response.status_code == 500
        assert "ドキュメント削除エラー" in response.json()["detail"]


# 実際のAPIが起動していない状態でのテスト実行時の設定
@pytest.fixture(autouse=True)
def mock_global_objects():
    """グローバルオブジェクトのモック（自動適用）"""
    with (
        patch("app.embedding_model") as mock_emb,
        patch("app.vector_store") as mock_vec,
        patch("app.llm_model") as mock_llm,
    ):
        # デフォルトの動作を設定
        mock_emb.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_emb.get_embedding_dimension.return_value = 4

        mock_vec.add_documents.return_value = (1, [])
        mock_vec.search.return_value = []
        mock_vec.get_index_stats.return_value = {
            "document_count": 0,
            "size_bytes": 0,
            "size_mb": 0.0,
        }
        mock_vec.delete_document.return_value = {
            "message": "ドキュメント default_doc を削除しました",
            "result": "deleted",
        }

        mock_llm.model_type = "gguf"
        mock_llm.model_size = "1b"
        mock_llm.generate.return_value = "デフォルトのLLM応答"
        mock_llm.extract_tag.return_value = {
            "tag": None,
            "timestamp": None,
            "content": "デフォルトの質問",
        }

        yield mock_emb, mock_vec, mock_llm
