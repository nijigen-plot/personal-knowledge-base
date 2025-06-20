import pytest
import pytest_asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import numpy as np

from app import app


@pytest.fixture
def mock_embedding_model():
    """エンベディングモデルのモック"""
    mock_model = Mock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])  # 4次元のダミーベクトル
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
            "metadata": {"type": "test"},
            "timestamp": 1640995200
        }
    ]
    mock_store.get_index_stats.return_value = {
        "document_count": 1,
        "size_bytes": 1024,
        "size_mb": 0.001
    }
    return mock_store


@pytest.fixture
def client(mock_embedding_model, mock_vector_store):
    """テスト用のFastAPIクライアント"""
    # テスト用のアプリケーションを作成（lifespanなし）
    from fastapi import FastAPI
    
    test_app = FastAPI(
        title="テスト用ナレッジベースAPI",
        description="テスト用API",
        version="1.0.0"
    )
    
    # 元のアプリからルートをコピー
    for route in app.routes:
        if hasattr(route, 'path') and route.path not in ['/openapi.json', '/docs', '/docs/oauth2-redirect', '/redoc']:
            test_app.routes.append(route)
    
    # モックをグローバル変数に注入
    with patch('app.embedding_model', mock_embedding_model), \
         patch('app.vector_store', mock_vector_store):
        
        with TestClient(test_app) as test_client:
            yield test_client


class TestRootEndpoint:
    """ルートエンドポイントのテスト"""
    
    def test_root_endpoint(self, client):
        """ルートエンドポイントが正しく動作することを確認"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "ナレッジベースAPIへようこそ"}


class TestSearchEndpoint:
    """検索エンドポイントのテスト"""
    
    def test_search_success(self, client, mock_embedding_model, mock_vector_store):
        """正常な検索が動作することを確認"""
        response = client.get("/search?q=テスト&k=5")
        
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
        assert result["metadata"]["type"] == "test"
        
        # モックが正しく呼び出されたことを確認
        mock_embedding_model.encode.assert_called_once_with(["テスト"])
        mock_vector_store.search.assert_called_once()
    
    def test_search_with_min_score_filter(self, client, mock_vector_store):
        """最小スコアフィルタリングが動作することを確認"""
        # 低いスコアの結果を返すようにモックを設定
        mock_vector_store.search.return_value = [
            {
                "id": "low_score_doc",
                "score": 0.3,
                "content": "低スコアドキュメント",
                "metadata": {},
                "timestamp": 1640995200
            }
        ]
        
        # min_scoreを0.5に設定（結果の0.3より高い）
        response = client.get("/search?q=テスト&min_score=0.5")
        
        assert response.status_code == 200
        data = response.json()
        
        # フィルタリングされて結果が0件になることを確認
        assert len(data) == 0
    
    def test_search_missing_query(self, client):
        """クエリパラメータが不足している場合のエラー処理"""
        response = client.get("/search")
        assert response.status_code == 422  # バリデーションエラー
    
    def test_search_default_parameters(self, client):
        """デフォルトパラメータでの検索が動作することを確認"""
        response = client.get("/search?q=テスト")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestAddEndpoint:
    """ドキュメント追加エンドポイントのテスト"""
    
    def test_add_document_success(self, client, mock_embedding_model, mock_vector_store):
        """正常なドキュメント追加が動作することを確認"""
        document_data = {
            "content": "新しいテストドキュメント",
            "metadata": {"category": "test", "author": "pytest"}
        }
        
        response = client.post("/add", json=document_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # レスポンスの内容を確認
        assert data["message"] == "ドキュメントが正常に追加されました"
        assert "processing_time" in data
        assert data["embedding_dimension"] == 4
        
        # モックが正しく呼び出されたことを確認
        mock_embedding_model.encode.assert_called_once_with(["新しいテストドキュメント"])
        mock_vector_store.add_documents.assert_called_once()
    
    def test_add_document_with_minimal_data(self, client):
        """最小限のデータでドキュメント追加"""
        document_data = {"content": "最小限のドキュメント"}
        
        response = client.post("/add", json=document_data)
        assert response.status_code == 200
    
    def test_add_document_missing_content(self, client):
        """コンテンツが不足している場合のエラー処理"""
        document_data = {"metadata": {"test": "value"}}
        
        response = client.post("/add", json=document_data)
        assert response.status_code == 422  # バリデーションエラー
    
    def test_add_document_storage_failure(self, client, mock_vector_store):
        """ストレージ失敗の場合のエラー処理"""
        # 保存失敗をシミュレート
        mock_vector_store.add_documents.return_value = (0, ["failed_item"])
        
        document_data = {"content": "失敗するドキュメント"}
        
        response = client.post("/add", json=document_data)
        assert response.status_code == 500
        assert "ドキュメントの保存に失敗しました" in response.json()["detail"]


class TestStatsEndpoint:
    """統計エンドポイントのテスト"""
    
    def test_get_stats_success(self, client, mock_vector_store):
        """正常な統計取得が動作することを確認"""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # レスポンスの構造を確認
        assert data["index_name"] == "knowledge-base"
        assert data["document_count"] == 1
        assert data["size_bytes"] == 1024
        assert data["size_mb"] == 0.001
        
        # モックが正しく呼び出されたことを確認
        mock_vector_store.get_index_stats.assert_called_once_with("knowledge-base")
    
    def test_get_stats_index_not_found(self, client, mock_vector_store):
        """インデックスが見つからない場合のエラー処理"""
        mock_vector_store.get_index_stats.return_value = {"error": "インデックスが見つかりません"}
        
        response = client.get("/stats")
        assert response.status_code == 404


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def test_search_embedding_error(self, client, mock_embedding_model):
        """エンベディング生成エラーの処理"""
        mock_embedding_model.encode.side_effect = Exception("エンベディングエラー")
        
        response = client.get("/search?q=エラーテスト")
        assert response.status_code == 500
        assert "検索エラー" in response.json()["detail"]
    
    def test_add_document_embedding_error(self, client, mock_embedding_model):
        """ドキュメント追加時のエンベディングエラー処理"""
        mock_embedding_model.encode.side_effect = Exception("エンベディングエラー")
        
        document_data = {"content": "エラーテストドキュメント"}
        response = client.post("/add", json=document_data)
        
        assert response.status_code == 500
        assert "追加エラー" in response.json()["detail"]


# 実際のAPIが起動していない状態でのテスト実行時の設定
@pytest.fixture(autouse=True)
def mock_global_objects():
    """グローバルオブジェクトのモック（自動適用）"""
    with patch('app.embedding_model') as mock_emb, \
         patch('app.vector_store') as mock_vec:
        
        # デフォルトの動作を設定
        mock_emb.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_emb.get_embedding_dimension.return_value = 4
        
        mock_vec.add_documents.return_value = (1, [])
        mock_vec.search.return_value = []
        mock_vec.get_index_stats.return_value = {
            "document_count": 0,
            "size_bytes": 0,
            "size_mb": 0.0
        }
        
        yield mock_emb, mock_vec