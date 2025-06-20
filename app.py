from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import os
from contextlib import asynccontextmanager

from embedding_model import PlamoEmbedding
from opensearch_client import OpenSearchVectorStore


class AddDocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    timestamp: int


embedding_model = None
vector_store = None
INDEX_NAME = "knowledge-base"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, vector_store
    
    print("ナレッジベースAPIを初期化中...")
    
    embedding_model = PlamoEmbedding("pfnet/plamo-embedding-1b")
    
    opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
    opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_pass = os.getenv("OPENSEARCH_PASS", "admin")
    
    vector_store = OpenSearchVectorStore(
        host=opensearch_host,
        port=opensearch_port,  
        username=opensearch_user,
        password=opensearch_pass
    )
    
    embedding_dim = embedding_model.get_embedding_dimension()
    vector_store.create_index(INDEX_NAME, embedding_dim)
    
    print("ナレッジベースAPI初期化完了")
    yield
    
    print("ナレッジベースAPIをシャットダウン中...")


app = FastAPI(
    title="ナレッジベースAPI",
    description="PlamoEmbeddingとOpenSearchを使用した問い合わせ・追加API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "ナレッジベースAPIへようこそ"}


@app.get("/search", response_model=List[SearchResult])
async def search_documents(
    q: str = Query(..., description="検索クエリ"),
    k: int = Query(10, description="取得する結果数"),
    min_score: float = Query(0.0, description="最小スコア閾値")
):
    """
    GET リクエストでドキュメントを検索します
    
    - **q**: 検索したいクエリ文字列
    - **k**: 取得する結果の最大数（デフォルト: 10）
    - **min_score**: 最小スコア閾値（デフォルト: 0.0）
    """
    try:
        start_time = time.perf_counter()
        
        query_embedding = embedding_model.encode([q])
        
        results = vector_store.search(
            INDEX_NAME,
            query_embedding[0],
            k=k
        )
        
        # 最小スコア閾値でフィルタリング
        filtered_results = [
            result for result in results 
            if result["score"] >= min_score
        ]
        
        end_time = time.perf_counter()
        processing_time = round(end_time - start_time, 2)
        
        search_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                content=result["content"],
                metadata=result["metadata"],
                timestamp=result["timestamp"]
            )
            for result in filtered_results
        ]
        
        return search_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {str(e)}")


@app.post("/add")
async def add_document(request: AddDocumentRequest):
    """
    新しいドキュメントを追加します
    
    - **content**: 追加するドキュメントの内容
    - **metadata**: オプションのメタデータ
    """
    try:
        start_time = time.perf_counter()
        
        embedding = embedding_model.encode([request.content])
        
        document = {
            "content": request.content,
            "metadata": request.metadata
        }
        
        success_count, failed_items = vector_store.add_documents(
            INDEX_NAME, 
            [document], 
            embedding
        )
        
        end_time = time.perf_counter()
        
        if failed_items:
            raise HTTPException(status_code=500, detail="ドキュメントの保存に失敗しました")
        
        return {
            "message": "ドキュメントが正常に追加されました",
            "processing_time": round(end_time - start_time, 2),
            "embedding_dimension": embedding.shape[1],
            "document_id": success_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"追加エラー: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    インデックスの統計情報を取得します
    """
    try:
        stats = vector_store.get_index_stats(INDEX_NAME)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return {
            "index_name": INDEX_NAME,
            "document_count": stats["document_count"],
            "size_bytes": stats["size_bytes"],
            "size_mb": stats["size_mb"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計取得エラー: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)