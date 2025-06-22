import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from embedding_model import PlamoEmbedding
from opensearch_client import OpenSearchVectorStore

load_dotenv('.env')

class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10
    filter_query: Optional[Dict] = None


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    timestamp: int


class IndexStats(BaseModel):
    document_count: int
    size_bytes: int
    size_mb: float


embedding_model = None
vector_store = None
INDEX_NAME = "knowledge-base"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, vector_store

    print("ナレッジベースAPIを初期化中...")

    embedding_model = PlamoEmbedding("./plamo-embedding-1b")

    opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
    opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_pass = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD", "admin")

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
    description="PlamoEmbeddingとOpenSearchを使用したドキュメント保存・検索API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "ナレッジベースAPIへようこそ"}


@app.post("/documents")
async def add_document(request: DocumentRequest):
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
            "embedding_dimension": embedding.shape[1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@app.post("/documents/batch")
async def add_documents_batch(requests: List[DocumentRequest]):
    try:
        start_time = time.perf_counter()

        contents = [req.content for req in requests]
        embeddings = embedding_model.encode(contents)

        documents = [
            {
                "content": req.content,
                "metadata": req.metadata
            }
            for req in requests
        ]

        success_count, failed_items = vector_store.add_documents(
            INDEX_NAME,
            documents,
            embeddings
        )

        end_time = time.perf_counter()

        return {
            "message": f"{success_count}件のドキュメントが追加されました",
            "success_count": success_count,
            "failed_count": len(failed_items) if failed_items else 0,
            "processing_time": round(end_time - start_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    try:
        start_time = time.perf_counter()

        query_embedding = embedding_model.encode([request.query])

        results = vector_store.search(
            INDEX_NAME,
            query_embedding[0],
            k=request.k,
            filter_query=request.filter_query
        )

        end_time = time.perf_counter()

        search_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                content=result["content"],
                metadata=result["metadata"],
                timestamp=result["timestamp"]
            )
            for result in results
        ]

        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {str(e)}")


@app.get("/stats", response_model=IndexStats)
async def get_index_stats():
    try:
        stats = vector_store.get_index_stats(INDEX_NAME)

        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])

        return IndexStats(
            document_count=stats["document_count"],
            size_bytes=stats["size_bytes"],
            size_mb=stats["size_mb"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計取得エラー: {str(e)}")


@app.delete("/index")
async def reset_index():
    try:
        vector_store.delete_index(INDEX_NAME)

        embedding_dim = embedding_model.get_embedding_dimension()
        vector_store.create_index(INDEX_NAME, embedding_dim)

        return {"message": "インデックスがリセットされました"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"インデックスリセットエラー: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.txt', '.md')):
            raise HTTPException(
                status_code=400,
                detail="サポートされているファイル形式: .txt, .md"
            )

        content = await file.read()
        text_content = content.decode('utf-8')

        chunks = []
        lines = text_content.split('\n')
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) < 1000:
                current_chunk += line + "\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if not chunks:
            raise HTTPException(status_code=400, detail="ファイルにテキストが含まれていません")

        embeddings = embedding_model.encode(chunks)

        documents = [
            {
                "content": chunk,
                "metadata": {
                    "source_file": file.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ]

        success_count, failed_items = vector_store.add_documents(
            INDEX_NAME,
            documents,
            embeddings
        )

        return {
            "message": f"ファイル '{file.filename}' から {success_count}個のチャンクが追加されました",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "chunks_saved": success_count,
            "failed_chunks": len(failed_items) if failed_items else 0
        }

    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="ファイルの文字エンコーディングが無効です（UTF-8が必要）")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイルアップロードエラー: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
