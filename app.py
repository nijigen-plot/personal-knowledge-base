import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from embedding_model import PlamoEmbedding
from llm import LargeLanguageModel
from log_config import get_logger
from opensearch_client import OpenSearchVectorStore

load_dotenv(".env")

logger = get_logger(__name__)


class DocumentRequest(BaseModel):
    content: str
    timestamp: str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    tag: Literal["lifestyle", "music", "technology"]


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10
    tag_filter: Optional[Literal["lifestyle", "music", "technology"]] = None
    timestamp_filter: Optional[dict[str, str]] = None


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    tag: Optional[Literal["lifestyle", "music", "technology"]]
    timestamp: str


class IndexStats(BaseModel):
    document_count: int
    size_bytes: int
    size_mb: float


class ConversationRequest(BaseModel):
    question: str
    use_history: Optional[bool] = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    search_k: Optional[int] = 10
    debug: Optional[bool] = False


class ConversationResponse(BaseModel):
    question: str
    answer: str
    search_results: Optional[List[SearchResult]] = None
    search_count: int
    used_knowledge: bool
    processing_time: float
    model_type: str
    model_size: str


embedding_model = None
vector_store = None
llm_model = None
INDEX_NAME = "knowledge-base"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, vector_store, llm_model

    logger.info("ナレッジベースAPIを初期化中...")

    # Embeddingモデル初期化
    embedding_model = PlamoEmbedding("./plamo-embedding-1b")

    # OpenSearch初期化
    opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
    opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    opensearch_user = os.getenv("OPENSEARCH_USER", "admin")
    opensearch_pass = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD", "admin")

    vector_store = OpenSearchVectorStore(
        host=opensearch_host,
        port=opensearch_port,
        username=opensearch_user,
        password=opensearch_pass,
    )

    embedding_dim = embedding_model.get_embedding_dimension()
    vector_store.create_index(INDEX_NAME, embedding_dim)

    llm_model = LargeLanguageModel()

    logger.info("ナレッジベースAPI初期化完了")
    yield

    logger.info("ナレッジベースAPIをシャットダウン中...")


app = FastAPI(
    title="ナレッジベース API",
    description="""# ナレッジベース API

このAPIは[@Quarkgabber](https://quark-hardcore.com/)の日常が分かるナレッジベースです。


## 使用例

### RAG会話
```bash
curl -X POST "https://home.quark-hardcore/personal-knowledge-base/api/v1/conversation" \
  -H "Content-Type: application/json" \
  -d '{"question": "最近あった出来事は？"}'
```

## 機能詳細
- **PlamoEmbedding** (pfnet/plamo-embedding-1b) を使用したドキュメントベクトル化
- **OpenSearch** を使用した高速ベクトル検索(RAG)
- **Gemma 3** モデルを使用した対話

    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    openapi_tags=[
        {
            "name": "system",
            "description": "システム情報",
        },
        {
            "name": "documents",
            "description": "ドキュメント管理",
        },
        {
            "name": "search",
            "description": "ベクトル検索",
        },
        {
            "name": "conversation",
            "description": "RAG会話",
        },
        {
            "name": "admin",
            "description": "管理機能",
        },
    ],
)

security = HTTPBearer(
    scheme_name="Bearer Token",
    description="AdminだけがリクエストできるAPIにアクセスするためのキーを設定",
    auto_error=True,
)


def require_json_content_type(content_type: str = Header(..., alias="content-type")):
    if content_type.lower() != "application/json":
        raise HTTPException(
            status_code=400, detail="Content-Type must be application/json"
        )
    return content_type


@staticmethod
def check_api_key(api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    expected_api_key = os.getenv("ADMIN_API_KEY")
    if not expected_api_key:
        raise HTTPException(status_code=500, detail="API key is not configured")
    if api_key != expected_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return check_api_key(credentials.credentials)


# APIルーターを作成
api_router = APIRouter(prefix="/api/v1")


@api_router.get("/", tags=["system"])
async def root():
    return {
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


@api_router.head("/health", tags=["system"])
async def health_check():
    return {"status": "ok"}


@api_router.post("/documents", tags=["documents"])
async def add_document(
    request: DocumentRequest,
    content_type: str = Depends(require_json_content_type),
    key: str = Depends(verify_api_key),
):
    try:
        start_time = time.perf_counter()

        embedding = embedding_model.encode(request.content)
        document = {
            "tag": request.tag,
            "timestamp": request.timestamp,
            "content": request.content,
        }

        success_count, failed_items = vector_store.add_documents(
            INDEX_NAME, [document], embedding
        )

        end_time = time.perf_counter()

        if failed_items:
            raise HTTPException(
                status_code=500, detail="ドキュメントの保存に失敗しました"
            )

        return {
            "message": "ドキュメントが正常に追加されました",
            "processing_time": round(end_time - start_time, 2),
            "embedding_dimension": embedding.shape[1],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


@api_router.post("/documents/batch", tags=["documents"])
async def add_documents_batch(
    requests: List[DocumentRequest],
    content_type: str = Depends(require_json_content_type),
    key: str = Depends(verify_api_key),
):
    try:
        start_time = time.perf_counter()

        total_success_count = 0
        total_failed_items = []

        # 50個ずつに分割して処理
        batch_size = 50
        for i in range(0, len(requests), batch_size):
            batch_requests = requests[i : i + batch_size]

            contents = [req.content for req in batch_requests]
            embeddings = embedding_model.encode(contents)

            documents = [
                {"tag": req.tag, "timestamp": req.timestamp, "content": req.content}
                for req in batch_requests
            ]

            success_count, failed_items = vector_store.add_documents(
                INDEX_NAME, documents, embeddings
            )

            total_success_count += success_count
            if failed_items:
                total_failed_items.extend(failed_items)

        end_time = time.perf_counter()

        return {
            "message": f"{total_success_count}件のドキュメントが追加されました",
            "success_count": total_success_count,
            "failed_count": len(total_failed_items),
            "processing_time": round(end_time - start_time, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラー: {str(e)}")


# SearchResultのtimestampを使えていない。後程考える
@api_router.post("/search", response_model=List[SearchResult], tags=["search"])
async def search_documents(
    request: SearchRequest, content_type: str = Depends(require_json_content_type)
):
    try:
        start_time = time.perf_counter()

        query_embedding = embedding_model.encode([request.query])

        results = vector_store.search(
            INDEX_NAME,
            query_embedding[0],
            k=request.k,
            tag_filter=request.tag_filter,
            timestamp_filter=request.timestamp_filter,
        )

        end_time = time.perf_counter()

        search_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                content=result["content"],
                tag=result["tag"],
                timestamp=result["timestamp"],
            )
            for result in results
        ]

        return search_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"検索エラー: {str(e)}")


@api_router.get("/stats", response_model=IndexStats, tags=["admin"])
async def get_index_stats(key: str = Depends(verify_api_key)):
    try:
        stats = vector_store.get_index_stats(INDEX_NAME)

        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])

        return IndexStats(
            document_count=stats["document_count"],
            size_bytes=stats["size_bytes"],
            size_mb=stats["size_mb"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"統計取得エラー: {str(e)}")


@api_router.delete("/index", tags=["admin"])
async def reset_index(key: str = Depends(verify_api_key)):
    try:
        vector_store.delete_index(INDEX_NAME)

        embedding_dim = embedding_model.get_embedding_dimension()
        vector_store.create_index(INDEX_NAME, embedding_dim, force_recreate=True)

        return {"message": "インデックスがリセットされました"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"インデックスリセットエラー: {str(e)}"
        )


@api_router.delete("/documents/{document_id}", tags=["documents"])
async def delete_document(document_id: str, key: str = Depends(verify_api_key)):
    """特定のドキュメントIDのドキュメントを削除"""
    try:
        result = vector_store.delete_document(INDEX_NAME, document_id)

        if "error" in result:
            if "見つかりません" in result["error"]:
                raise HTTPException(status_code=404, detail=result["error"])
            else:
                raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ドキュメント削除エラー: {str(e)}")


@api_router.post(
    "/conversation", response_model=ConversationResponse, tags=["conversation"]
)
async def conversation_with_rag(
    request: ConversationRequest, content_type: str = Depends(require_json_content_type)
):
    """
    RAGを使用した会話: 質問→Embedding→OpenSearch検索→LLMが回答
    """
    try:
        total_start_time = time.perf_counter()

        # 1. 質問からタグとタイムスタンプを抽出
        extracted_filter = llm_model.extract_tag(text=request.question)
        # 2. 質問をEmbeddingモデルでベクトル化
        query_embedding = embedding_model.encode([request.question])

        # 3. OpenSearchで近傍探索
        search_results = vector_store.search(
            INDEX_NAME,
            query_embedding[0],
            k=request.search_k,
            tag_filter=extracted_filter.get("tag", None),
            timestamp_filter=extracted_filter.get("timestamp", None),
        )

        # 4. 検索結果をナレッジコンテキストとして整形
        if search_results:
            knowledge_context = "\n\n".join(
                [
                    f"【参考情報 {i+1}】（{result['timestamp']}）\n{result['content']}"
                    for i, result in enumerate(search_results)
                ]
            )
            used_knowledge = True
        else:
            knowledge_context = None
            used_knowledge = False

        # 5. LLMで回答生成
        answer = llm_model.generate(
            prompt=request.question,
            rag_context=knowledge_context,
            use_history=request.use_history,
            stream=True,  # streamはTrueに固定
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            silent=True,  # API用にログ出力を抑制
        )

        total_end_time = time.perf_counter() - total_start_time

        # 6. 検索結果をSearchResultモデルに変換（debug時のみ）
        search_result_models = None
        if request.debug:
            search_result_models = [
                SearchResult(
                    id=result["id"],
                    score=result["score"],
                    content=result["content"],
                    tag=result["tag"],
                    timestamp=result["timestamp"],
                )
                for result in search_results
            ]

        return ConversationResponse(
            question=request.question,
            answer=answer,
            search_results=search_result_models,
            search_count=len(search_results),
            used_knowledge=used_knowledge,
            processing_time=round(total_end_time, 2),
            model_type=llm_model.model_type,
            model_size=llm_model.model_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"会話生成エラー: {str(e)}")


# APIルーターをアプリに追加
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8050)),
    )
