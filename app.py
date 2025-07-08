import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

from embedding_model import PlamoEmbedding
from gemma3 import Gemma3Model
from opensearch_client import OpenSearchVectorStore

load_dotenv(".env")
JST = timezone(timedelta(hours=9), name="JST")


class DocumentRequest(BaseModel):
    content: str
    timestamp: Optional[str] = None
    tag: Literal["lifestyle", "music", "technology"]

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v):
        if v is None:
            # OpenSearchの期待するフォーマット: yyyy-MM-dd'T'HH:mm:ss.SSSSSS
            return datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S.%f")

        # OpenSearchのタイムスタンプフォーマットをバリデーション
        try:
            # yyyy-MM-dd'T'HH:mm:ss.SSSSSS の形式をチェック
            datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%f")
            return v
        except ValueError:
            raise ValueError(
                'timestamp must be in OpenSearch format: "yyyy-MM-ddTHH:mm:ss.SSSSSS" (e.g., "2024-01-01T10:00:00.123456")'
            )


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 10
    tag_filter: Optional[Literal["lifestyle", "music", "technology"]] = None


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
    use_history: Optional[bool] = True
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    search_k: Optional[int] = 5
    min_score: Optional[float] = 0.5


class ConversationResponse(BaseModel):
    question: str
    answer: str
    search_results: List[SearchResult]
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

    print("ナレッジベースAPIを初期化中...")

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

    # LLMモデル初期化（デフォルト：GGUF）
    llm_model_type = os.getenv("LLM_MODEL_TYPE", "gguf")
    llm_model_size = os.getenv("LLM_MODEL_SIZE", "1b")
    llm_model = Gemma3Model(model_type=llm_model_type, model_size=llm_model_size)

    print("ナレッジベースAPI初期化完了")
    yield

    print("ナレッジベースAPIをシャットダウン中...")


app = FastAPI(
    title="ナレッジベース API",
    description="""# ナレッジベース API

このAPIは[@Quarkgabber](https://quark-hardcore.com/)の日常が分かるナレッジベースです。


## 使用例

### RAG会話
```bash
curl -X POST "https://home.quark-hardcore/personal-knowledge-base/conversation" \
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
    openapi_tags=[
        {
            "name": "documents",
            "description": "ドキュメントの追加",
        },
        {
            "name": "documents/batch",
            "description": "ドキュメントの一括追加",
        },
        {
            "name": "search",
            "description": "OpenSearchへの検索リクエスト",
        },
        {
            "name": "conversation",
            "description": "LLM+RAGによる私との会話",
        },
        {
            "name": "index",
            "description": "indexの削除",
        },
        {
            "name": "stats",
            "description": "インデックスの文書量等統計情報",
        },
    ],
)

header_scheme = APIKeyHeader(
    name="admin-api-key",
    scheme_name="Admin API Key",
    description="一部アクセスを制限するためのAPIキー",
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


def verify_api_key(api_key: str = Depends(header_scheme)):
    return check_api_key(api_key)


@app.get("/")
async def root():
    return {"message": "ナレッジベースAPIへようこそ"}


@app.head("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/documents")
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


@app.post("/documents/batch")
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
@app.post("/search", response_model=List[SearchResult])
async def search_documents(
    request: SearchRequest,
    content_type: str = Depends(require_json_content_type),
    key: str = Depends(verify_api_key),
):
    try:
        start_time = time.perf_counter()

        query_embedding = embedding_model.encode([request.query])

        results = vector_store.search(
            INDEX_NAME,
            query_embedding[0],
            k=request.k,
            tag_filter=request.tag_filter,
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


@app.get("/stats", response_model=IndexStats)
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


@app.delete("/index")
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


@app.post("/conversation", response_model=ConversationResponse)
async def conversation_with_rag(
    request: ConversationRequest, content_type: str = Depends(require_json_content_type)
):
    """
    RAGを使用した会話: 質問→Embedding→OpenSearch検索→LLMが回答
    """
    try:
        total_start_time = time.perf_counter()

        # 1. 質問をEmbeddingモデルでベクトル化
        search_start_time = time.perf_counter()
        query_embedding = embedding_model.encode([request.question])

        # 2. OpenSearchで近傍探索
        search_results = vector_store.search(
            INDEX_NAME, query_embedding[0], k=request.search_k
        )

        # 3. スコアフィルタリング
        filtered_results = [
            result
            for result in search_results
            if result.get("score", 0) >= request.min_score
        ]

        search_end_time = time.perf_counter() - search_start_time

        # 4. 検索結果をナレッジコンテキストとして整形
        if filtered_results:
            knowledge_context = "\n\n".join(
                [
                    f"【参考情報 {i+1}】\n{result['content']}"
                    for i, result in enumerate(filtered_results)
                ]
            )
            used_knowledge = True
        else:
            knowledge_context = None
            used_knowledge = False

        # 5. LLMで回答生成
        llm_start_time = time.perf_counter()
        answer = llm_model.generate(
            prompt=request.question,
            use_history=request.use_history,
            stream=True,  # streamはTrueに固定
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            knowledge_context=knowledge_context,
            silent=True,  # API用にログ出力を抑制
        )
        llm_end_time = time.perf_counter() - llm_start_time

        total_end_time = time.perf_counter() - total_start_time

        # 6. 検索結果をSearchResultモデルに変換
        search_result_models = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                content=result["content"],
                tag=result["tag"],
                timestamp=result["timestamp"],
            )
            for result in filtered_results
        ]

        return ConversationResponse(
            question=request.question,
            answer=answer,
            search_results=search_result_models,
            search_count=len(filtered_results),
            used_knowledge=used_knowledge,
            processing_time=round(total_end_time, 2),
            model_type=llm_model.model_type,
            model_size=llm_model.model_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"会話生成エラー: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8050)),
    )
