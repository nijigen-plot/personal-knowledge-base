import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

from log_config import get_logger

load_dotenv(".env")

logger = get_logger(__name__)


class OpenSearchVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        username: str = "admin",
        password: str = "admin",
        use_ssl: bool = True,
        verify_certs: bool = False,
    ):

        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(username, password),
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

        logger.info(f"OpenSearchクライアント初期化完了: {host}:{port}")

        try:
            info = self.client.info()
            logger.info(f"OpenSearch接続確認: {info['version']['number']}")
        except Exception as e:
            logger.error(f"OpenSearch接続エラー: {e}")
            raise

    def create_index(
        self, index_name: str, embedding_dimension: int, force_recreate: bool = False
    ):
        if self.client.indices.exists(index=index_name):
            if force_recreate:
                logger.info(f"インデックス削除中: {index_name}")
                self.client.indices.delete(index_name)
            else:
                logger.info(f"インデックスは既に存在します: {index_name}")
                return

        index_mapping = {
            "settings": {"index.knn": True},
            "mappings": {
                "properties": {
                    "content_vector": {
                        "type": "knn_vector",
                        "dimension": embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "faiss",
                        },
                    },
                    "content": {"type": "text"},
                    "tag": {"type": "keyword"},
                    "timestamp": {
                        "type": "date",
                        "format": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
                    },
                }
            },
        }

        self.client.indices.create(index=index_name, body=index_mapping)
        logger.info(f"インデックス作成完了: {index_name}")

    def add_documents(
        self, index_name: str, documents: List[Dict[str, Any]], embeddings: np.ndarray
    ):
        if len(documents) != len(embeddings):
            raise ValueError("ドキュメント数とembedding数が一致しません")

        start_time = time.perf_counter()

        actions = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            action = {
                "_index": index_name,
                "_source": {
                    "content": doc.get("content", ""),
                    "content_vector": embedding.tolist(),
                    "tag": doc.get("tag"),
                    "timestamp": doc.get("timestamp"),
                },
            }
            actions.append(action)

        success_count, failed_items = helpers.bulk(
            self.client, actions, chunk_size=100, timeout=60
        )

        end_time = time.perf_counter()
        logger.info(
            f"ドキュメント追加完了: {success_count}件成功、{end_time - start_time:.2f}秒"
        )

        if failed_items:
            logger.error(f"失敗したアイテム: {len(failed_items)}件")

        return success_count, failed_items

    def search(
        self,
        index_name: str,
        query_embedding: np.ndarray,
        k: int = 10,
        tag_filter: Optional[str] = None,
        timestamp_filter: Optional[dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:

        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "content_vector": {
                        "vector": query_embedding.tolist(),
                        "min_score": 0.5,
                    }
                }
            },
        }

        filters = []
        if tag_filter:
            filters.append({"term": {"tag": tag_filter}})

        if timestamp_filter:
            filters.append({"range": {"timestamp": timestamp_filter}})

        if filters:
            search_body["query"] = {
                "bool": {
                    "must": [search_body["query"]],
                    "filter": filters,
                }
            }

        start_time = time.perf_counter()
        response = self.client.search(index=index_name, body=search_body)
        end_time = time.perf_counter()

        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                "content": hit["_source"]["content"],
                "tag": hit["_source"]["tag"],
                "timestamp": str(
                    datetime.fromisoformat(hit["_source"]["timestamp"])
                    .replace(tzinfo=timezone.utc)
                    .astimezone(timezone(timedelta(hours=9)))
                ),
            }
            results.append(result)

        logger.info(f"検索完了: {len(results)}件ヒット、{end_time - start_time:.2f}秒")
        return results

    def delete_document(self, index_name: str, document_id: str) -> Dict[str, Any]:
        """特定のドキュメントIDのドキュメントを削除"""
        try:
            if not self.client.indices.exists(index_name):
                return {"error": f"インデックスが存在しません: {index_name}"}

            response = self.client.delete(index=index_name, id=document_id)
            logger.info(f"ドキュメント削除完了: {document_id}")
            return {
                "message": f"ドキュメント {document_id} を削除しました",
                "result": response["result"],
            }
        except Exception as e:
            if "not_found" in str(e).lower():
                return {"error": f"ドキュメントが見つかりません: {document_id}"}
            else:
                return {"error": f"削除エラー: {str(e)}"}

    def delete_index(self, index_name: str):
        if self.client.indices.exists(index_name):
            self.client.indices.delete(index_name)
            logger.info(f"インデックス削除完了: {index_name}")
        else:
            logger.info(f"インデックスが存在しません: {index_name}")

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        if not self.client.indices.exists(index_name):
            return {"error": f"インデックスが存在しません: {index_name}"}

        stats = self.client.indices.stats(index_name)
        doc_count = stats["indices"][index_name]["total"]["indexing"]["index_total"]
        size_bytes = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]

        return {
            "document_count": doc_count,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        }


if __name__ == "__main__":
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

    test_documents = [
        {"content": "これは最初のテストドキュメントです。", "tag": "lifestyle"},
        {"content": "二番目のドキュメントです。日本語のテストです。", "tag": "music"},
        {
            "content": "三番目のテストデータです。ベクトル検索の確認用です。",
            "tag": "technology",
        },
    ]

    # 固定のベクトルを作成（テスト用）
    test_vector = np.random.random((1, 768)).astype(np.float32).flatten()
    test_embeddings = np.array([test_vector, test_vector * 0.9, test_vector * 0.8])

    index_name = "test-knowledge-base"

    vector_store.create_index(index_name, 768, force_recreate=True)

    # インデックスが完全に準備されるまで少し待つ
    time.sleep(2)

    vector_store.add_documents(index_name, test_documents, test_embeddings)

    # ドキュメントがインデックスされるまで待つ
    time.sleep(3)

    stats = vector_store.get_index_stats(index_name)
    logger.info(f"インデックス統計: {stats}")

    # 完全に同じベクトルで検索
    logger.info("\n=== 完全に同じベクトルで検索 ===")
    results = vector_store.search(index_name, test_vector, k=3)
    logger.info(f"検索結果: {len(results)}件")
    for i, result in enumerate(results):
        logger.info(
            f"  {i+1}. スコア: {result['score']:.6f}, タグ: {result['tag']}, 内容: {result['content'][:30]}..."
        )

    # 少し違うベクトルで検索
    logger.info("\n=== 少し違うベクトルで検索 ===")
    similar_vector = test_vector * 1.1
    results2 = vector_store.search(index_name, similar_vector, k=3)
    logger.info(f"検索結果: {len(results2)}件")
    for i, result in enumerate(results2):
        logger.info(
            f"  {i+1}. スコア: {result['score']:.6f}, タグ: {result['tag']}, 内容: {result['content'][:30]}..."
        )
