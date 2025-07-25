import gc
import os
import time
from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from log_config import get_logger

# メモリ解放を積極的に行う設定
os.environ["MALLOC_TRIM_THRESHOLD_"] = "-1"

logger = get_logger(__name__)


class PlamoEmbedding:
    def __init__(self, model_path: str = "./plamo-embedding-1b"):
        logger.info(f"PlamoEmbeddingモデルを初期化中: {model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"デバイス: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()  # モデルを評価モードに設定
        logger.info("PlamoEmbeddingモデルの初期化完了")

    def encode(
        self, texts: Union[str, List[str]], normalize: bool = True
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        start_time = time.perf_counter()
        document_embeddings = self.model.encode_document(
            sentences=texts, tokenizer=self.tokenizer
        )
        end_time = time.perf_counter()
        logger.info(
            f"Embedding生成完了: {len(texts)}件のテキスト、{end_time - start_time:.2f}秒"
        )

        return document_embeddings

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def clear_memory(self):
        """メモリを明示的に解放"""
        logger.info("PlamoEmbeddingモデルのメモリを解放中...")

        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer

        # Python ガベージコレクション
        gc.collect()
        logger.info("PlamoEmbeddingモデルのメモリ解放完了")

    def __del__(self):
        """デストラクタでメモリ解放"""
        try:
            self.clear_memory()
        except:
            pass  # エラーが発生してもデストラクタでは例外を投げない


if __name__ == "__main__":
    embedding_model = PlamoEmbedding()

    test_texts = [
        "これはテストの文章です。",
        "日本語のembeddingを生成します。",
        "ナレッジベースシステムのためのベクトル化処理です。",
    ]

    embeddings = embedding_model.encode(test_texts)
    logger.info(f"生成されたembeddingの形状: {embeddings.shape}")
    logger.info(f"Embedding次元数: {embedding_model.get_embedding_dimension()}")
