import time
from typing import List, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class PlamoEmbedding:
    def __init__(self, model_path: str = "./plamo-embedding-1b"):
        print(f"PlamoEmbeddingモデルを初期化中: {model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"デバイス: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("PlamoEmbeddingモデルの初期化完了")

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        start_time = time.perf_counter()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.cpu().numpy()

        end_time = time.perf_counter()
        print(f"Embedding生成完了: {len(texts)}件のテキスト、{end_time - start_time:.2f}秒")

        return embeddings

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size


if __name__ == "__main__":
    embedding_model = PlamoEmbedding()

    test_texts = [
        "これはテストの文章です。",
        "日本語のembeddingを生成します。",
        "ナレッジベースシステムのためのベクトル化処理です。"
    ]

    embeddings = embedding_model.encode(test_texts)
    print(f"生成されたembeddingの形状: {embeddings.shape}")
    print(f"Embedding次元数: {embedding_model.get_embedding_dimension()}")
