import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from llama_cpp import Llama
from transformers import pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

HISTORY_FILE = "history.txt"


class Gemma3Model:
    def __init__(self, model_type: str = "gguf", model_size: str = "1b"):
        """
        Gemma3モデルの統合クラス

        Args:
            model_type: "gguf" または "pytorch"
            model_size: "1b" または "4b" (pytorchの場合のみ)
        """
        self.model_type = model_type
        self.model_size = model_size
        self.model = None

        if model_type == "gguf":
            self._load_gguf_model()
        elif model_type == "pytorch":
            self._load_pytorch_model()
        else:
            raise ValueError("model_typeは'gguf'または'pytorch'を指定してください")

    def _load_gguf_model(self):
        """GGUF量子化モデルを読み込み"""
        model_path = "./gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf"
        logger.info(f"GGUFモデルを読み込み中: {model_path}")

        self.model = Llama(
            model_path=model_path,
            verbose=False,
            n_ctx=32768,
        )
        logger.info("GGUFモデルの読み込み完了")

    def _load_pytorch_model(self):
        """PyTorchモデルを読み込み"""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_size == "1b":
            model_path = "./gemma-3-1b-it"
        elif self.model_size == "4b":
            model_path = "./gemma-3-4b-it"
        else:
            raise ValueError("model_sizeは'1b'または'4b'を指定してください")

        logger.info(f"PyTorchモデルを読み込み中: {model_path} (device: {device})")

        self.model = pipeline(
            task="text-generation",
            model=model_path,
            device=device,
            torch_dtype="auto",
        )
        logger.info("PyTorchモデルの読み込み完了")

    def _load_history(self, n_ctx: int = 32768) -> str:
        """会話履歴を読み込み"""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                memory_file = f.read()
            memory = memory_file[-n_ctx:]
            logger.info(f"{HISTORY_FILE} から過去の会話履歴を読み込みました。")
            return memory
        else:
            logger.info(
                f"{HISTORY_FILE} が存在しないため、過去の会話履歴は考慮されません。"
            )
            return ""

    def _save_memory(self, message: str):
        """会話履歴を保存"""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def generate(
        self,
        prompt: str,
        use_history: bool = True,
        stream: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.3,
        knowledge_context: Optional[str] = None,
        silent: bool = False,
    ) -> str:
        """
        テキスト生成

        Args:
            prompt: ユーザーからの質問
            use_history: 会話履歴を使用するか
            stream: ストリーミング出力するか
            max_tokens: 最大トークン数
            temperature: 生成の温度パラメータ
            knowledge_context: ナレッジベースからの追加情報
            silent: ログ出力を抑制するか（API用）

        Returns:
            生成されたテキスト
        """
        start = time.perf_counter()

        # システムプロンプトの構築
        system_content = [
            {"type": "text", "text": "あなたは日本語を話すAIアシスタントです。"}
        ]

        # ナレッジベースからの情報を追加
        if knowledge_context:
            system_content.append(
                {
                    "type": "text",
                    "text": f"以下の情報を参考にして回答してください：\n{knowledge_context}",
                }
            )

        # 会話履歴を追加（全モデル対応）
        if use_history:
            memory = self._load_history()
            if memory:
                system_content.append(
                    {
                        "type": "text",
                        "text": f"以下は過去の会話情報です。必要だと思った場合は活用してください：\n{memory}",
                    }
                )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        response_text = ""

        if self.model_type == "gguf":
            response_text = self._generate_gguf(
                messages, stream, max_tokens, temperature, silent
            )
        else:
            response_text = self._generate_pytorch(messages, max_tokens, silent)

        end = time.perf_counter() - start
        if not silent:
            logger.info(f"処理時間: {end:.2f}秒")

        # 会話履歴を保存（全モデル対応）
        if use_history:
            self._save_memory(prompt)

        return response_text

    def _generate_gguf(
        self,
        messages: List[Dict[str, Any]],
        stream: bool,
        max_tokens: int,
        temperature: float,
        silent: bool = False,
    ) -> str:
        """GGUF モデルでテキスト生成"""
        resp = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
        )

        if stream:
            partial_message = ""
            for msg in resp:
                message = msg["choices"][0]["delta"]
                if "content" in message:
                    content = message["content"]
                    if not silent:
                        print(content, end="", flush=True)
                    partial_message += content
            if not silent:
                print()
            return partial_message.strip()
        else:
            content = resp["choices"][0]["message"]["content"]
            return content.strip()

    def _generate_pytorch(
        self, messages: List[Dict[str, Any]], max_tokens: int, silent: bool = False
    ) -> str:
        """PyTorch モデルでテキスト生成"""
        output = self.model(text_inputs=messages, max_new_tokens=max_tokens)
        response_text = output[0]["generated_text"][-1]["content"].strip()
        # PyTorchモデルはストリーミング非対応なので常に出力（silentでない場合）
        if not silent:
            print(response_text)
        return response_text


def main():
    parser = argparse.ArgumentParser(description="Gemma3統合AIアシスタント")
    parser.add_argument(
        "prompt", nargs="?", type=str, help="AIに問い合わせるプロンプト"
    )
    parser.add_argument(
        "--model-type",
        choices=["gguf", "pytorch"],
        default="gguf",
        help="使用するモデルタイプ (デフォルト: gguf)",
    )
    parser.add_argument(
        "--model-size",
        choices=["1b", "4b"],
        default="1b",
        help="PyTorchモデルのサイズ (デフォルト: 1b)",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="ストリーミング出力を無効化"
    )
    parser.add_argument("--no-history", action="store_true", help="会話履歴を無効化")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大トークン数")
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="生成の温度パラメータ"
    )

    args = parser.parse_args()

    if not args.prompt:
        args.prompt = "こんにちは～あなたが駆動しているモデルを教えてください"

    # モデル初期化
    model = Gemma3Model(model_type=args.model_type, model_size=args.model_size)

    # テキスト生成
    response = model.generate(
        prompt=args.prompt,
        use_history=not args.no_history,
        stream=not args.no_stream,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
