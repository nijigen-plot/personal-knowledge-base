import argparse
import gc
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
from llama_cpp import Llama
from transformers import pipeline

# メモリ解放を積極的に行う設定
os.environ["MALLOC_TRIM_THRESHOLD_"] = "-1"

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
        rag_context: Optional[str] = None,
        use_history: bool = True,
        stream: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.3,
        silent: bool = False,
    ) -> str:
        """
        テキスト生成

        Args:
            prompt: ユーザーからの質問
            use_history: 会話履歴を使用するか
            stream: ストリーミング出力するか
            max_tokens: LLM回答の最大トークン数
            temperature: 生成の温度パラメータ
            silent: ログ出力を抑制するか（API用）

        Returns:
            生成されたテキスト
        """
        start = time.perf_counter()

        # システムプロンプトの構築
        system_content = [
            {
                "type": "text",
                "text": f"""
                あなたは会話アシスタントです。
                ユーザーの質問に対して、適切な回答を生成してください。
                また、あなたはRAG（Retrieval-Augmented Generation）と繋がっています。
                ユーザーがRAGコンテキスト機能を利用した場合は、この後に続いてRAGコンテキストが提供されるので、
                それを活用して回答を生成してください。
            """,
            }
        ]

        # RAGコンテキストが提供されている場合は追加
        if rag_context:
            system_content.append(
                {
                    "type": "text",
                    "text": f"RAGコンテキストが提供されました:\n{rag_context}",
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

    def clear_memory(self):
        """メモリを明示的に解放"""
        print("Gemma3モデルのメモリを解放中...")

        if hasattr(self, "model") and self.model is not None:
            del self.model

        # Python ガベージコレクション
        gc.collect()
        print("Gemma3モデルのメモリ解放完了")

    def _extract_timestamp_with_regex(self, text: str) -> Optional[Dict[str, str]]:
        """正規表現を使用してタイムスタンプを抽出"""
        now = datetime.now(timezone(timedelta(hours=9)))

        # 相対的な時間表現のマッピング
        time_patterns = {
            r"今日|きょう": (0, "days"),
            r"昨日|きのう": (1, "days"),
            r"一昨日|おととい": (2, "days"),
            r"最近|近頃": (7, "days"),
            r"先週|前週": (7, "days"),
            r"今週|こんしゅう": (0, "days"),
            r"先月|前月": (30, "days"),
            r"今月|こんげつ": (0, "days"),
            r"去年|昨年": (365, "days"),
            r"今年|こんねん": (0, "days"),
            r"(\d+)日前": (None, "days"),
            r"(\d+)週間前": (None, "weeks"),
            r"(\d+)ヶ月前|(\d+)か月前": (None, "months"),
            r"(\d+)年前": (None, "years"),
        }

        for pattern, (days_offset, unit) in time_patterns.items():
            match = re.search(pattern, text)
            if match:
                if days_offset is None:
                    # 数字を抽出
                    number = int(match.group(1))
                    if unit == "days":
                        start_date = now - timedelta(days=number)
                    elif unit == "weeks":
                        start_date = now - timedelta(weeks=number)
                    elif unit == "months":
                        start_date = now - timedelta(days=number * 30)
                    elif unit == "years":
                        start_date = now - timedelta(days=number * 365)
                else:
                    start_date = now - timedelta(days=days_offset)

                # 期間の終了時刻を設定
                if "今日" in pattern or "きょう" in pattern:
                    end_date = now
                elif days_offset == 0:  # 今週、今月、今年
                    end_date = now
                else:
                    end_date = now

                return {
                    "gte": start_date.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    "lte": end_date.strftime("%Y-%m-%dT%H:%M:%S.%f"),
                }

        return None

    def extract_tag(self, text: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        文章からタグとタイムスタンプ期間を抽出

        Args:
            text: 解析対象の文章
            max_retries: JSON解析失敗時のリトライ回数

        Returns:
            {
                "tag": ["lifestyle", "music", "technology"] のいずれか1個か無し。
                "timestamp": {
                    "gte": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
                    "lte": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
                },
                "content": "元の文章がここにはいる"
            }
        """
        # まず正規表現でタイムスタンプを抽出
        timestamp_result = self._extract_timestamp_with_regex(text)

        prompt = f"""以下の文章を分析して、該当するタグを抽出してください。

文章: {text}

抽出ルール:
1. タグは lifestyle, music, technology のいずれか１つで、何にも該当しない場合はtagのkey valueを含めないでください。
2. 必ず以下のJSON形式で回答してください：

{{
  "tag": ["ここに該当するタグを1つだけ入れてください"],
}}

または、該当するタグがない場合：

{{}}

他の形式での回答は禁止されています。JSONのみ出力してください。"""

        for attempt in range(max_retries):
            try:
                response = self.generate(
                    prompt=prompt,
                    use_history=False,
                    stream=False,
                    max_tokens=512,
                    temperature=0.1,
                    silent=False,
                )
                print(f"抽出結果 (試行 {attempt + 1}/{max_retries}): {response}")

                # JSONブロックを抽出
                json_str = self._extract_json_from_response(response)
                if json_str:
                    result = json.loads(json_str)

                    # 結果を組み立て
                    final_result = {"content": text}

                    # タグが含まれていれば追加
                    if "tag" in result:
                        final_result["tag"] = result["tag"]

                    # 正規表現で抽出したタイムスタンプを追加
                    if timestamp_result:
                        final_result["timestamp"] = timestamp_result

                    # バリデーション
                    if self._validate_extraction_result(final_result):
                        print(final_result)
                        return final_result

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    f"JSON解析エラー (試行 {attempt + 1}/{max_retries}): {e}"
                )
                continue

        # 3回リトライした場合はcontentと正規表現抽出結果を返す
        final_result = {"content": text}
        if timestamp_result:
            final_result["timestamp"] = timestamp_result
        return final_result

    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """レスポンスからJSONを抽出"""
        # JSONブロックを抽出
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            return match.group(1)

        # 直接JSONを探す
        json_pattern = r"(\{.*?\})"
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            return match.group(1)

        return None

    def _validate_extraction_result(self, result: Dict[str, Any]) -> bool:
        """抽出結果のバリデーション"""
        if not isinstance(result, dict):
            return False

        # contentの検証（必須）
        if "content" not in result or not isinstance(result["content"], str):
            return False

        # tagの検証（オプション）
        if "tag" in result:
            if not isinstance(result["tag"], list):
                return False
            # 1つのタグのみ許容
            if len(result["tag"]) != 1:
                return False
            valid_tags = {"lifestyle", "music", "technology"}
            if result["tag"][0] not in valid_tags:
                return False

        # timestampの検証（オプション）
        if "timestamp" in result:
            timestamp = result["timestamp"]
            if timestamp is not None:
                if not isinstance(timestamp, dict):
                    return False

                # gteとlteが存在する場合は文字列または null であることを確認
                if "gte" in timestamp and timestamp["gte"] is not None:
                    if not isinstance(timestamp["gte"], str):
                        return False

                if "lte" in timestamp and timestamp["lte"] is not None:
                    if not isinstance(timestamp["lte"], str):
                        return False

        return True

    def __del__(self):
        """デストラクタでメモリ解放"""
        try:
            self.clear_memory()
        except:
            pass  # エラーが発生してもデストラクタでは例外を投げない


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
    parser.add_argument(
        "--extract-tag",
        action="store_true",
        help="文章からOpenSearch Filter用のタグとタイムスタンプを抽出",
    )

    args = parser.parse_args()

    if not args.prompt:
        args.prompt = "こんにちは～あなたが駆動しているモデルを教えてください"

    # モデル初期化
    model = Gemma3Model(model_type=args.model_type, model_size=args.model_size)

    if args.extract_tag:
        model.extract_tag(
            text=args.prompt,
        )
    else:
        # テキスト生成
        model.generate(
            prompt=args.prompt,
            use_history=not args.no_history,
            stream=not args.no_stream,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
