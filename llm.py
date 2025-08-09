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
from dotenv import load_dotenv
from llama_cpp import Llama
from openai import OpenAI
from transformers import pipeline

from log_config import get_logger

# 環境変数読み込み
load_dotenv()

logger = get_logger(__name__)

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


class LargeLanguageModel:
    def __init__(self, model_type: str = "openai-api"):
        """
        大規模言語モデルの統合クラス

        Args:
            model_type: "openai-api" または "opanai-20b"
        """
        self.model_type = model_type
        self.model = None

        if model_type == "openai-api":
            self._load_openai_api_client()
        elif model_type == "openai-20b":
            self._load_openai_20b_client()
        else:
            raise ValueError(
                "model_typeは'openaiapi' または'openai-20b'を指定してください"
            )

    def _load_openai_20b_client(self):
        """PyTorchモデルを読み込み"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "./gpt-oss-20b"

        logger.info(f"PyTorchモデルを読み込み中: {model_path} (device: {device})")

        self.model = pipeline(
            task="text-generation",
            model=model_path,
            device=device,
            torch_dtype="auto",
        )
        logger.info(f"PyTorchモデルの読み込み完了: ")

    def _load_openai_api_client(self):
        """OpenAI APIクライアントを初期化"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEYが環境変数に設定されていません")

        self.model = OpenAI(api_key=api_key)
        logger.info("OpenAI APIクライアントを初期化しました")

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
        temperature: float = 0.5,
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
                あなたはQuarkgabberです。データエンジニアでテクノロジーに深い関心を持っています。
                また、音楽も作ります。

                以下の記録に含まれる情報は、全てあなた自身の実際の体験、思考、行動、発言です。
                この情報を基に、あなた自身として自然に答えてください。

                重要なガイドライン：
                - 一人称（私、僕、自分など）で話す
                - 記録された情報を自分の体験として語る
                - 日時や時期の情報があれば積極的に言及する
                - 具体的な体験や感情を含めて話す
                - 質問に対して関連する記録があれば必ず活用する
                - 誕生日、好きなもの、経験したことなど、個人的な情報も積極的に共有する
                - 記録にない情報について聞かれた場合は、正直に「記録にない」旨を伝える

                あなたの性格：
                - 技術的な話題に興味深い
                - オープンで親しみやすい
                - 詳しく具体的に説明する傾向がある
                - 自分の経験を積極的に共有する
            """,
            }
        ]

        # RAGコンテキストが提供されている場合は追加
        if rag_context:
            logger.info(f"RAGコンテキストが提供されました。{rag_context}")
            system_content.append(
                {
                    "type": "text",
                    "text": f"""
                あなたの記録とログ：
                {rag_context}

                上記の記録は、あなた自身の実際の体験、思考、行動の記録です。
                これらの情報を使って、以下の点を意識して答えてください：

                1. 日時や時期の情報（timestamp）が含まれていれば、「〜の時に」「〜年頃に」など具体的に言及する
                2. 記録された体験を自分の実際の体験として語る（「私は〜した」「その時僕は〜と思った」など）
                3. 感情や考えが記録されていれば、それも含めて話す
                4. 複数の関連する記録がある場合は、時系列や関連性を考慮して統合的に答える
                5. 記録の内容が質問に直接関連する場合は、積極的に詳細を共有する

                記録された情報は全て事実として扱い、推測や創作は避けてください。
                """,
                }
            )

        # 会話履歴を追加（全モデル対応）
        if use_history:
            memory = self._load_history()
            if memory:
                system_content.append(
                    {
                        "type": "text",
                        "text": f"""
                以下は過去の会話履歴です。これもあなた自身の記録の一部です：
                {memory}

                この会話履歴も参考にして、一貫性のある回答を心がけてください。
                過去の会話で言及した内容や、継続的な話題がある場合は、それを踏まえて回答してください。
                """,
                    }
                )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        # モデル生成関数のマッピング
        generators = {
            "openai-api": lambda: self._generate_openai(
                messages, stream, max_tokens, temperature, silent
            ),
            "openai-20b": lambda: self._generate_pytorch(messages, max_tokens, silent),
        }

        generator = generators.get(self.model_type)
        if not generator:
            raise ValueError(
                f"未対応のmodel_type: {self.model_type}. "
                f"対応可能: {list(generators.keys())}"
            )

        response_text = generator()

        end = time.perf_counter() - start
        if not silent:
            logger.info(f"処理時間: {end:.2f}秒")

        # 会話履歴を保存（全モデル対応）
        if use_history:
            self._save_memory(prompt)

        return response_text

    def _generate_openai(
        self,
        messages: List[Dict[str, Any]],
        stream: bool,
        max_tokens: int,
        temperature: float,
        silent: bool = False,
    ) -> str:
        """OpenAI APIでテキスト生成"""
        # OpenAI APIのメッセージフォーマットに変換
        openai_messages = []
        for message in messages:
            if message["role"] == "system":
                # システムメッセージをテキストに変換
                system_text = ""
                for content in message["content"]:
                    system_text += content["text"] + "\n"
                openai_messages.append({"role": "system", "content": system_text})
            else:
                # ユーザーメッセージをテキストに変換
                user_text = ""
                for content in message["content"]:
                    user_text += content["text"] + "\n"
                openai_messages.append({"role": "user", "content": user_text})

        try:
            response = self.model.chat.completions.create(
                model="gpt-4o",
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )

            if stream:
                partial_message = ""
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        if not silent:
                            print(content, end="", flush=True)
                        partial_message += content
                if not silent:
                    print(content)
                return partial_message.strip()
            else:
                content = response.choices[0].message.content
                if not silent:
                    print(content)
                return content.strip()
        except Exception as e:
            logger.error(f"OpenAI API呼び出しエラー: {e}")
            return f"エラーが発生しました: {e}"

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
        logger.info("言語モデルのメモリを解放中...")

        if hasattr(self, "model") and self.model is not None:
            del self.model

        # Python ガベージコレクション
        gc.collect()
        logger.info("言語モデルのメモリ解放完了")

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
                logger.info(f"抽出結果 (試行 {attempt + 1}/{max_retries}): {response}")

                # JSONブロックを抽出
                json_str = self._extract_json_from_response(response)
                if json_str:
                    result = json.loads(json_str)

                    # 結果を組み立て
                    final_result = {"content": text}

                    # タグが含まれていれば追加→タグをつけるほどの量が今の所ないので、tagは無し
                    # if "tag" in result:
                    # final_result["tag"] = result["tag"]

                    # 正規表現で抽出したタイムスタンプを追加
                    if timestamp_result:
                        final_result["timestamp"] = timestamp_result

                    # バリデーション
                    if self._validate_extraction_result(final_result):
                        logger.info(final_result)
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
            # tagが配列の場合は最初の要素を使用
            if isinstance(result["tag"], list):
                if len(result["tag"]) > 0:
                    result["tag"] = result["tag"][0]
                else:
                    result["tag"] = None

            # 単一の値として検証
            if result["tag"] is not None:
                valid_tags = {"lifestyle", "music", "technology"}
                if result["tag"] not in valid_tags:
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
    parser = argparse.ArgumentParser(description="大規模言語モデル統合AIアシスタント")
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="AIに問い合わせるプロンプト",
        default="こんにちは～あなたが駆動しているモデルを教えてください",
    )
    parser.add_argument(
        "--model-type",
        choices=["openai-api", "openai-20b"],
        default="openai-api",
        help="使用するモデルタイプ (デフォルト: openai-api)",
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

    # モデル初期化
    model = LargeLanguageModel(model_type=args.model_type)

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
