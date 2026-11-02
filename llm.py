import argparse
import gc
import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import pipeline

from log_config import get_logger

# 環境変数読み込み
load_dotenv(".env")

logger = get_logger(__name__)

# メモリ解放を積極的に行う設定
os.environ["MALLOC_TRIM_THRESHOLD_"] = "-1"

HISTORY_FILE = "history.txt"

# OpenAI API の設定（.env で上書き可能）
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
# gpt-6-luna / gpt-6-sol は none / low / medium(既定) / high / xhigh / max に対応。
# none にすると推論をせず gpt-4o と同じ感覚で使え、temperature も指定できる。
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none")

JST = timezone(timedelta(hours=9))

# 検索期間を判断させるときの出力形式（Structured Outputs）。
# 時期の指定がない質問では timestamp が null になる。
SEARCH_PERIOD_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "search_period",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "gte": {"type": "string", "description": "開始日 (YYYY-MM-DD)"},
                                "lte": {"type": "string", "description": "終了日 (YYYY-MM-DD)"},
                            },
                            "required": ["gte", "lte"],
                            "additionalProperties": False,
                        },
                        {"type": "null"},
                    ]
                }
            },
            "required": ["timestamp"],
            "additionalProperties": False,
        },
    },
}


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
            raise ValueError("model_typeは'openaiapi' または'openai-20b'を指定してください")

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
            logger.info(f"{HISTORY_FILE} が存在しないため、過去の会話履歴は考慮されません。")
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

        # OpenAI API用とPyTorch用でmessage形式を分ける
        if self.model_type == "openai-api":
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
        else:
            # PyTorch用はstring形式
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

        # モデル生成関数のマッピング
        generators = {
            "openai-api": lambda: self._generate_openai(messages, stream, max_tokens, temperature, silent),
            "openai-20b": lambda: self._generate_pytorch(messages, max_tokens, silent),
        }

        generator = generators.get(self.model_type)
        if not generator:
            raise ValueError(f"未対応のmodel_type: {self.model_type}. " f"対応可能: {list(generators.keys())}")

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

        # 新しいモデルは max_tokens を受け付けないので max_completion_tokens を使う。
        # 推論をする設定（none 以外）では temperature を送るとエラーになるので外す。
        request_params: Dict[str, Any] = {
            "model": OPENAI_MODEL,
            "messages": openai_messages,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "stream": stream,
        }
        if OPENAI_REASONING_EFFORT == "none":
            request_params["temperature"] = temperature

        try:
            response = self.model.chat.completions.create(**request_params)

            if stream:
                partial_message = ""
                for chunk in response:
                    # 最後に choices が空のチャンクが来ることがあるので飛ばす
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        if not silent:
                            print(content, end="", flush=True)
                        partial_message += content
                if not silent:
                    print()
                return partial_message.strip()
            else:
                content = response.choices[0].message.content or ""
                if not content:
                    logger.warning(f"空の応答でした (finish_reason={response.choices[0].finish_reason})")
                if not silent:
                    print(content)
                return content.strip()
        except Exception as e:
            logger.error(f"OpenAI API呼び出しエラー: {e}")
            return f"エラーが発生しました: {e}"

    def _generate_pytorch(self, messages: List[Dict[str, Any]], max_tokens: int, silent: bool = False) -> str:
        """PyTorch モデルでテキスト生成"""
        # messagesをchat templateで処理
        output = self.model(messages, max_new_tokens=max_tokens)
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

    def extract_tag(self, text: str, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        質問文から、記録を検索するときの期間を LLM に判断させる

        Args:
            text: 質問文
            now: 現在時刻（テスト用。省略時は今の日本時間）

        Returns:
            {
                "content": "元の文章",
                "timestamp": {  # 時期の指定がない、または判断に失敗したときはキー自体なし
                    "gte": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",  # UTC
                    "lte": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",  # UTC
                },
            }
        """
        result: Dict[str, Any] = {"content": text}
        timestamp = self._extract_timestamp_with_llm(text, now or datetime.now(JST))
        if timestamp:
            result["timestamp"] = timestamp
        logger.info(f"抽出結果: {result}")
        return result

    def _extract_timestamp_with_llm(self, text: str, now: datetime) -> Optional[Dict[str, str]]:
        """質問が指している期間を LLM に判断させる。判断できないときは None（期間で絞り込まない）"""
        if self.model_type != "openai-api":
            # Structured Outputs が使えるのは OpenAI API のときだけ
            return None

        weekday = "月火水木金土日"[now.weekday()]
        prompt = f"""現在の日時は {now:%Y-%m-%d %H:%M}（{weekday}曜日、日本時間）です。

次の質問は、ある人の日記や記録を検索するために使われます。
質問が特定の時期について尋ねている場合は、検索すべき期間を開始日と終了日（YYYY-MM-DD）で答えてください。
「最近」「この前」のような曖昧な表現は、記録を探すうえで取りこぼしが少ない自然な範囲を判断してください。
未来の日付は含めないでください。時期についての指定がない質問では timestamp を null にしてください。

質問: {text}"""

        params: Dict[str, Any] = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "response_format": SEARCH_PERIOD_FORMAT,
        }
        if OPENAI_REASONING_EFFORT == "none":
            params["temperature"] = 0
            params["max_completion_tokens"] = 200
        else:
            # 推論に使うトークンも上限に含まれるので多めに取る
            params["max_completion_tokens"] = 4000

        try:
            response = self.model.chat.completions.create(**params)
            data = json.loads(response.choices[0].message.content or "")
            return self._to_timestamp_filter(data.get("timestamp"), now)
        except Exception as e:
            logger.warning(f"期間の判断に失敗したため、期間で絞り込まずに検索します: {e}")
            return None

    @staticmethod
    def _to_timestamp_filter(period: Optional[Dict[str, str]], now: datetime) -> Optional[Dict[str, str]]:
        """LLM が答えた日付（日本時間）を、OpenSearch の range 条件（UTC）に変換する"""
        if not period:
            return None

        start_day = datetime.strptime(period["gte"], "%Y-%m-%d").date()
        end_day = datetime.strptime(period["lte"], "%Y-%m-%d").date()
        if start_day > end_day:
            start_day, end_day = end_day, start_day

        start = datetime.combine(start_day, datetime.min.time(), tzinfo=JST)
        end = datetime.combine(end_day, datetime.max.time(), tzinfo=JST)
        if start > now:
            return None
        end = min(end, now)

        # 記録は UTC で保存されているので UTC に直す
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        return {
            "gte": start.astimezone(timezone.utc).strftime(fmt),
            "lte": end.astimezone(timezone.utc).strftime(fmt),
        }

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
    parser.add_argument("--no-stream", action="store_true", help="ストリーミング出力を無効化")
    parser.add_argument("--no-history", action="store_true", help="会話履歴を無効化")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.3, help="生成の温度パラメータ")
    parser.add_argument(
        "--extract-tag",
        action="store_true",
        help="質問文から検索に使う期間を LLM に判断させて表示",
    )

    args = parser.parse_args()

    # モデル初期化
    model = LargeLanguageModel(model_type=args.model_type)

    if args.extract_tag:
        result = model.extract_tag(text=args.prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))
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
