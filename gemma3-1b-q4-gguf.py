import argparse
import logging
import os
import time

from llama_cpp import Llama

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
HISTORY_FILE = "history.txt"

def save_memory(message: str):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    else:
        pass

def main(stream: bool = True, max_tokens: int = 512, n_ctx: int = 32768):
    llm = Llama(
        model_path="./gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf",
        verbose=False,
        n_ctx=n_ctx,
    )
    parser = argparse.ArgumentParser(
        description="第一引数をプロンプトとしてうけとりAIに問い合わせます"
    )
    parser.add_argument("prompt", type=str, help="AIに問い合わせるプロンプト")
    args = parser.parse_args()
    prompt = args.prompt
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            # https://gemma-llm.readthedocs.io/en/latest/colab_tokenizer.html?utm_source=chatgpt.com
            # 単語でトークンカウントしているので、文字数でn_ctxの数だけ読む分にはまずオーバーしないはず
            memory_file = f.read()
        memory = memory_file[-n_ctx:]
        logger.info(f"{HISTORY_FILE} から過去の会話履歴を読み込みました。")
    else:
        logger.info(f"{HISTORY_FILE} が存在しないため、過去の会話履歴は考考されません。必要な場合は{HISTORY_FILE} を作成してください。")
        memory = ""
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": f"あなたは日本語を話すAIアシスタントです。"}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": memory},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    start = time.perf_counter()
    resp = llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, stream=stream, temperature=0.3
    )
    if stream:
        partial_message = ""
        for msg in resp:
            message = msg["choices"][0]["delta"]
            if "content" in message:
                content = message["content"]
                print(content, end="", flush=True)
                partial_message += content
        print()
    else:
        content = resp["choices"][0]["message"]["content"]
        print(content)
    end = time.perf_counter() - start
    logger.info(f"処理時間: {end:.2f}秒")
    save_memory(prompt)


if __name__ == "__main__":
    main()
