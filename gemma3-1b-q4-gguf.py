import argparse
import logging
import time

from llama_cpp import Llama

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def main(stream: bool = True, max_tokens: int = 512):
    parser = argparse.ArgumentParser(
        description="第一引数をプロンプトとしてうけとりAIに問い合わせます"
    )
    parser.add_argument("prompt", type=str, help="AIに問い合わせるプロンプト")
    args = parser.parse_args()
    prompt = args.prompt
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": f"あなたは日本語を話すAIアシスタントです。"}
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    llm = Llama(
        model_path="./gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf",
        verbose=False,
        n_ctx=1024,
    )
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
                print(content, end="")
                partial_message += content
        logger.info(f"最終的なメッセージ: {partial_message}")
    else:
        print(resp["choices"][0]["message"]["content"])
        logger.info(f"最終的なメッセージ: {resp['choices'][0]['message']['content']}")
    end = time.perf_counter() - start
    logger.info(f"処理時間: {end:.2f}秒")


# loading shardsはpyファイル（プロセス）実行のたびに入るので、いざ使う時は発生しないように対処する
if __name__ == "__main__":
    main()
