import argparse
import time

from llama_cpp import Llama


def main():
    parser = argparse.ArgumentParser(description="第一引数をプロンプトとしてうけとりAIに問い合わせます")
    parser.add_argument("prompt", type=str, help="AIに問い合わせるプロンプト")
    args = parser.parse_args()
    prompt = args.prompt
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "あなたは日本語を話すAIアシスタントです。"}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    llm = Llama(
        model_path="./gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf",
        verbose=False
        )
    start = time.perf_counter()
    resp = llm.create_chat_completion(
        messages=messages
    )
    print(resp["choices"][0]["message"]["content"])
    end = time.perf_counter() - start
    print(f"処理時間: {end:.2f}秒")

# loading shardsはpyファイル（プロセス）実行のたびに入るので、いざ使う時は発生しないように対処する
if __name__ == "__main__":
    main()
