import time

import torch
from llama_cpp import Llama

# loading shardsはpyファイル（プロセス）実行のたびに入るので、いざ使う時は発生しないように対処する
if __name__ == "__main__":
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = Llama(
        model_path="./gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf",
        )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "あなたは日本語を話すAIアシスタントです。"}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "こんにちは～あなたが駆動しているモデルをおしえてください"}
            ]
        }
    ]
    resp = llm.create_chat_completion(
        messages=messages
    )
    print(resp["choices"][0]["message"]["content"])
    end = time.perf_counter() - start
    print(f"処理時間: {end:.2f}秒")
