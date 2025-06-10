import time

import torch
from transformers import pipeline


def main(pipe):
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

    output = pipe(text_inputs=messages, max_new_tokens=200)
    print(output[0]["generated_text"][-1]["content"])

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hugging Faceにログインすれば自動的にデータがダウンロードされる
    pipe = pipeline(
        task="text-generation",
        model="google/gemma-3-4b-it",
        device=device,
        # CPUの場合、Raspberry Pi 5(Cortex-A86 ARMv8.2-A)はfloat16対応。メイン機(i9-9980XE)はbfloat16,float16どちらも対応してない
        # 上は変換の話じゃなくてネイティブ計算の話で、変換はどちらも対応してる
        torch_dtype="auto"
    )
    start = time.perf_counter()
    main(pipe)
    end = time.perf_counter() - start
    print(f"処理時間: {end:.2f}秒")
