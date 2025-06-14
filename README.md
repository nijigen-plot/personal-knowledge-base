Run the [Gemma3](https://huggingface.co/google/gemma-3-4b-it) Model for local.

# Setup

1. install uv
2. run `uv sync`
3. get Hugging Face Write access token https://huggingface.co/docs/hub/security-tokens
4. install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/main/guides/cli)
5. run `huggingface-cli login`
6. paste Hugging Face access token
7. install [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation)
8. clone repository `git clone https://huggingface.co/google/gemma-3-4b-it` or `git clone git@hf.co:google/gemma-3-4b-it`(Need Write Permission Access Token)


# CPUメモ

1. Raspberry Pi 5はBCM2712 SoCで、CPUはArm Cortex-A76 (https://eetimes.itmedia.co.jp/ee/articles/2309/28/news177.html)
2. Cortex-A76はARM v8.2-Aアーキテクチャで半精度16bitの計算に対応している (https://en.wikipedia.org/wiki/AArch64#ARMv8.2-A)
3. Intel x86 CPUは4世代Xeonから半精度16bit計算対応で普通の人はまず持ってない (https://zenn.dev/mod_poppo/articles/half-precision-floating-point)
    a. 変換はIvy Bridgeから対応(3世代)

# Benchmarkメモ

4bは簡素で1bは文章量が多いので生成秒数が逆転している
リアルタイムで描画するようにしたらまた評価かわるかも
1b-4bit unslothのモデルはめっちゃ遅い。370秒掛かった。多分自分の設定が悪い
google gemma 3 1b ggufはめっちゃ速い 3~5秒くらい

## i9-9980XE

- 4b -> 10秒程
- 1b -> 30秒程

## Cortex-A76

- 4b -> 40~50秒
- 1b -> 40~50秒
