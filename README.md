Raspberry Pi 5で[gemma-3-1b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf)モデルを動かします

# 動作

こんなかんじ

![20250615_hello.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/635079/41c0a813-a96a-4591-a2be-278eb3ad85ec.gif)

![20250615_qiita_zenn.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/635079/4612a11e-aa77-4d85-b0a2-5e060448d26d.gif)

![20250615_tokyo.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/635079/053d6586-17c6-4c5d-951a-ebca0df15c06.gif)

# Setup & Run

1. [setup raspberry pi 5](https://qiita.com/nijigen_plot/items/5f5299af6aebc54b42d3#raspberry-pi-5-%E3%82%BB%E3%83%83%E3%83%88%E3%82%A2%E3%83%83%E3%83%97)
1. install [uv](https://docs.astral.sh/uv/getting-started/installation/)
1. register [Hugging Face](https://huggingface.co/)
1. register SSH Key on Hugging Face
1. install [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation)
1. clone this repository`$ git clone -b gemma-3-1b-gguf-with-raspberrypi-5 git@github.com:nijigen-plot/personal-knowledge-base.git`
1. run `uv sync`
1. `$ cd personal-knowledge-base`
1. run `touch history.txt`
1. install [git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation)
1. clone hugging face repository `$ git clone git@hf.co:google/gemma-3-1b-it-qat-q4_0-gguf`
1. run script `$ uv run python gemma3-1b-q4-gguf.py prompt`
