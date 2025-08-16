import os
import time
from typing import Generator

import requests
import streamlit as st
from dotenv import load_dotenv

from log_config import get_logger

load_dotenv(".env")

logger = get_logger(__name__)


def post_conversation(prompt: str) -> Generator[str, None, None]:
    # 環境変数からFastAPIのホストとポートを取得
    app_host = os.getenv("APP_HOST", "localhost")
    app_port = os.getenv("APP_PORT", "8050")
    url = f"http://{app_host}:{app_port}/api/v1/conversation"
    try:
        response = requests.post(
            url,
            json={"question": prompt},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        response_str = response.json().get(
            "answer",
            "申し訳ありませんが、内部エラーにより回答できませんでした。管理者にお問い合わせください。",
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"APIリクエストエラー: {e}")
        response_str = "申し訳ありませんが、サーバーとの通信でエラーが発生しました。管理者にお問い合わせください。"

    for char in response_str:
        yield char
        time.sleep(0.02)


st.title("[Quarkgabber](https://quark-hardcore.com/)のナレッジベース")
st.markdown(
    """
    なんでも質問OK！2012年からのTwitter情報や定期的に記録している情報をもとに、あなたの質問に答えます。\n
    [GitHub Repository](https://github.com/nijigen-plot/personal-knowledge-base)
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("趣味・出来事なんでも質問OK！"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("回答を生成中..."):
            response = st.write_stream(post_conversation(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
