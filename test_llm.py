"""llm.py の OpenAI 呼び出しパラメータのテスト（API は呼ばずにモックで確認する）"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import llm
from llm import LargeLanguageModel

MESSAGES = [
    {"role": "system", "content": [{"type": "text", "text": "システム"}]},
    {"role": "user", "content": [{"type": "text", "text": "こんにちは"}]},
]


@pytest.fixture
def openai_llm():
    """API キーなしで使えるように、__init__ を通さずにインスタンスを作る"""
    model = LargeLanguageModel.__new__(LargeLanguageModel)
    model.model_type = "openai-api"
    model.model = Mock()
    return model


def make_response(content, finish_reason="stop"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


def make_chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class TestGenerateOpenAI:
    def test_uses_new_parameters(self, openai_llm, monkeypatch):
        monkeypatch.setattr(llm, "OPENAI_MODEL", "gpt-6-sol")
        monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "none")
        openai_llm.model.chat.completions.create.return_value = make_response("回答")

        result = openai_llm._generate_openai(MESSAGES, stream=False, max_tokens=512, temperature=0.5, silent=True)

        assert result == "回答"
        params = openai_llm.model.chat.completions.create.call_args.kwargs
        assert params["model"] == "gpt-6-sol"
        assert params["max_completion_tokens"] == 512
        assert "max_tokens" not in params
        assert params["reasoning_effort"] == "none"
        assert params["temperature"] == 0.5
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][1]["content"] == "こんにちは\n"

    def test_omits_temperature_when_reasoning(self, openai_llm, monkeypatch):
        monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "low")
        openai_llm.model.chat.completions.create.return_value = make_response("回答")

        openai_llm._generate_openai(MESSAGES, stream=False, max_tokens=512, temperature=0.5, silent=True)

        params = openai_llm.model.chat.completions.create.call_args.kwargs
        assert params["reasoning_effort"] == "low"
        assert "temperature" not in params

    def test_stream_skips_empty_chunks(self, openai_llm):
        empty_chunk = SimpleNamespace(choices=[])
        openai_llm.model.chat.completions.create.return_value = iter(
            [make_chunk("こん"), make_chunk(None), make_chunk("にちは"), empty_chunk]
        )

        result = openai_llm._generate_openai(MESSAGES, stream=True, max_tokens=512, temperature=0.5, silent=True)

        assert result == "こんにちは"

    def test_empty_response_returns_empty_string(self, openai_llm):
        openai_llm.model.chat.completions.create.return_value = make_response(None, finish_reason="length")

        result = openai_llm._generate_openai(MESSAGES, stream=False, max_tokens=512, temperature=0.5, silent=True)

        assert result == ""
