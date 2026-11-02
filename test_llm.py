"""llm.py の OpenAI 呼び出しパラメータのテスト（API は呼ばずにモックで確認する）"""

import json
from datetime import datetime
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
        monkeypatch.setattr(llm, "OPENAI_MODEL", "gpt-6-luna")
        monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "none")
        openai_llm.model.chat.completions.create.return_value = make_response("回答")

        result = openai_llm._generate_openai(MESSAGES, stream=False, max_tokens=512, temperature=0.5, silent=True)

        assert result == "回答"
        params = openai_llm.model.chat.completions.create.call_args.kwargs
        assert params["model"] == "gpt-6-luna"
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


# 2026-11-02 12:00 日本時間（= 03:00 UTC）
NOW = datetime(2026, 11, 2, 12, 0, tzinfo=llm.JST)


def set_llm_answer(openai_llm, timestamp):
    content = json.dumps({"timestamp": timestamp})
    openai_llm.model.chat.completions.create.return_value = make_response(content)


class TestExtractTag:
    def test_period_is_converted_to_utc_range(self, openai_llm, monkeypatch):
        monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "none")
        set_llm_answer(openai_llm, {"gte": "2026-10-01", "lte": "2026-10-31"})

        result = openai_llm.extract_tag("先月の音楽制作の話を教えて", now=NOW)

        assert result == {
            "content": "先月の音楽制作の話を教えて",
            "timestamp": {
                "gte": "2026-09-30T15:00:00.000000",
                "lte": "2026-10-31T14:59:59.999999",
            },
        }
        params = openai_llm.model.chat.completions.create.call_args.kwargs
        assert params["response_format"]["type"] == "json_schema"
        assert params["response_format"]["json_schema"]["strict"] is True
        assert params["temperature"] == 0
        assert "2026-11-02 12:00" in params["messages"][0]["content"]
        assert "月曜日" in params["messages"][0]["content"]
        assert "先月の音楽制作の話を教えて" in params["messages"][0]["content"]

    def test_no_period(self, openai_llm):
        set_llm_answer(openai_llm, None)

        result = openai_llm.extract_tag("好きな食べ物は？", now=NOW)

        assert result == {"content": "好きな食べ物は？"}

    def test_today_is_not_empty_range(self, openai_llm):
        set_llm_answer(openai_llm, {"gte": "2026-11-02", "lte": "2026-11-02"})

        result = openai_llm.extract_tag("今日なにした？", now=NOW)

        # 日本時間の今日 0:00 から今まで
        assert result["timestamp"] == {
            "gte": "2026-11-01T15:00:00.000000",
            "lte": "2026-11-02T03:00:00.000000",
        }

    def test_future_end_is_clamped_to_now(self, openai_llm):
        set_llm_answer(openai_llm, {"gte": "2026-11-01", "lte": "2026-11-30"})

        result = openai_llm.extract_tag("今月どう？", now=NOW)

        assert result["timestamp"]["lte"] == "2026-11-02T03:00:00.000000"

    def test_future_period_is_ignored(self, openai_llm):
        set_llm_answer(openai_llm, {"gte": "2026-12-01", "lte": "2026-12-31"})

        result = openai_llm.extract_tag("来月の予定は？", now=NOW)

        assert "timestamp" not in result

    def test_reversed_dates_are_swapped(self, openai_llm):
        set_llm_answer(openai_llm, {"gte": "2026-10-31", "lte": "2026-10-01"})

        result = openai_llm.extract_tag("先月", now=NOW)

        assert result["timestamp"]["gte"] == "2026-09-30T15:00:00.000000"

    def test_invalid_date_falls_back_to_no_filter(self, openai_llm):
        set_llm_answer(openai_llm, {"gte": "最近", "lte": "2026-10-31"})

        result = openai_llm.extract_tag("最近どう？", now=NOW)

        assert result == {"content": "最近どう？"}

    def test_api_error_falls_back_to_no_filter(self, openai_llm):
        openai_llm.model.chat.completions.create.side_effect = Exception("API エラー")

        result = openai_llm.extract_tag("最近どう？", now=NOW)

        assert result == {"content": "最近どう？"}

    def test_reasoning_mode_omits_temperature(self, openai_llm, monkeypatch):
        monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "low")
        set_llm_answer(openai_llm, None)

        openai_llm.extract_tag("最近どう？", now=NOW)

        params = openai_llm.model.chat.completions.create.call_args.kwargs
        assert "temperature" not in params
        assert params["max_completion_tokens"] > 200

    def test_local_model_does_not_filter(self, openai_llm):
        openai_llm.model_type = "openai-20b"

        result = openai_llm.extract_tag("最近どう？", now=NOW)

        assert result == {"content": "最近どう？"}
        openai_llm.model.chat.completions.create.assert_not_called()
