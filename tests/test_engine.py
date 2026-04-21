import pytest
from unittest.mock import MagicMock
from chatbot.engine import ChatEngine
from chatbot.config import RemoteConfig

@pytest.fixture
def mock_config():
    return RemoteConfig(
        chat_mode="remote",
        model_name="gemini-2.5-flash",
        api_key="fake-key",
        max_history_turns=2
    )

@pytest.fixture
def engine_with_mocks(mock_config, mocker):
    """Returns (engine, backend_mock, judge_mock) with real backends patched out."""
    backend = MagicMock()
    judge = MagicMock()
    mocker.patch("chatbot.engine.create_backend", return_value=backend)
    mocker.patch("chatbot.engine.create_judge_backend", return_value=judge)
    engine = ChatEngine(mock_config)
    return engine, backend, judge

def test_engine_chat_success(engine_with_mocks):
    engine, backend, judge = engine_with_mocks

    judge.get_response.return_value = "This is a safe response"
    backend.get_response.return_value = "This is a safe response"

    response = engine.chat("Hello there")

    assert response == "This is a safe response"
    assert len(engine.history) == 2
    assert engine.history[0]["role"] == "user"
    assert engine.history[0]["content"] == "Hello there"
    assert engine.history[1]["role"] == "model"
    assert engine.history[1]["content"] == "This is a safe response"

def test_engine_chat_backend_failure(engine_with_mocks):
    engine, backend, judge = engine_with_mocks

    judge.get_response.return_value = "SAFE"
    backend.get_response.side_effect = ConnectionError("Network down")

    response = engine.chat("Hello there")

    assert "❌" in response
    assert "Network down" in response
    # History should be popped, leaving it empty
    assert len(engine.history) == 0

def test_engine_history_trimming(engine_with_mocks):
    # max_history_turns is 2 (so max 4 messages)
    engine, backend, judge = engine_with_mocks

    judge.get_response.return_value = "SAFE RESPONSE"
    backend.get_response.return_value = "SAFE RESPONSE"

    # Send 3 messages (should result in 6 history items, but trimmed to 4)
    engine.chat("Message 1")
    engine.chat("Message 2")
    engine.chat("Message 3")

    assert len(engine.history) == 5
    assert engine.history[0]["role"] == "model"
    assert engine.history[1]["content"] == "Message 2"
    assert engine.history[2]["content"] == "SAFE RESPONSE"
    assert engine.history[3]["content"] == "Message 3"
    assert engine.history[4]["content"] == "SAFE RESPONSE"
