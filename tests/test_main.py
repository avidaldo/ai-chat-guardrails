from pathlib import Path

from pydantic import ValidationError

import main


def test_main_shows_friendly_message_when_env_file_is_missing(mocker, capsys):
    validation_error = ValidationError.from_exception_data(
        "BaseChatConfig",
        [{"type": "missing", "loc": ("chat_mode",), "input": {}, "msg": "Field required"}],
    )
    mocker.patch("main.load_config", side_effect=validation_error)
    mocker.patch("main.PROJECT_ROOT", Path("/tmp/nonexistent-ai-chat-guardrails"))

    main.main()

    output = capsys.readouterr().out
    assert "missing .env file" in output
    assert ".env.local.example" in output
    assert ".env.remote.example" in output


def test_main_preserves_generic_config_errors(mocker, capsys):
    mocker.patch("main.load_config", side_effect=RuntimeError("broken config"))

    main.main()

    output = capsys.readouterr().out
    assert "Configuration error: broken config" in output