import builtins
import model_tuner


def test_custom_help_override(monkeypatch, capsys):
    # Simulate calling help() on the model_tuner module
    monkeypatch.setattr(builtins, "help", model_tuner.custom_help)

    # Call help() and capture output
    model_tuner.custom_help(model_tuner)

    captured = capsys.readouterr()

    # Adjusted assertions for actual content
    assert "Version: 0.0.31b" in captured.out
    assert "https://pypi.org/project/model-tuner/" in captured.out


def test_custom_help_on_regular_object(capsys):
    # Call help on a regular object like `int`
    model_tuner.custom_help(int)
    captured = capsys.readouterr()

    # Adjusted for formatting in Python 3
    assert "Help on class int in module builtins" in captured.out
