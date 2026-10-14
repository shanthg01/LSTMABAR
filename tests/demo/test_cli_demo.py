import pytest

from lstmabar.cli import SHARE_WARNING, build_parser, demo_launch_kwargs


def test_demo_defaults_are_local_and_capped(capsys):
    args = build_parser().parse_args(["demo"])
    kwargs = demo_launch_kwargs(args)
    assert kwargs == {"server_port": 7860, "share": False, "auth": None, "max_file_size": "25mb"}
    assert capsys.readouterr().err == ""


def test_demo_share_warns_and_passes_auth(capsys):
    args = build_parser().parse_args(["demo", "--share", "--port", "9000", "--auth", "me:pw:x"])
    kwargs = demo_launch_kwargs(args)
    assert kwargs["share"] is True and kwargs["server_port"] == 9000
    assert kwargs["auth"] == ("me", "pw:x")
    assert SHARE_WARNING in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["nopass", ":pw", "user:"])
def test_demo_rejects_malformed_auth(bad):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["demo", "--auth", bad])


def test_demo_config_resolves_from_the_repo_root(tmp_path, monkeypatch, capsys):
    from lstmabar.cli import demo_config

    monkeypatch.chdir(tmp_path)  # outside the repo: the relative default would not exist
    cfg = demo_config(build_parser().parse_args(["demo", "demo.text_method=noop"]))
    assert cfg is not None and cfg.demo.text_method == "noop"
    assert demo_config(build_parser().parse_args(["demo", "--config", "nope.yaml"])) is None
    assert "could not load the demo config" in capsys.readouterr().err


def test_demo_config_falls_back_without_the_default(monkeypatch, capsys):
    import lstmabar.demo.text as text
    from lstmabar.cli import demo_config

    monkeypatch.setattr(text, "DEFAULT_CONFIG", text.REPO_ROOT / "missing-demo.yaml")
    assert demo_config(build_parser().parse_args(["demo"])) is None
    err = capsys.readouterr().err
    assert "text action is disabled" in err and "missing-demo" not in err
