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
