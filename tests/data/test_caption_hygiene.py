"""Decision 6: caption code must not reference the held-out instruction set.

This test reads only the caption code (and any caption template data files), never the
held-out data itself.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_SRC = REPO_ROOT / "src" / "lstmabar" / "data"
CAPTION_FILES = [
    DATA_SRC / "captions.py",
    DATA_SRC / "vocab.py",  # caption vocabulary rules shared with the leakage tests
    # caption template data files (none yet; templates live in code) are picked up below
    *sorted(DATA_SRC.glob("caption*.yaml")),
    *sorted(DATA_SRC.glob("caption*.json")),
    *sorted(DATA_SRC.glob("templates/**/*")),
]

_FORBIDDEN = re.compile("g" + "old", re.IGNORECASE)  # split so this file does not match itself


def test_caption_code_never_mentions_the_heldout_dir():
    for path in CAPTION_FILES:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        hits = [
            f"{path.name}:{i}: {line.strip()}"
            for i, line in enumerate(text.splitlines(), 1)
            if _FORBIDDEN.search(line)
        ]
        assert not hits, "caption code references the held-out set:\n" + "\n".join(hits)


def test_hygiene_check_catches_a_reference(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('PATH = "data/' + "g" + 'old/instructions.yaml"\n', encoding="utf-8")
    assert _FORBIDDEN.search(bad.read_text(encoding="utf-8"))
