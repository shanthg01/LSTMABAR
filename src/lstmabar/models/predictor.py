"""P6 parameter predictor (task 6.1; design §4.1-4.2) on frozen, cached embeddings.

``text emb (per caption) -> 2-layer adapter --+
                                              +--> fusion MLP --> per-block heads
 audio emb (per item)    -> 2-layer adapter --+                  (knobs + one gate logit)``

The output is a complete normalized board as a ``(B, P)`` tensor in
:data:`lstmabar.eval.baselines.base.BOARD_LAYOUT` order (per block: its knobs, then
``"enabled"``), which is what every method is scored on.

Ablation flags (6.5):

- ``inputs``: ``both`` | ``text`` | ``audio`` (the unused adapter is not built);
- ``mapping`` (how the heads parameterize knobs):

  - ``physics`` (default): each knob head emits a logit, ``u = sigmoid(z)`` is the
    **normalized** knob, and the block's ``ParamSpec`` taper (linear or log) maps it to physical
    units when the board renders. Predictions are bounded by the physical knob range and the
    log tapers come from the pedal-derived ranges (design §4.2-4.4). This is the board
    contract (``dsp/base.py``).
  - ``unconstrained``: each knob head emits a linear (unbounded, no sigmoid) value ``z`` read
    as a physical value scaled linearly by the spec range, ``v = min + (max - min) * z``, with
    **no taper**. ``L_param`` is the L1 in that linear-physical space (labels are mapped
    through the taper to physical units and scaled by ``(max - min)``). For scoring and
    rendering, ``v`` is converted back to a normalized knob with ``ParamSpec.normalize``
    (taper + clamp to [0, 1]). For linear-taper knobs this drops only the sigmoid bound; for
    log-taper knobs (``drive.pre_hpf_hz``, ``eq.mid_hz``, compressor ratio/attack/release)
    it also drops the taper.

  Gates are a sigmoid of a logit in both mappings (trained with BCE).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lstmabar.dsp.base import ParamSpec
from lstmabar.dsp.pedalboard import ENABLED, default_pedalboard

INPUTS = ("both", "text", "audio")
MAPPINGS = ("physics", "unconstrained")


@dataclass(frozen=True)
class BoardHeadLayout:
    """Index bookkeeping between per-block heads and the flat board vector."""

    layout: tuple[tuple[str, str], ...]  # BOARD_LAYOUT
    blocks: tuple[str, ...]
    specs: tuple[ParamSpec, ...]  # one per knob, in knob order
    knob_index: tuple[int, ...]  # knob j -> position in the board vector
    gate_index: tuple[int, ...]  # block g -> position of its "enabled" entry
    knob_block: tuple[int, ...]  # knob j -> its block g
    knobs_per_block: tuple[int, ...]

    @classmethod
    def default(cls) -> BoardHeadLayout:
        board = default_pedalboard()
        layout = tuple(board.layout())
        blocks = tuple(board.block_names)
        specs: list[ParamSpec] = []
        knob_index: list[int] = []
        gate_index: list[int] = []
        knob_block: list[int] = []
        for pos, (b, k) in enumerate(layout):
            g = blocks.index(b)
            if k == ENABLED:
                gate_index.append(pos)
            else:
                spec = next(s for s in board.blocks[b].param_specs if s.name == k)
                specs.append(spec)
                knob_index.append(pos)
                knob_block.append(g)
        per_block = tuple(len(board.blocks[b].param_specs) for b in blocks)
        return cls(
            layout=layout,
            blocks=blocks,
            specs=tuple(specs),
            knob_index=tuple(knob_index),
            gate_index=tuple(gate_index),
            knob_block=tuple(knob_block),
            knobs_per_block=per_block,
        )

    @property
    def n_knobs(self) -> int:
        return len(self.knob_index)

    @property
    def n_gates(self) -> int:
        return len(self.gate_index)

    def split(self, board: Tensor) -> tuple[Tensor, Tensor]:
        """``(B, P)`` board -> ``(B, K)`` knobs, ``(B, G)`` gates."""
        return board[:, list(self.knob_index)], board[:, list(self.gate_index)]

    def join(self, knobs: Tensor, gates: Tensor) -> Tensor:
        out = knobs.new_zeros(knobs.shape[0], len(self.layout))
        out[:, list(self.knob_index)] = knobs
        out[:, list(self.gate_index)] = gates
        return out

    # -- unconstrained mapping helpers ---------------------------------------------------

    def _range(self, like: Tensor) -> tuple[Tensor, Tensor]:
        lo = torch.tensor([s.min for s in self.specs], dtype=like.dtype, device=like.device)
        hi = torch.tensor([s.max for s in self.specs], dtype=like.dtype, device=like.device)
        return lo, hi

    def normalized_to_linear(self, u: Tensor) -> Tensor:
        """Normalized knobs ``(B, K)`` -> physical values scaled linearly to [0, 1] by the spec
        range (no taper): the ``unconstrained`` regression target."""
        lo, hi = self._range(u)
        phys = torch.stack([s.denormalize(u[:, j]) for j, s in enumerate(self.specs)], dim=-1)
        return (phys - lo) / (hi - lo)

    def linear_to_normalized(self, z: Tensor) -> Tensor:
        """Inverse of :meth:`normalized_to_linear` for unbounded ``z``: ``v = min + (max-min)z``
        (clamped into the physical range first, so log tapers stay defined), then the spec's
        taper normalization, clamped to [0, 1]."""
        lo, hi = self._range(z)
        v = torch.minimum(torch.maximum(lo + (hi - lo) * z, lo), hi)
        return torch.stack([s.normalize(v[:, j]) for j, s in enumerate(self.specs)], dim=-1)


def _adapter(dim_in: int, dim: int, dropout: float) -> nn.Sequential:
    """The 2-layer adapter on a frozen embedding (design §4.1)."""
    return nn.Sequential(
        nn.Linear(dim_in, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
    )


@dataclass(frozen=True)
class PredictorConfig:
    text_dim: int
    audio_dim: int
    inputs: str = "both"
    mapping: str = "physics"
    adapter_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.inputs not in INPUTS:
            raise ValueError(f"model.inputs must be one of {INPUTS}, got {self.inputs!r}")
        if self.mapping not in MAPPINGS:
            raise ValueError(f"model.mapping must be one of {MAPPINGS}, got {self.mapping!r}")

    def to_dict(self) -> dict:
        return asdict(self)


class ParamPredictor(nn.Module):
    """Adapters + fusion MLP + per-block heads (knobs and one gate logit per block)."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.heads_layout = BoardHeadLayout.default()
        c = config
        self.text_adapter = (
            _adapter(c.text_dim, c.adapter_dim, c.dropout) if c.inputs != "audio" else None
        )
        self.audio_adapter = (
            _adapter(c.audio_dim, c.adapter_dim, c.dropout) if c.inputs != "text" else None
        )
        n_in = c.adapter_dim * (2 if c.inputs == "both" else 1)
        self.fusion = nn.Sequential(
            nn.Linear(n_in, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                b: nn.Linear(c.hidden_dim, n + 1)  # n knobs + 1 gate logit
                for b, n in zip(
                    self.heads_layout.blocks, self.heads_layout.knobs_per_block, strict=True
                )
            }
        )

    def forward(self, text: Tensor | None, audio: Tensor | None) -> dict[str, Tensor]:
        """Returns ``raw`` ``(B, K)`` knob head outputs (logits for ``physics``, linear values
        for ``unconstrained``), ``knobs`` ``(B, K)`` normalized knobs in [0, 1] and
        ``gate_logits`` ``(B, G)``."""
        feats = []
        if self.text_adapter is not None:
            if text is None:
                raise ValueError("this model needs text embeddings")
            feats.append(self.text_adapter(text))
        if self.audio_adapter is not None:
            if audio is None:
                raise ValueError("this model needs audio embeddings")
            feats.append(self.audio_adapter(audio))
        h = self.fusion(torch.cat(feats, dim=-1))
        raws, gates = [], []
        for b in self.heads_layout.blocks:
            o = self.heads[b](h)
            raws.append(o[:, :-1])
            gates.append(o[:, -1])
        raw = torch.cat(raws, dim=-1)
        if self.config.mapping == "physics":
            knobs = torch.sigmoid(raw)
        else:
            knobs = self.heads_layout.linear_to_normalized(raw)
        return {"raw": raw, "knobs": knobs, "gate_logits": torch.stack(gates, dim=-1)}

    def board(self, out: dict[str, Tensor]) -> Tensor:
        """``(B, P)`` normalized board (gates = sigmoid of the logits) in layout order."""
        return self.heads_layout.join(out["knobs"], torch.sigmoid(out["gate_logits"]))


CHECKPOINT_FORMAT = "lstmabar.p6.predictor/1"


def save_checkpoint(
    path: str | Path, model: ParamPredictor, meta: dict[str, Any] | None = None
) -> Path:
    """State dict + predictor config + ``meta`` (resolved run config, git SHA, model ids,
    epoch, val metrics; plain containers only, so the file loads with ``weights_only``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "predictor": model.config.to_dict(),
        "layout": [list(p) for p in model.heads_layout.layout],
        "state_dict": model.state_dict(),
        "meta": dict(meta or {}),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path) -> tuple[ParamPredictor, dict[str, Any]]:
    """``(model in eval mode, payload)``; refuses a checkpoint whose board layout differs
    from the current ``default_pedalboard`` (the chain or knobs changed since training)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"not a P6 predictor checkpoint: {Path(path).name}")
    missing = [k for k in ("predictor", "layout", "state_dict", "meta") if k not in payload]
    if missing:
        raise ValueError(f"{Path(path).name}: checkpoint lacks {missing}")
    model = ParamPredictor(PredictorConfig(**payload["predictor"]))
    if [tuple(p) for p in payload["layout"]] != list(model.heads_layout.layout):
        raise ValueError(
            f"{Path(path).name}: board layout differs from the current default_pedalboard"
        )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


__all__ = [
    "CHECKPOINT_FORMAT",
    "INPUTS",
    "MAPPINGS",
    "BoardHeadLayout",
    "ParamPredictor",
    "PredictorConfig",
    "load_checkpoint",
    "save_checkpoint",
]
