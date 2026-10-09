"""M1 demo: manual knobs on the differentiable pedalboard with A/B playback.

Run with ``lstmabar demo`` (needs the ``demo`` extra: ``uv sync --extra demo``). Controls are
generated from each block's ``param_specs``, so new blocks and knobs show up automatically.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from lstmabar.audio import RIFF_KINDS
from lstmabar.demo.render import (
    default_knobs,
    describe_text,
    iter_specs,
    knobs_to_params,
    prepare_input,
    random_knobs,
    render,
    to_int16,
)

MAX_SECONDS = 15.0
SLIDER_STEPS = 1000


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The demo needs gradio. Install the demo extra: uv sync --extra demo"
        ) from e
    return gr


def _default_board_factory():
    from lstmabar.dsp.pedalboard import default_pedalboard

    return default_pedalboard()


def _slider_step(spec) -> float:
    """A step fine enough for the low end of the range (log knobs span decades)."""
    span = spec.max - spec.min
    step = span / SLIDER_STEPS
    if spec.taper == "log":
        step = min(step, spec.min / 100)
    return float(f"{step:.2g}")


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def build_app(board_factory: Callable[[], Any] | None = None):
    """Build the Gradio Blocks app around a pedalboard (default: ``default_pedalboard()``)."""
    gr = _import_gradio()
    board = (board_factory or _default_board_factory)()
    board.eval()
    sr = int(board.sample_rate)
    specs = list(iter_specs(board))  # flat (block, spec) list in board order
    defaults = default_knobs(board)
    rng = np.random.default_rng()

    def collect(enabled_vals, slider_vals):
        enabled = dict(zip(board.block_names, (bool(v) for v in enabled_vals), strict=True))
        values: dict[str, dict[str, float]] = {name: {} for name in board.block_names}
        for (name, spec), v in zip(specs, slider_vals, strict=True):
            values[name][spec.name] = float(spec.default if v is None else v)
        return knobs_to_params(board, values, enabled)

    n_blocks = len(board.block_names)

    def on_render(audio_value, riff_kind, match, *controls):
        enabled_vals, slider_vals = controls[:n_blocks], controls[n_blocks:]
        params = collect(enabled_vals, slider_vals)
        clip = prepare_input(audio_value, riff_kind, sr, MAX_SECONDS)
        dry, wet = render(board, clip, params, match_loudness=bool(match))
        return (sr, to_int16(dry)), (sr, to_int16(wet)), describe_text(board, params)

    def on_reset():
        return [True] * n_blocks + [defaults[name][spec.name] for name, spec in specs]

    def on_randomize():
        knobs = random_knobs(board, rng)
        return [knobs[name][spec.name] for name, spec in specs]

    with gr.Blocks(title="LSTMABAR pedalboard") as app:
        gr.Markdown(
            "# LSTMABAR — differentiable pedalboard (M1)\n"
            "Upload or record a guitar clip (or pick an example riff), set the knobs, and "
            f"compare dry vs wet. Clips are capped at {MAX_SECONDS:.0f} s."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["upload", "microphone"], type="numpy", label="Input clip"
                )
                riff = gr.Dropdown(
                    choices=list(RIFF_KINDS),
                    value="power_chords",
                    label="Example riff (used when no clip is given)",
                )
                match = gr.Checkbox(value=True, label="Loudness-match A/B (wet RMS = dry RMS)")
                with gr.Row():
                    render_btn = gr.Button("Render", variant="primary")
                    reset_btn = gr.Button("Reset to defaults")
                    random_btn = gr.Button("Randomize")
                dry_out = gr.Audio(label="A: Dry", type="numpy", interactive=False)
                wet_out = gr.Audio(label="B: Wet", type="numpy", interactive=False)
                readout = gr.Markdown()
            with gr.Column(scale=1):
                checkboxes = []
                sliders = []
                for name in board.block_names:
                    with gr.Accordion(_title(name), open=True):
                        checkboxes.append(gr.Checkbox(value=True, label="enabled"))
                        for spec in board.blocks[name].param_specs:
                            unit = f" ({spec.unit})" if spec.unit else ""
                            sliders.append(
                                gr.Slider(
                                    minimum=spec.min,
                                    maximum=spec.max,
                                    value=spec.default,
                                    step=_slider_step(spec),
                                    label=f"{spec.name}{unit}",
                                    info="log taper" if spec.taper == "log" else None,
                                )
                            )

        render_btn.click(
            on_render,
            inputs=[audio_in, riff, match, *checkboxes, *sliders],
            outputs=[dry_out, wet_out, readout],
        )
        reset_btn.click(on_reset, inputs=None, outputs=[*checkboxes, *sliders])
        random_btn.click(on_randomize, inputs=None, outputs=sliders)

    return app
