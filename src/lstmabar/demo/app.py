"""M1 demo: manual knobs on the differentiable pedalboard with A/B playback.

Run with ``lstmabar demo`` (needs the ``demo`` extra: ``uv sync --extra demo``). Controls are
generated from each block's ``param_specs``, so new blocks and knobs show up automatically.
Slider values follow the convention in :mod:`lstmabar.demo.render`: physical units for linear
knobs, normalized position for log-taper knobs.
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
    log_slider_info,
    physical_from_slider,
    prepare_input,
    random_knobs,
    render,
    slider_config,
    slider_from_physical,
    to_int16,
)

MAX_SECONDS = 15.0
MAX_FILE_SIZE = "25mb"


def _import_gradio():
    msg = "The demo needs gradio. Install the demo extra: uv sync --extra demo"
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(msg, name="gradio") from e
    if not hasattr(gr, "Blocks"):  # e.g. an empty namespace dir left behind by an uninstall
        raise ImportError(msg, name="gradio")
    return gr


def _default_board_factory():
    from lstmabar.dsp.pedalboard import default_pedalboard

    return default_pedalboard()


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def build_app(board_factory: Callable[[], Any] | None = None):
    """Build the Gradio Blocks app around a pedalboard (default: ``default_pedalboard()``)."""
    gr = _import_gradio()
    board = (board_factory or _default_board_factory)()
    board.eval()
    sr = int(board.sample_rate)
    specs = list(iter_specs(board))  # flat (block, spec) list in board order
    n_blocks = len(board.block_names)
    defaults = default_knobs(board)
    rng = np.random.default_rng()

    def collect(enabled_vals, slider_vals):
        enabled = dict(zip(board.block_names, (bool(v) for v in enabled_vals), strict=True))
        values: dict[str, dict[str, float]] = {name: {} for name in board.block_names}
        for (name, spec), s in zip(specs, slider_vals, strict=True):
            values[name][spec.name] = physical_from_slider(spec, s)
        return knobs_to_params(board, values, enabled)

    def on_render(audio_value, riff_kind, match, *controls):
        enabled_vals, slider_vals = controls[:n_blocks], controls[n_blocks:]
        params = collect(enabled_vals, slider_vals)
        try:
            clip = prepare_input(audio_value, riff_kind, sr, MAX_SECONDS)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        dry, wet = render(board, clip, params, match_loudness=bool(match))
        return (sr, to_int16(dry)), (sr, to_int16(wet)), describe_text(board, params)

    def on_reset():
        sliders = [slider_from_physical(spec, defaults[name][spec.name]) for name, spec in specs]
        return [True] * n_blocks + sliders

    def on_randomize():
        knobs = random_knobs(board, rng)
        return [slider_from_physical(spec, knobs[name][spec.name]) for name, spec in specs]

    with gr.Blocks(title="LSTMABAR pedalboard") as app:
        gr.Markdown(
            "# LSTMABAR — differentiable pedalboard (M1)\n"
            "Upload or record a guitar clip (or pick an example riff), set the knobs, and "
            f"compare dry vs wet. Clips are trimmed to {MAX_SECONDS:.0f} s."
        )
        with gr.Row():
            with gr.Column(scale=1):
                # filepath (not numpy): gradio would decode non-WAV uploads with a system
                # ffmpeg; we decode ourselves with libsndfile / PyAV instead.
                audio_in = gr.Audio(
                    sources=["upload", "microphone"], type="filepath", label="Input clip"
                )
                riff = gr.Dropdown(
                    choices=list(RIFF_KINDS),
                    value="power_chords",
                    label="Example riff (used when no clip is given)",
                )
                match = gr.Checkbox(
                    value=True,
                    label="Loudness-match A/B (wet RMS = dry RMS; "
                    "wet may stay quieter when peak-limited)",
                )
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
                            slider = gr.Slider(**slider_config(spec))
                            sliders.append(slider)
                            if spec.taper == "log":
                                # Show the physical value of a log knob as it moves.
                                slider.change(
                                    lambda s, spec=spec: gr.update(
                                        info=log_slider_info(spec, physical_from_slider(spec, s))
                                    ),
                                    inputs=slider,
                                    outputs=slider,
                                    show_progress="hidden",
                                )

        render_btn.click(
            on_render,
            inputs=[audio_in, riff, match, *checkboxes, *sliders],
            outputs=[dry_out, wet_out, readout],
        )
        reset_btn.click(on_reset, inputs=None, outputs=[*checkboxes, *sliders])
        random_btn.click(on_randomize, inputs=None, outputs=sliders)

    return app
