"""M1/M2 demo: knobs on the differentiable pedalboard, pedal presets and A/B playback.

Run with ``lstmabar demo`` (needs the ``demo`` extra: ``uv sync --extra demo``; the archetype
panel also wants ``analysis``, and white-box renders are much faster with ``whitebox``).

- **Manual board** (M1): controls are generated from each block's ``param_specs``, so new
  blocks and knobs show up automatically. Slider values follow the convention in
  :mod:`lstmabar.demo.render`: physical units for linear knobs, normalized position for
  log-taper knobs.
- **Pedal presets** (M2): picking a pedal from the knowledge base shows its own dials (0–10);
  moving them derives the grey-box board knobs and writes them into the sliders. The EQ is
  locked while a pedal is active; see :mod:`lstmabar.demo.presets` for the exact behaviour.
- **White-box A/B**: pedals with a registered circuit simulation get a third player.
- **Archetype panel** (P3.5): before/after archetype readout and H1–H10 levels, see
  :mod:`lstmabar.demo.archetype_panel`.
"""

import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from lstmabar.audio import RIFF_KINDS
from lstmabar.demo.archetype_panel import (
    ANALYSIS_SECONDS,
    analyse_signals,
    archetype_markdown,
    archetype_rows,
    harmonic_rows,
)
from lstmabar.demo.presets import (
    MANUAL,
    WHITEBOX_MAX_SECONDS,
    WHITEBOX_NEEDS_NUMBA,
    derive_preset,
    dial_config,
    is_locked,
    pedal_choices,
    preset_board_values,
    preset_controls,
    preset_readout,
    resolve_render,
    whitebox_available,
    whitebox_ready,
    whitebox_render,
)
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

log = logging.getLogger(__name__)

MAX_SECONDS = 15.0
MAX_FILE_SIZE = "25mb"
EQ_LOCK_NOTE = (
    "*EQ locked (sliders and on/off): while a pedal is active this block holds the pedal's "
    'fitted tone stack. Pick "Manual board" to use it as a free EQ.*'
)
PANEL_PLACEHOLDER = "*Render to see the archetype readout.*"
PANEL_OFF = (
    "*Archetype analysis is off. Tick the box above to see the sine/triangle/square/saw + "
    "noise readout (adds a few seconds per render).*"
)
BAR_COLUMNS = ["signal", "component", "share"]
HARMONIC_COLUMNS = ["signal", "harmonic", "dbc"]


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


def _default_kb() -> dict:
    from lstmabar.physics.kb import load_kb

    try:
        return load_kb()
    except Exception:  # the manual board still works without the knowledge base
        log.exception("could not load the pedal knowledge base")
        return {}


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def _private(gr) -> dict:
    """Listener kwargs that keep an internal helper event out of the public API."""
    if "api_visibility" in inspect.signature(gr.Blocks.load).parameters:  # gradio >= 6
        return {"api_visibility": "private"}
    return {"show_api": False}  # pragma: no cover - gradio 5


def _frame(rows: list[dict], columns: list[str]):
    import pandas as pd  # a gradio dependency

    return pd.DataFrame(rows, columns=columns)


def build_app(
    board_factory: Callable[[], Any] | None = None,
    kb: Mapping[str, Any] | None = None,
):
    """Build the Gradio Blocks app around a pedalboard (default: ``default_pedalboard()``)
    and a pedal knowledge base (default: ``load_kb()``; ``{}`` gives the manual board only)."""
    gr = _import_gradio()
    board = (board_factory or _default_board_factory)()
    board.eval()
    sr = int(board.sample_rate)
    specs = list(iter_specs(board))  # flat (block, spec) list in board order
    n_blocks = len(board.block_names)
    n_sliders = len(specs)
    defaults = default_knobs(board)
    rng = np.random.default_rng()
    kb = dict(_default_kb() if kb is None else kb)
    # Pedal dials are flattened in kb order, then pots order.
    dial_slices: dict[str, slice] = {}
    start = 0
    for pid, pedal in kb.items():
        dial_slices[pid] = slice(start, start + len(pedal.pots))
        start += len(pedal.pots)
    n_dials = start
    wb_seconds = WHITEBOX_MAX_SECONDS
    private = _private(gr)

    def active_pedal(pedal_id):
        return kb.get(pedal_id) if pedal_id != MANUAL else None

    def collect(enabled_vals, slider_vals):
        enabled = dict(zip(board.block_names, (bool(v) for v in enabled_vals), strict=True))
        values: dict[str, dict[str, float]] = {name: {} for name in board.block_names}
        for (name, spec), s in zip(specs, slider_vals, strict=True):
            values[name][spec.name] = physical_from_slider(spec, s)
        return knobs_to_params(board, values, enabled)

    def board_updates(preset):
        """Checkbox + slider updates for a pedal preset (``None``: unlock, keep values)."""
        if preset is None:
            return [gr.update(interactive=True)] * (n_blocks + n_sliders)
        enabled, sliders = preset_controls(board, preset)
        return [
            gr.update(value=v, interactive=not is_locked(name, True))
            for name, v in zip(board.block_names, enabled, strict=True)
        ] + [
            gr.update(value=s, interactive=not is_locked(name, True))
            for (name, _), s in zip(specs, sliders, strict=True)
        ]

    def on_render(audio_value, riff_kind, match, pedal_id, analyse, derived, *controls):
        dials = controls[:n_dials]
        enabled_vals = controls[n_dials : n_dials + n_blocks]
        slider_vals = controls[n_dials + n_blocks :]
        pedal = active_pedal(pedal_id)
        preset, edited, wb, wb_msg = None, False, False, ""
        if pedal is not None:
            choice = resolve_render(
                board,
                pedal,
                list(dials[dial_slices[pedal.id]]),
                derived_for(derived, pedal.id),
                enabled_vals,
                slider_vals,
                sr,
            )
            preset, edited = choice.preset, choice.edited
            wb = whitebox_available(pedal.id)
            if wb and not whitebox_ready():
                wb, wb_msg = False, WHITEBOX_NEEDS_NUMBA
        if preset is not None and not edited:
            enabled, values = preset_board_values(board, preset)  # exact, not slider-rounded
            params = knobs_to_params(board, values, enabled)
        else:
            params = collect(enabled_vals, slider_vals)
        try:
            clip = prepare_input(audio_value, riff_kind, sr, MAX_SECONDS)
        except ValueError as e:
            raise gr.Error(str(e)) from e
        dry, wet = render(board, clip, params, match_loudness=bool(match))

        wb_audio = None
        if wb:  # same dial state as the grey-box preset
            res = whitebox_render(pedal, preset.knobs, clip, sr, bool(match), wb_seconds)
            wb_audio, wb_msg = res.audio, res.message
        wb_update = gr.update(
            value=None if wb_audio is None else (sr, to_int16(wb_audio)), visible=wb
        )
        preset_md = preset_readout(pedal, preset, wb, edited) if preset is not None else ""

        if analyse:
            wet_label = "wet" if preset is None else ("custom" if edited else "grey-box")
            signals = {"dry": dry, wet_label: wet}
            if wb_audio is not None:
                signals["white-box"] = wb_audio
            panel = analyse_signals(signals, sr, ANALYSIS_SECONDS)
            arche_md = archetype_markdown(panel)
            bars = _frame(archetype_rows(panel), BAR_COLUMNS)
            harms = _frame(harmonic_rows(panel), HARMONIC_COLUMNS)
        else:
            arche_md = PANEL_OFF
            bars, harms = _frame([], BAR_COLUMNS), _frame([], HARMONIC_COLUMNS)
        return (
            (sr, to_int16(dry)),
            (sr, to_int16(wet)),
            wb_update,
            wb_msg,
            describe_text(board, params),
            preset_md,
            arche_md,
            bars,
            harms,
        )

    def derived_for(derived, pid):
        """Dial values the board sliders were last derived from for ``pid`` (or None)."""
        if isinstance(derived, dict) and derived.get("pedal") == pid:
            return list(derived["dials"])
        return None

    def wb_shown(pid):
        return whitebox_available(pid) and whitebox_ready()

    def cleared_outputs():
        """Clear the previous render (B, C message, readouts, panel) on a pedal switch."""
        return [
            None,
            "",
            "",
            PANEL_PLACEHOLDER,
            _frame([], BAR_COLUMNS),
            _frame([], HARMONIC_COLUMNS),
        ]

    def on_pedal(pedal_id, *dials):
        pedal = active_pedal(pedal_id)
        groups = [gr.update(visible=pedal is not None and pid == pedal.id) for pid in kb]
        if pedal is None:
            hidden = [gr.update(visible=False), gr.update(visible=False, value=None)]
            return [*groups, "", *hidden, *cleared_outputs(), None, *board_updates(None)]
        pedal_dials = list(dials[dial_slices[pedal.id]])
        preset = derive_preset(pedal, pedal_dials, sr)
        wb = wb_shown(pedal.id)
        return [
            *groups,
            preset_readout(pedal, preset, wb),
            gr.update(visible=True),
            gr.update(visible=wb, value=None),
            *cleared_outputs(),
            {"pedal": pedal.id, "dials": pedal_dials},
            *board_updates(preset),
        ]

    def make_on_dial(pid):
        pedal = kb[pid]

        def on_dial(pedal_id, *pedal_dials):
            if pedal_id != pid:  # the pedal was switched while this derive was queued
                return [gr.update()] * (2 + n_blocks + n_sliders)
            preset = derive_preset(pedal, list(pedal_dials), sr)
            md = preset_readout(pedal, preset, wb_shown(pid))
            state = {"pedal": pid, "dials": list(pedal_dials)}
            return [md, state, *board_updates(preset)]

        on_dial.__name__ = f"on_dial_{pid}"
        return on_dial

    def dial_outputs(pedal, dial_vals):
        """Reset/randomize outputs for an active pedal: its dials, the derived board, readout."""
        dial_updates = [gr.update()] * n_dials
        for i, v in zip(range(n_dials)[dial_slices[pedal.id]], dial_vals, strict=True):
            dial_updates[i] = v
        preset = derive_preset(pedal, dial_vals, sr)
        md = preset_readout(pedal, preset, wb_shown(pedal.id))
        state = {"pedal": pedal.id, "dials": list(dial_vals)}
        return [*dial_updates, *board_updates(preset), md, state]

    def on_reset(pedal_id):
        pedal = active_pedal(pedal_id)
        if pedal is None:
            sliders = [
                slider_from_physical(spec, defaults[name][spec.name]) for name, spec in specs
            ]
            return [gr.update()] * n_dials + [True] * n_blocks + sliders + ["", None]
        return dial_outputs(pedal, [dial_config(p)["value"] for p in pedal.pots.values()])

    def on_randomize(pedal_id):
        pedal = active_pedal(pedal_id)
        if pedal is None:
            knobs = random_knobs(board, rng)
            sliders = [slider_from_physical(spec, knobs[name][spec.name]) for name, spec in specs]
            return [gr.update()] * (n_dials + n_blocks) + sliders + [gr.update(), None]
        return dial_outputs(pedal, [round(float(rng.uniform(0, 10)), 1) for _ in pedal.pots])

    with gr.Blocks(title="LSTMABAR pedalboard") as app:
        gr.Markdown(
            "# LSTMABAR — physics-grounded pedalboard (M2)\n"
            "Upload or record a guitar clip (or pick an example riff), pick a pedal and set its "
            "knobs (or use the manual board), and compare dry vs grey-box (vs white-box). "
            f"Clips are trimmed to {MAX_SECONDS:.0f} s; white-box renders use the first "
            f"{wb_seconds:.0f} s."
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
                analyse = gr.Checkbox(
                    value=False,
                    label=f"Archetype analysis (loudest {ANALYSIS_SECONDS:.0f} s; adds a few "
                    "seconds per render, the first run is slow while the pitch tracker compiles)",
                )
                with gr.Row():
                    render_btn = gr.Button("Render", variant="primary")
                    reset_btn = gr.Button("Reset to defaults")
                    random_btn = gr.Button("Randomize")
                dry_out = gr.Audio(label="A: Dry", type="numpy", interactive=False)
                wet_out = gr.Audio(label="B: Wet", type="numpy", interactive=False)
                wb_out = gr.Audio(
                    label=f"C: White-box (circuit simulation, first {wb_seconds:.0f} s)",
                    type="numpy",
                    interactive=False,
                    visible=False,
                )
                wb_msg = gr.Markdown()
                with gr.Accordion("Archetype panel", open=True):
                    arche_md = gr.Markdown(PANEL_OFF)
                    arche_bar = gr.BarPlot(
                        value=_frame([], BAR_COLUMNS),
                        x="signal",
                        y="share",
                        color="component",
                        y_lim=[0, 1],
                        label="Archetype mix (sums to 1)",
                    )
                    harm_plot = gr.LinePlot(
                        value=_frame([], HARMONIC_COLUMNS),
                        x="harmonic",
                        y="dbc",
                        color="signal",
                        x_lim=[1, 10],
                        y_title="dBc (re. H1)",
                        label="Harmonics H1–H10",
                    )
                readout = gr.Markdown()
            with gr.Column(scale=1):
                pedal_dd = gr.Dropdown(
                    choices=pedal_choices(kb), value=MANUAL, label="Pedal (circuit preset)"
                )
                dial_groups, dials = [], []
                for pedal in kb.values():
                    with gr.Group(visible=False) as group:
                        dials.extend(gr.Slider(**dial_config(pot)) for pot in pedal.pots.values())
                    dial_groups.append(group)
                preset_md = gr.Markdown()
                derived = gr.State(None)  # {"pedal", "dials"} the sliders were derived from
                eq_note = gr.Markdown(EQ_LOCK_NOTE, visible=False)
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
                                    **private,
                                )

        render_btn.click(
            on_render,
            inputs=[
                audio_in,
                riff,
                match,
                pedal_dd,
                analyse,
                derived,
                *dials,
                *checkboxes,
                *sliders,
            ],
            outputs=[
                dry_out,
                wet_out,
                wb_out,
                wb_msg,
                readout,
                preset_md,
                arche_md,
                arche_bar,
                harm_plot,
            ],
        )
        stale = [wet_out, wb_msg, readout, arche_md, arche_bar, harm_plot]
        pedal_dd.change(
            on_pedal,
            inputs=[pedal_dd, *dials],
            outputs=[
                *dial_groups,
                preset_md,
                eq_note,
                wb_out,
                *stale,
                derived,
                *checkboxes,
                *sliders,
            ],
            **private,
        )
        for pid in kb:
            pedal_dials = dials[dial_slices[pid]]
            # release: mouse drags; change: keyboard / number-box edits and reset/randomize
            # (re-deriving there is idempotent). always_last: a move made while a derive is
            # running is queued instead of dropped.
            gr.on(
                triggers=[d.release for d in pedal_dials] + [d.change for d in pedal_dials],
                fn=make_on_dial(pid),
                inputs=[pedal_dd, *pedal_dials],
                outputs=[preset_md, derived, *checkboxes, *sliders],
                trigger_mode="always_last",
                show_progress="hidden",
                **private,
            )
        board_outputs = [*dials, *checkboxes, *sliders, preset_md, derived]
        reset_btn.click(on_reset, inputs=pedal_dd, outputs=board_outputs, **private)
        random_btn.click(on_randomize, inputs=pedal_dd, outputs=board_outputs, **private)

    return app
