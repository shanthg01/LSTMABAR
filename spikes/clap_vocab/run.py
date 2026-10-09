"""Spike A (plan P0.6): does CLAP understand guitar-tone vocabulary?

Renders a small set of known guitar tones from synthetic (Karplus-Strong) riffs, then checks
whether CLAP text-audio similarity ranks them the way a guitarist would:

1. Vocabulary: for each tone word, ROC-AUC of similarity separating the tones a guitarist
   would call that word (pre-registered in EXPECTED below, written before any run) from the
   rest. 0.5 = chance.
2. Sweeps: does similarity move monotonically along a drive-gain sweep ("distorted" up,
   "clean" down) and a low-pass cutoff sweep ("bright" up, "dark" down)? This is what a
   CLAP loss would need to provide useful gradients.

Usage: uv run --extra spikes python spikes/clap_vocab/run.py [--models ...] [--save-audio]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pedalboard as pb
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.stats import spearmanr
from transformers import ClapModel, ClapProcessor

SR = 44100
CLAP_SR = 48000
OUT = Path(__file__).parent / "results"

# --------------------------------------------------------------------------------------
# Synthetic guitar riffs (Karplus-Strong plucked string)
# --------------------------------------------------------------------------------------


def pluck(freq: float, dur: float, rng: np.random.Generator, decay: float = 0.996) -> np.ndarray:
    n = int(dur * SR)
    period = int(SR / freq)
    buf = rng.uniform(-1, 1, period)
    out = np.empty(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = decay * 0.5 * (buf[i % period] + buf[(i + 1) % period])
    return out


def midi(m: int) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def render_riff(notes: list[tuple[float, list[int], float]], total: float, seed: int) -> np.ndarray:
    """notes: (start_s, [midi pitches], dur_s)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(total * SR))
    for start, pitches, dur in notes:
        for p in pitches:
            s = pluck(midi(p), dur, rng)
            i = int(start * SR)
            y[i : i + len(s)] += s[: len(y) - i]
    return y / np.max(np.abs(y))


def riffs() -> dict[str, np.ndarray]:
    # E minor pentatonic single-note line (E2 = midi 40), eighth notes at ~100 bpm.
    line = [40, 43, 45, 47, 50, 52, 50, 47, 45, 43, 45, 47, 52, 55, 52, 50]
    lead = [(i * 0.3, [m + 12], 0.6) for i, m in enumerate(line)]
    # Power chords E5 - G5 - A5 - C5 (root, fifth, octave).
    chords = [(i * 1.2, [r, r + 7, r + 12], 1.2) for i, r in enumerate([40, 43, 45, 48])]
    chords += [(4.8 + c[0], c[1], c[2]) for c in chords[:1]]
    return {"lead": render_riff(lead, 6.0, seed=1), "chords": render_riff(chords, 6.0, seed=2)}


# --------------------------------------------------------------------------------------
# Tones
# --------------------------------------------------------------------------------------


def chain(x: np.ndarray, *plugins) -> np.ndarray:
    return pb.Pedalboard(list(plugins))(x.astype(np.float32), SR).astype(np.float64)


def drive(x, pre_hpf, gain, shaper, post_lpf, mids_db=0.0, mid_hz=800.0):
    y = chain(x, pb.HighpassFilter(pre_hpf))
    y = shaper(gain * y)
    plugins = [pb.LowpassFilter(post_lpf)]
    if mids_db:
        plugins.append(pb.PeakFilter(mid_hz, mids_db, 0.8))
    return chain(y, *plugins)


def soft(x):
    return np.tanh(x)


def hard(x):
    return np.clip(x, -0.4, 0.4) / 0.4


def asym(x, bias=0.35):
    return np.tanh(x + bias) - np.tanh(bias)


TONES = {
    "clean": lambda x: x,
    "dark": lambda x: chain(x, pb.LowpassFilter(900)),
    "bright": lambda x: chain(x, pb.HighShelfFilter(2500, 10), pb.HighpassFilter(200)),
    "soft_od": lambda x: drive(x, 720, 6, soft, 5000, mids_db=5),
    "hard_dist": lambda x: drive(x, 300, 60, hard, 6500),
    "fuzz": lambda x: drive(x, 80, 300, asym, 3500),
    "metal_scooped": lambda x: chain(
        drive(x, 400, 120, hard, 9000, mids_db=-12, mid_hz=700), pb.LowShelfFilter(150, 6)
    ),
    "compressed": lambda x: chain(x, pb.Compressor(-35, 10, 2, 150), pb.Gain(10)),
    "chorus": lambda x: chain(x, pb.Chorus(1.2, 0.5, 7, 0.3, 0.5)),
    "reverb": lambda x: chain(x, pb.Reverb(0.9, 0.3, 0.5, 0.6)),
    "lofi": lambda x: chain(x, pb.Bitcrush(5), pb.LowpassFilter(3000)),
}

# Pre-registered expectations: which tones a guitarist would describe with each word.
EXPECTED = {
    "clean": ["clean", "bright", "compressed", "chorus", "reverb"],
    "glassy": ["clean", "bright"],
    "sparkly": ["bright", "chorus"],
    "bright": ["bright", "metal_scooped"],
    "twangy": ["bright", "clean"],
    "dark": ["dark", "lofi"],
    "muddy": ["dark", "fuzz"],
    "warm": ["dark", "soft_od"],
    "smooth": ["dark", "soft_od", "compressed"],
    "overdriven": ["soft_od", "hard_dist"],
    "crunchy": ["soft_od", "hard_dist"],
    "distorted": ["hard_dist", "fuzz", "metal_scooped"],
    "heavy": ["metal_scooped", "hard_dist", "fuzz"],
    "aggressive": ["metal_scooped", "hard_dist"],
    "fuzzy": ["fuzz"],
    "fizzy": ["fuzz", "hard_dist"],
    "gritty": ["soft_od", "hard_dist", "fuzz"],
    "saturated": ["soft_od", "hard_dist", "fuzz", "metal_scooped"],
    "creamy": ["soft_od"],
    "mid-heavy": ["soft_od"],
    "scooped": ["metal_scooped"],
    "chugging metal": ["metal_scooped"],
    "compressed": ["compressed"],
    "squashed": ["compressed"],
    "lo-fi": ["lofi"],
    "shimmering": ["chorus"],
    "watery": ["chorus"],
    "spacious": ["reverb"],
    "ambient": ["reverb"],
    "Tube Screamer": ["soft_od"],
}

assert set(sum(EXPECTED.values(), [])) <= set(TONES), "EXPECTED names an unknown tone"

TEMPLATES = {
    "word": "{w}",
    "tone": "a {w} electric guitar tone",
    "sound": "the sound of a {w} electric guitar",
}

SWEEP_GAINS = [1, 3, 10, 30, 100, 300]
SWEEP_CUTOFFS = [700, 1400, 2800, 5600, 11000, 18000]


REF_RMS = 0.03  # low enough that no clip needs peak limiting, so RMS is exactly matched


def loudness_match(y: np.ndarray, ref_rms: float = REF_RMS) -> np.ndarray:
    y = y * ref_rms / (np.sqrt(np.mean(y**2)) + 1e-12)
    peak = np.max(np.abs(y))
    if peak > 1.0:
        raise ValueError(f"peak {peak:.2f} > 1 after RMS matching; lower REF_RMS")
    return y


SANITY_TEXTS = ["a pure sine wave tone", "white noise", "a plucked guitar string"]


def sanity_clips() -> list[np.ndarray]:
    """Pipeline control: CLAP should trivially match these to SANITY_TEXTS.

    Not loudness-matched: this only checks that the embedding pipeline works at all.
    """
    t = np.arange(int(6 * SR)) / SR
    sine = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.default_rng(0).normal(0, 0.1, len(t))
    guitar = riffs()["lead"] * 0.3
    return [sine, noise, guitar]


def build_clips() -> tuple[list[dict], list[dict]]:
    tone_clips, sweep_clips = [], []
    for riff_name, dry in riffs().items():
        for tone, fx in TONES.items():
            tone_clips.append({"riff": riff_name, "tone": tone, "audio": loudness_match(fx(dry))})
        for g in SWEEP_GAINS:
            y = loudness_match(drive(dry, 300, g, soft, 8000))
            sweep_clips.append({"riff": riff_name, "sweep": "gain", "value": g, "audio": y})
        for c in SWEEP_CUTOFFS:
            y = loudness_match(chain(dry, pb.LowpassFilter(c)))
            sweep_clips.append({"riff": riff_name, "sweep": "cutoff", "value": c, "audio": y})
    return tone_clips, sweep_clips


# --------------------------------------------------------------------------------------
# CLAP scoring
# --------------------------------------------------------------------------------------


def _features(out) -> torch.Tensor:
    return out if isinstance(out, torch.Tensor) else out.pooler_output


@torch.no_grad()
def embed(model_id: str, audios: list[np.ndarray], texts: list[str]):
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id).eval()
    a48 = [resample_poly(a, CLAP_SR, SR).astype(np.float32) for a in audios]
    a_in = processor.feature_extractor(a48, sampling_rate=CLAP_SR, return_tensors="pt")
    t_in = processor.tokenizer(texts, padding=True, return_tensors="pt")
    a = _features(model.get_audio_features(**a_in))
    t = _features(model.get_text_features(**t_in))
    a = torch.nn.functional.normalize(a, dim=-1)
    t = torch.nn.functional.normalize(t, dim=-1)
    return a.numpy(), t.numpy()


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def evaluate(model_id: str, tone_clips: list[dict], sweep_clips: list[dict]) -> dict:
    words = list(EXPECTED)
    texts = [TEMPLATES[t].format(w=w) for t in TEMPLATES for w in words]
    sweep_texts = [TEMPLATES["tone"].format(w=w) for w in ["distorted", "clean", "bright", "dark"]]
    clips = [c["audio"] for c in tone_clips + sweep_clips] + sanity_clips()
    a, t = embed(model_id, clips, texts + sweep_texts + SANITY_TEXTS)
    n_tone, n_sweep = len(tone_clips), len(sweep_clips)
    a_tone, a_sweep, a_sanity = (a[:n_tone], a[n_tone : n_tone + n_sweep], a[n_tone + n_sweep :])
    n_vocab, n_sw = len(texts), len(sweep_texts)
    t_vocab, t_sweep, t_sanity = t[:n_vocab], t[n_vocab : n_vocab + n_sw], t[n_vocab + n_sw :]

    sanity_sims = a_sanity @ t_sanity.T
    sanity = {
        "top1_acc": float(np.mean(np.argmax(sanity_sims, axis=1) == np.arange(3))),
        "sims": np.round(sanity_sims, 3).tolist(),
    }
    aa = a_tone @ a_tone.T
    off = aa[~np.eye(len(aa), dtype=bool)]
    audio_spread = {"mean_offdiag_cos": float(off.mean()), "min_offdiag_cos": float(off.min())}

    tone_names = np.array([c["tone"] for c in tone_clips])
    vocab = {}
    for ti, tname in enumerate(TEMPLATES):
        per_word = {}
        for wi, w in enumerate(words):
            sims = a_tone @ t_vocab[ti * len(words) + wi]
            mask = np.isin(tone_names, EXPECTED[w])
            top = tone_names[np.argmax(sims)]
            per_word[w] = {
                "auc": auc(sims[mask], sims[~mask]),
                "top1_tone": str(top),
                "top1_hit": bool(top in EXPECTED[w]),
            }
        aucs = [v["auc"] for v in per_word.values()]
        vocab[tname] = {
            "mean_auc": float(np.mean(aucs)),
            "top1_hit_rate": float(np.mean([v["top1_hit"] for v in per_word.values()])),
            "per_word": per_word,
        }

    # Per tone (template "tone"): which words rank highest, after z-scoring each word across
    # clips so that "hub" words that score high against everything don't dominate.
    tw = t_vocab[len(words) : 2 * len(words)]
    s = a_tone @ tw.T
    z = (s - s.mean(0)) / (s.std(0) + 1e-9)
    top_words = {}
    for tone in TONES:
        order = np.argsort(-z[tone_names == tone].mean(0))[:5]
        top_words[tone] = [words[i] for i in order]

    probe_idx = {w: i for i, w in enumerate(["distorted", "clean", "bright", "dark"])}
    sweep_defs = [
        ("gain", "distorted", None),
        ("gain", "clean", None),
        ("gain", "distorted", "clean"),
        ("cutoff", "bright", None),
        ("cutoff", "dark", None),
        ("cutoff", "bright", "dark"),
    ]
    sweeps = {}
    for sweep, pos, neg in sweep_defs:
        rhos = []
        for riff in ["lead", "chords"]:
            idx = [
                i for i, c in enumerate(sweep_clips) if c["sweep"] == sweep and c["riff"] == riff
            ]
            vals = [sweep_clips[i]["value"] for i in idx]
            sims = a_sweep[idx] @ t_sweep[probe_idx[pos]]
            if neg:
                sims = sims - a_sweep[idx] @ t_sweep[probe_idx[neg]]
            rhos.append(float(spearmanr(vals, sims).statistic))
        sweeps[f"{sweep}:{pos}" + (f"-{neg}" if neg else "")] = rhos
    return {
        "sanity": sanity,
        "audio_spread": audio_spread,
        "vocab": vocab,
        "top_words_per_tone": top_words,
        "sweeps": sweeps,
    }


def write_report(results: dict, path: Path) -> None:
    lines = ["# CLAP vocabulary spike: results", ""]
    lines.append("Pipeline sanity (sine / white noise / plucked string vs matching text) and")
    lines.append("audio-embedding spread across the 22 tone clips (cosine; 1.0 = identical).")
    lines.append("")
    lines.append("| Model | Sanity top-1 | Mean off-diag cos | Min off-diag cos |")
    lines.append("|---|---|---|---|")
    for m, r in results.items():
        sp = r["audio_spread"]
        lines.append(
            f"| {m} | {r['sanity']['top1_acc']:.2f} | {sp['mean_offdiag_cos']:.3f} "
            f"| {sp['min_offdiag_cos']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Mean ROC-AUC over 30 tone words (0.5 = chance) and top-1 hit rate, per prompt template."
    )
    lines.append("")
    lines.append("| Model | Template | Mean AUC | Top-1 hit |")
    lines.append("|---|---|---|---|")
    for m, r in results.items():
        for t, v in r["vocab"].items():
            lines.append(f"| {m} | {t} | {v['mean_auc']:.3f} | {v['top1_hit_rate']:.2f} |")
    lines.append("")
    lines.append("Sweeps: Spearman rho of similarity vs sweep value, per riff (lead, chords). ")
    lines.append("Expected sign: distorted +, clean −, bright +, dark −; contrasts (a-b) +.")
    lines.append("")
    lines.append("| Model | " + " | ".join(next(iter(results.values()))["sweeps"]) + " |")
    lines.append("|---|" + "---|" * len(next(iter(results.values()))["sweeps"]))
    for m, r in results.items():
        cells = [f"{a:+.2f} / {b:+.2f}" for a, b in r["sweeps"].values()]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    for m, r in results.items():
        lines += ["", f"## {m}: per-word AUC (template 'tone' for both models)", ""]
        lines.append("| Word | AUC | Top-1 tone | Hit |")
        lines.append("|---|---|---|---|")
        for w, v in r["vocab"]["tone"]["per_word"].items():
            hit = "yes" if v["top1_hit"] else "no"
            lines.append(f"| {w} | {v['auc']:.2f} | {v['top1_tone']} | {hit} |")
        lines += ["", f"## {m}: top-5 words per tone (z-scored per word)", ""]
        for tone, ws in r["top_words_per_tone"].items():
            lines.append(f"- **{tone}**: {', '.join(ws)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["laion/clap-htsat-unfused", "laion/larger_clap_general"]
    )
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tone_clips, sweep_clips = build_clips()
    n_tone, n_sweep = len(tone_clips), len(sweep_clips)
    print(f"rendered {n_tone} tone + {n_sweep} sweep clips in {time.time() - t0:.0f}s")
    if args.save_audio:
        (OUT / "audio").mkdir(exist_ok=True)
        for c in tone_clips:
            sf.write(OUT / "audio" / f"{c['riff']}_{c['tone']}.wav", c["audio"], SR)

    results = {}
    for m in args.models:
        t0 = time.time()
        results[m] = evaluate(m, tone_clips, sweep_clips)
        best = max(v["mean_auc"] for v in results[m]["vocab"].values())
        print(f"{m}: best mean AUC {best:.3f} ({time.time() - t0:.0f}s)")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    write_report(results, OUT / "report.md")
    print(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
