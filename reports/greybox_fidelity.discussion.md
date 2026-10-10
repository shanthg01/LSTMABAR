Hand-written commentary on the generated tables above (kept in
`reports/greybox_fidelity.discussion.md`; re-render with
`lstmabar fidelity --from-json reports/greybox_fidelity.json`).

**Gate.** Both gated pedals pass with margin: one Drive + EQ3 parameter set per knob
setting reaches 0.30–1.16 dB (DS-1) and 1.25–1.62 dB (TS808) mean harmonic error across all
four pitches and all four input levels. No topology fix (`clean_db`, `pre_lpf_hz`) is needed
for the gate, so `Drive`'s ranges and chain are unchanged and `lstmabar recover` does not
need a re-pass.

**TS808 clean path (P2 decision 4).** The gap the plan predicted is visible in the per-level
tables, but it is not large enough to fail the mean. The signed H1 error of the best fit
runs from +1.7 … +5.5 dB at -30 dBFS to -3.7 … -5.7 dB at 0 dBFS: the grey-box is too loud
at low input and too quiet at full input. That is the signature of feedback clipping's
unity clean path: the white-box output keeps growing with the input (at 0 dBFS the clean
1 V exceeds the ~0.6 V clipped part, so every TS808 setting outputs ≈ -19.5 dBFS whatever
the drive), while `Drive` saturates, so a single level cannot match both ends and the fit
splits the difference. The harmonic (dBc) error also peaks at 0 dBFS (2.1–3.2 dB vs
0.3–1.8 dB below), because the clean component dilutes the harmonics relative to H1 in the
circuit but not in `Drive`. The H1 term is where TS808's error lives (best-fit H1 mean
2.1–4.3 dB vs 0.9–1.4 dB dBc). If later work needs the TS808's level dependence (e.g. the
demo's input-gain behaviour or P4 renders at hot levels), the `clean_db` blend of decision 4
is the fix; it would need `lstmabar recover` to re-pass and an update to design §4.3.

**DS-1.** The DS-1's ~60 dB of pre-clip gain puts every level above -30 dBFS into hard
clipping, so the level dependence is small and one parameter set fits well; most of the
remaining error is at -30/-20 dBFS on the Dist-min setting (37.7 dB fitted gain), where the
circuit is just entering clipping and the shunt diode knee differs from Drive's shaper.
Drive's gain clamps at 60 dB (the derivation asks for more; the fit sits at 59.5–59.8 dB),
which did not limit the fit. The white-box ignores the booster's own soft clipping (an
ideal linear transistor), so this result does not cover the booster's even-harmonic
contribution; that cascaded-stage gap remains documented, not measured.

**Derived vs best fit.** Derived (unfitted) errors are 0.5–1.6 dB (DS-1) and 2.1–2.7 dB
(TS808), i.e. the derivations land within ~1 dB of the best the grey-box family can do,
and the derived start was the winning start in 9 of 10 fits. The largest systematic
differences: the derived waveshaper prior (`softness` 0.5 for shunt, 0.2 for feedback
clipping) is harder than the fitted values (0.06–0.18 except DS-1 Dist max, 0.71); the
TS808 low shelf (four of five fits want +3 … +7 dB `eq.low_db` against ~+0.6 derived,
plausibly compensating the clean path's full-range content); and the TS808 derived H1 offset at
-30 dBFS (+3.3 … +5.1 dB), the same clean-path signature. These are candidates for the
per-pedal calibration the plan mentions for P4, not blockers.

**Large single-term maxima** (3–27 dB) come from individual terms out of 160 per setting;
the report does not attribute them to a harmonic, but a term where one model sits near the
-60 dBc floor and the other well above it (weak upper harmonics, or even harmonics the
grey-box adds through its small fitted asymmetry/bias) produces exactly this. They barely
move the means; per-term errors are a cheap addition to the JSON if they need diagnosing.

**Known model differences between white-box and derivation** (both documented in code):
the DS-1 white-box loads the tone stack with the level pot (Premier Guitar's signal path;
0.2–0.8 dB lower than the derivation's R18 + buffer load at level 0.5), and it omits
coupling high-passes below 10 Hz that the derivation includes. The TS808 white-box and the
derivation share `derive.ts808_tone_stage`; the DS-1 booster and tone stack share
`derive.ds1_booster` and `derive.lp_hp_blend_tone_stack`.

**Not covered.** No RAT white-box was built (optional and diagnostic-only per the plan): the
RAT has no capacitor at the diode node, so the one-capacitor shunt solver does not apply
directly, and its LM308 bandwidth limit is the more important gap anyway. Big Muff and the
Fuzz Faces have no white-box.

**Runtime.** 958 s for the full report on a 10-thread laptop CPU (≈80–160 s per pedal
setting, almost all of it the grey-box fit). To fit the ~20 min budget: 22.05 kHz,
fitting on 0.2 s of each 0.5 s analysis window with a 0.05 s grey-box pre-roll (the metric
always uses the full window), 3 starts × 100 Adam steps. In a 150-step, 4-start trial the
TS808/DS-1 gain-mid errors were 1.53/0.36 dB vs 1.54/0.35 dB here, so the shorter schedule
does not change the result.
