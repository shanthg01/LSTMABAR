Hand-written commentary on the generated tables above (kept in
`reports/greybox_fidelity.discussion.md`; re-render with
`lstmabar fidelity --from-json reports/greybox_fidelity.json`).

**Gate metric.** The gate uses the *masked* mean: every H1 term plus every dBc term that is
not floored on both sides. Decision 6 says harmonics that both models put in the noise
"don't count", and those terms are 0 by construction. Only 58–79 of 160 terms count per
TS808 setting and 80–86 per DS-1 setting, so the all-terms mean (still shown, secondary)
roughly halves the error and should not be quoted as the result.

**Verdict.**
- **DS-1 passes outright:** best-fit masked mean 1.13 dB (0.61–2.17 per setting).
- **TS808 misses the 3 dB threshold:** 3.48 dB (2.54–4.02; four of five settings are over).
  It passes only via the exit gate's documented-deviation clause, with the two causes below.
  The deviation has ceilings set from this measurement: masked mean ≤ 4.1 dB and every
  setting ≤ 4.5 dB, roughly 0.5 dB above the measured values. A future run beyond either
  ceiling FAILs instead of being excused.
  This is not a comfortable pass. The TS808 grey-box is a measurably worse match than the
  DS-1's.

**TS808 cause 1: the clean path (P2 decision 4).** Feedback clipping adds the input to the
clipped gain path. At 0 dBFS the clean 1 V exceeds the ~0.6 V clipped part, so every TS808
setting outputs ≈ -19.5 dBFS whatever the drive. `Drive` saturates instead, so one output
level cannot match both ends of the input range:
- The best fit's signed H1 error runs from +1.7 … +5.5 dB at -30 dBFS to -3.7 … -5.7 dB
  at 0 dBFS.
- The per-pitch max |H1 error| reaches 10.6 dB (gain max, 82 Hz at -30 dBFS: grey -22.4
  vs white -33.0 dBFS) and 8.7 dB (gain max, 659 Hz at 0 dBFS).
- The masked mean is highest at 0 dBFS in every setting (4.4–6.5 dB). There the clean
  component also dilutes the circuit's harmonics relative to H1 (H3 of 165 Hz: white
  -19.1…-21.9 dBc vs grey -11…-13 dBc).

**TS808 cause 2: upper odd harmonics of the lowest note.** The largest single terms (10–23
dB) are H5/H7/H9 (and H3 at -20 dBFS) of the 82 Hz tone at -20/-10/0 dBFS. There the
circuit has them at about -26…-50 dBc (H3 -26…-32, H5 -27…-31, H7 -37…-43, H9 -45…-50) and
the grey-box has them at -36 dBc down to the -60 dBc floor. In other words, the grey-box loses upper odd
harmonics on low notes; these are not even harmonics from the fitted asymmetry. A plausible
mechanism (a hypothesis, not tested here) is the TS808's frequency-dependent clipping gain:
- In the circuit, the gain leg's ~720 Hz corner gives an 82 Hz fundamental little gain,
  while the harmonics the feedback diodes generate still reach the output.
- In `Drive`, a pre-clip HPF strong enough to keep the 82 Hz fundamental out of saturation
  also keeps that note's shaper from generating them.

**Fix if this matters.** The decision-4 `clean_db` blend on `Drive` addresses cause 1 and
probably part of cause 2. It changes Drive's chain, so it needs an `lstmabar recover`
re-pass and a design §4.3 update. It is not done here.

**DS-1.** The DS-1's ~60 dB of pre-clip gain hard-clips every level, so its level
dependence is small. The worst setting is Dist min (2.17 dB; 3.0 dB masked at -30/-20
dBFS, fitted gain 37.7 dB), where the circuit is just entering clipping and the shunt
diode knee differs from Drive's shaper. Its worst terms are mixed in sign (H3–H9 with the
white-box at -34…-57 dBc, up to 9.7 dB). The white-box ignores the booster's own soft clipping (an ideal
linear transistor), so this result does not cover the booster's even-harmonic contribution.

**Derived (unfitted) vs best fit.**
- **Error gaps:** derived errors are 1.78 dB (DS-1) and 5.35 dB (TS808) masked, i.e.
  0.65 dB and 1.9 dB above the best fit. For TS808 the derivation is clearly off, not
  merely close: the derived H1 offset at -30 dBFS (+3.3 … +5.1 dB) is the same clean-path
  signature, and the derived dBc error is 5.7–6.8 dB.
- **The derivation is a valuable start:** the derived start won 9 of 10 fits. TS808
  random starts ended at 5–20 dB, so the fit landscape is strongly multimodal there.
- **Systematic differences** (see the parameter tables), candidates for P4's per-pedal
  calibration:
  - The derived shaper priors (`softness` 0.5 shunt / 0.2 feedback) are harder than most
    fitted values (0.06–0.17, except DS-1 Dist max at 0.71).
  - Four of five TS808 fits raise `eq.low_db` to +3…+7 dB (derived ≈ +0.6).

**Selection is in-sample by design.** The best start is chosen on the same clips it was
fitted to. The question is how close the grey-box family can get (an expressiveness bound),
not generalization, so the best-fit numbers are optimistic for unseen material by
construction.

**Fit objective (follow-up).** The fit loss is the all-terms L1, not the masked mean.
Both-floored terms already contribute zero gradient. Masking would only rescale each start's
loss by its count of counted terms, which changes as harmonics cross the floor, so the
fitted optimum is not expected to move much. Switching the loss to the masked objective is
left as a follow-up rather than refitting here.

**Known model differences between white-box and derivation** (documented in code):
- The DS-1 white-box loads the tone stack with the level pot (Premier Guitar's signal path).
  This is 0.2–0.8 dB lower than the derivation's R18 + buffer load at level 0.5.
- The DS-1 white-box omits coupling high-passes below 10 Hz that the derivation includes.
- Shared linear models: the TS808 white-box and the derivation share
  `derive.ts808_tone_stage`; the DS-1 booster and tone stack share `derive.ds1_booster` and
  `derive.lp_hp_blend_tone_stack`.

**Not covered.** No RAT white-box was built; it is optional and diagnostic-only per the
plan. The RAT has no capacitor at its diode node, so the one-capacitor shunt solver does not
apply directly. Big Muff and the Fuzz Faces have no white-box.

**Runtime.** This run took 1403 s, a bit over the ~20 min target, on a 10-thread laptop CPU
at 80–250 s per setting, almost all of it in the grey-box fit. An identical earlier run of
the same computation took 958 s, so machine load varies. Settings used to stay near budget:
- 22.05 kHz sample rate.
- Fitting on 0.2 s of each 0.5 s window with a 0.05 s grey-box pre-roll; the metric always
  uses the full window.
- 3 starts × 100 Adam steps. A 150-step, 4-start trial gave the same gain-mid result to
  within 0.01 dB (all-terms mean).
