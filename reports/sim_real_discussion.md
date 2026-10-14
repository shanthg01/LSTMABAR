## Discussion

Hand-written for the 2026-10-14 run (`reports/sim_real_discussion.md`, appended verbatim by `lstmabar eval sim-real`). Every number below is from `reports/sim_real.json`. Re-check them if the report is regenerated from a new run.

**Settings mapping and knob direction.**
- *Mapping.* The `test_real` manifest stores EGFxSet settings as KB pot rotations in [0, 1] (`data.test_real.map_settings` / `knob_rotation`). `test_real.egfxset.knob_map` renames the CSV's TubeScreamer `overdrive` to the KB pot `drive`, and the ingest rejects any name that is not a KB pot. KB knobs are rotations too (`physics.kb.Pot` applies the taper), so the values pass through unchanged:
  - TubeScreamer -> `ts808` {drive 1.0, tone 0.5, level 0.5}
  - RAT -> `rat` {distortion 1.0, filter 0.5, volume 1.0}
- *One setting per unit.* EGFxSet recorded each unit at a single setting, and `aggregate_kb` asserts this. So knob direction cannot be checked within the KB groups. What the run does show:
  - **KB knob sweeps go the documented way.** `ts808` tone clockwise is brighter (centroid shift +0.13 / +0.20 / +0.83 oct at 0 / 0.5 / 1), and drive is brighter and lowers the crest factor. `rat` filter clockwise is darker (+2.01 / +1.52 / +0.45 oct), and distortion is brighter and lowers the crest factor.
  - **Per-item sign agreement.** Every KB render moves the crest factor the same way as the real pedal (1.00 for every engine and level). At −6 dBFS the centroid moves the same way for 0.97 (TS white-box), 0.93 (TS grey-box) and 0.98 (RAT grey-box) of items. This check discriminates weakly: the default-knob grey-box also scores 0.86 (TS) and 0.98 (RAT) at −6 dBFS.
  - **pOD-set `tubedreamer` grid.** This is a TS-derived pedal, not a TS808; it is internal (aggregate only), and the gain -> drive mapping is an assumption. Tone vs centroid shift agrees between real and KB (Spearman +0.66 real, +0.57 to +0.63 KB), and so does gain vs crest shift (−0.77 real, −0.71 to −0.95 KB). Gain vs centroid does not: −0.43 real, while the KB sign depends on the assumed input level (−0.24 at −24 dBFS, +0.13 to +0.52 at −18 to −6 dBFS).
  - **RAT.** `test_real` has no RAT setting grid, so `rat` direction is checked only on the KB side.

**KB items: the sim-to-real gap.**
- *MR-STFT.* At −12 and −6 dBFS, every render at the item settings has a lower MR-STFT than both references.
  - TS808 at −6 dBFS: 0.97 (white-box) and 1.34 (grey-box), vs 2.17 for the dry and 1.86 for the default-knob grey-box.
  - RAT (unseen pedal) grey-box at −6 dBFS: 1.53, vs 5.19 for the dry and 2.00 for default knobs.
  - This does not hold at every level: the TS grey-box at −24 dBFS (2.171) is no better than the dry (2.166).
  - The white-box has a lower MR-STFT than the grey-box at every sweep level.
- *Descriptors, TS.* The descriptors agree with MR-STFT for the TS. White-box at −6 dBFS: |Δ centroid| 0.10 octave and |Δ odd/even| 2.2 dB, vs 0.38 octave and 10.4 dB for the default-knob grey-box.
- *Descriptors, RAT.* For the RAT they do not agree. The grey-box centroid error grows with the input level (0.19, 0.17, 0.26, 0.35 octave from −24 to −6 dBFS). At −12 and −6 dBFS it is *worse* than the default-knob grey-box (0.21 octave at both levels). The odd/even error is about the same as default knobs (7.4–8.0 vs 7.4–9.9 dB).
- *HNR.* The HNR errors (2–5 dB) are about the size of the dry-vs-wet HNR difference (3.7 dB TS, 5.5 dB RAT), so they say little.

**Caveats.**
- **Input level.** The best sweep level is −6 dBFS, the hottest level tried, for 100 % of TS items and 93 % of RAT items, and MR-STFT is still falling at −6 dBFS. The real pedals were probably driven harder than −6 dBFS peak at `physics.volts_per_full_scale` = 1 V. So the true minimum is likely outside the sweep, and "best" is optimistic twice over: it is chosen per item on that item, and it is capped by the sweep range. A wider sweep (e.g. 0 / +6 dBFS) is a cheap follow-up.
- **Cross-pedal comparison.** MR-STFT magnitudes are not comparable across pedals: the RAT's dry reference is 5.19 against 2.17 for the TS, because the RAT changes the signal much more.
- **Alignment.** Dry and wet are assumed to be the same take, time-aligned (not verified). The metrics are magnitude-only, so a few ms of offset matters little, but a different take would inflate every number, the references included.
- **Sample.** Per group, n = 100 items, chosen deterministically (`eval.sim_real.max_items_per_group`) out of 689–690 per unit, with one setting per unit. The CIs are over notes, not over settings or pedal units.

**Family-level shifts (aggregate only).** The synthetic references are the v1 `train` KB renders of each family: overdrive = `ts808`, distortion = `ds1` (`rat` is held out, so it is not in `train`).
- *EGFxSet BluesDriver and RAT* are much more saturated than the synthetic renders of their families. BluesDriver (overdrive) crest shift is −15.6 dB vs −5.5 dB synthetic; RAT (distortion) is −16.6 dB vs −6.1 dB. Both units were recorded at maximum gain/distortion, while v1 samples knobs across their range.
- *Odd/even shift.* All three EGFxSet pedals have a far larger shift than the synthetic renders: +12.7 (TubeScreamer), +13.4 (RAT) and +15.1 dB (BluesDriver), vs +4.0 (overdrive) and +7.2 dB (distortion).
- *EGFxSet TubeScreamer.* It matches the synthetic overdrive crest shift (−5.7 vs −5.5 dB) but is brighter (+0.46 vs +0.20 octave).
- *pOD-set pedals.* They show small crest shifts (−1.7 to −4.0 dB). Four of five lower the centroid, with CIs excluding 0: kingoftone −1.08, ktr −0.80, ocd −1.17 and tubedreamer −0.62 octave. The bluesdriver's centroid shift is −0.11 [−0.26, 0.03], i.e. no clear change.
- *pOD-set comparability.* Their input is ~6 min of mixed material (10 s analysed), not DI guitar, and the outputs are normalized per pedal. So these numbers are not comparable to the DI-note shifts. They mainly show that the v1 synthetic overdrive distribution (centroid +0.20 octave) does not cover that input regime.

Text-driven methods are not evaluated on `test_real` (no instruction text, D11).
