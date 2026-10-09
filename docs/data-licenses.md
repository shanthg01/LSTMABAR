# Dataset license review (P0.7)

**Date:** 2026-10-09
**Scope:** audio datasets and capture libraries we might use for (a) clean DI electric-guitar inputs, (b) recordings of real pedals for the `test_real` sim-to-real set, and (c) neural captures (NAM / Proteus) for rendering realistic training audio.
**Project context assumed:** a solo, non-commercial side project. The code is open source on public GitHub. A Hugging Face Spaces demo may follow, plus possibly a public write-up. No revenue, ads, or paid tiers.

> **Not legal advice.** This is an engineering due-diligence note, written from the license fields and terms pages as they stood on the date above. Where it says "unclear", ask the rights holder before doing anything public.

## How to read the CC terms for our use

- **BY**: credit the source and link the license.
- **NC**: no use "primarily intended for commercial advantage or monetary compensation". A free, unmonetised side project and a free HF Space should qualify. Ads, paid tiers, or a later commercial pivot would not.
- **ND** (CC 4.0): you may make adapted material privately, but you may not **share** it. Rendering a dataset's audio through our pedals produces adapted material, so we cannot publish those renders (audio examples in a write-up, a released training set, demo presets built from them). Whether trained **model weights** count as "adapted material" is legally unsettled. Treat publishing weights trained on ND audio as a grey area.
- **SA**: anything adapted and shared must carry the same license.
- **No license stated** = all rights reserved by default. That rules out redistribution, and any use beyond what the authors explicitly invite is unclear.

## Summary table

| Dataset | Role | License (verified source) | OK for our use? | Key conditions |
|---|---|---|---|---|
| **Guitar-TECHS** (Pedroza et al. 2025) | DI input (also mic'd clean amp) | CC BY 4.0 (Zenodo + project site) | **Yes** | Attribution. 48 kHz / 32-bit float; resample to 44.1 kHz. |
| **EGFxSet** (Pedroza et al. 2022) | Real-pedal test set (also clean DI single notes) | CC BY 4.0 (Zenodo). mirdata lists CC BY-SA 4.0 (discrepancy) | **Yes** | Attribution. Cite 2 papers (ISMIR LBD 2022 + DAFx 2024). Single 5 s notes only. Reportedly noisy. |
| **pOD-set** (Dal Rì et al. 2025) | Real-pedal test set (27 overdrives × 6 gain × 6 tone grid) | CC BY-NC 4.0 (Zenodo) | **Yes w/ conditions** | Non-commercial. Input is a mixed-instrument file, not pure DI guitar. 45 GB. |
| **ToneTwist AFx** (Comunità et al.) | Real-pedal test set (paired dry/wet, ~40 devices) | CC BY-NC 4.0 per Zenodo record (2 records checked). Repo code: MIT | **Yes w/ conditions** | Non-commercial. Dry inputs include **IDMT-SMT-Guitar (CC BY-NC-ND)** and **YouTube bass recordings** (inherited-rights red flag). Internal eval only. Don't republish audio. |
| **IDMT-SMT-Guitar** (Fraunhofer IDMT) | DI input | CC BY-NC-ND 4.0 (Zenodo + IDMT page: "provided for evaluation purpose") | **Yes w/ conditions** (private training/eval only) | NC + ND: no sharing of renders or derived audio. Weights trained on it are a grey area. "Evaluation purpose" wording. |
| **IDMT-SMT-Audio-Effects** (Fraunhofer IDMT) | Effects test set (mostly not real pedals) | CC BY-NC-ND 4.0 (Zenodo + IDMT page) | **Yes w/ conditions** (private eval only) | As above. Effect chain not documented on Zenodo, so don't treat it as real-pedal audio. |
| **GuitarSet** (NYU MARL) | ~~DI input~~ (acoustic guitar) | CC BY 4.0 (Zenodo). GitHub code: MIT | **Yes** (license), but **poor fit** | Attribution. **Acoustic** guitar (mic + hexaphonic pickup), not electric DI. |
| **EGDB** (Chen et al. 2022) | DI input | **No license stated** (project page, paper) | **Unclear** | Google Drive download, no terms. Amp tones rendered with Guitar Rig 5 (commercial plugin). Email authors before use. |
| **GOAT** (Loth et al. 2025) | DI input | CC BY-NC 4.0 + "research purposes only", restricted (request) access on Zenodo | **Unclear → avoid for now** | Request-gated, research-only. Contains covers of popular songs (third-party copyright). |
| **GUITAR-FX-DIST** (Comunità et al. 2021) | Effects (digital distortion emulations) | Labeled CC BY 4.0 on Zenodo, but built from IDMT-SMT-Audio-Effects (CC BY-NC-ND) | **Unclear → avoid** | License conflict with source audio. Plugin emulations, not real pedals. |
| **TONE3000 (ex-ToneHunt) NAM captures** | Captures for rendering | **Per-tone** license (T3K or one of 7 CC variants), plus site ToS (eff. 2025-03-12) and API terms (eff. 2026-07-07) | **Yes w/ conditions** | Hand-pick a small set. No bots or bulk download ("substantial portion" banned). Never redistribute `.nam` files. Record each tone's license. Prefer T3K / CC BY / CC0. Avoid ND. |
| **GuitarML Proteus ToneLibrary** | Captures for rendering | **Not stated for model files.** Plugin code GPL-3.0. ToneLibrary repo has GPL-3.0 LICENSE (scope unclear). Website: "All rights reserved" | **Unclear** (low risk for private rendering) | Don't redistribute `.json` models. Email GuitarML to confirm. Used by Open-Amp (ICASSP 2025) for research. |
| **Neural Amp Modeler (software)** | Tooling for captures | MIT (code) | **Yes** | MIT notice if we vendor code. Captures are licensed separately (see TONE3000). |

## Per-dataset details

### Guitar-TECHS
- **URLs:** project <https://guitar-techs.github.io/>; data <https://zenodo.org/records/14963133>; paper arXiv:2501.03720 (ICASSP 2025).
- **License:** CC BY 4.0. The Zenodo license field and the project site both say "All data is licensed under a Creative Commons Attribution 4.0 License". **Confidence: high.**
- **Permits:** research, redistribution, derived renders, training and publishing weights, public demo, and commercial use, all with attribution.
- **Attribution:** Pedroza, Abreu, Corey, Roman, "Guitar-TECHS: An Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords and Scales Using a Diverse Array of Hardware", ICASSP 2025 / arXiv:2501.03720.
- **Contents:** about 5 h 12 min from 3 professional players. Covers techniques (single notes, palm mute, vibrato, harmonics, pinch harmonics, bends), chords, scales, and 12 musical excerpts. Signals: **DI** (split directly to the interface), mic'd amp (Orange CR-60 / CR-12, Yamaha B-15, flat EQ), plus egocentric and exocentric stereo mics. MIDI is synchronized.
- **Format:** WAV at 48 kHz, 32-bit float (paper). 9 zips, 4.1 GB total, open download from Zenodo.
- **Red flags:** none found. The players and amps differ per subset, so split by player.

### EGFxSet
- **URLs:** <https://zenodo.org/record/7044411>; <https://egfxset.github.io/>; ISMIR 2022 LBD paper <https://ccrma.stanford.edu/~iran/papers/Pedroza_et_al_ISMIR_2022.pdf>.
- **License:** **CC BY 4.0** per the Zenodo license field. **Confidence: high** for Zenodo. The mirdata quick-reference table lists **CC BY-SA 4.0**, and egfxset.github.io only says "full open-access rights". Zenodo is the primary source. If we ever redistribute derived audio, honouring SA as well costs nothing.
- **Permits:** everything with attribution (including SA if we take the conservative reading).
- **Attribution:** the Zenodo page asks for both citations. Pedroza, Meza, Roman, ISMIR LBD 2022 (EGFxSet), and Pedroza, Abreu, Corey, Roman, DAFx 2024.
- **Contents:** 8,970 files of 5 s each (12 h 28 min). Single notes on a Fender American Stratocaster: 138 notes × 5 pickup positions. **Clean** (instrument input of an Audient iD14) plus the same notes through **12 real units**: Boss BD-2 Blues Driver, Ibanez Mini Tube Screamer, ProCo RAT2, Boss CE-3 Chorus, MXR Phase 45, Mooer E-Lady flanger, Line 6 DL4 (digital delay, tape echo, sweep echo), and Orange CR-60 amp reverbs (plate, hall, spring). A metadata CSV records effect settings. Recordings are normalized.
- **Format:** 13 zips + CSV, 5.8 GB. Zenodo doesn't state the sample rate or bit depth (check on download). Also loadable via `mirdata`.
- **Red flags:** single notes only, no chords or phrases. The ToneTwist authors called it unsuitable for effects *modelling* because of background noise and artifacts. That matters less for a held-out test set, but listen first. Each pedal was captured at one setting (confirm against the CSV).

### pOD-set (Parametric Overdrive pedal dataset)
- **URL:** <https://zenodo.org/records/15389653>.
- **License:** CC BY-NC 4.0 (Zenodo license field). **Confidence: high.**
- **Permits:** non-commercial research, redistribution, renders, weights, and a free demo, with attribution. Nothing commercial.
- **Attribution:** Dal Rì, Stefani, Turchet, Conci, "Morphdrive", DAFx25, pp. 23–30. DOI 10.5281/zenodo.15389653.
- **Contents:** **27 real boutique overdrive pedals** (e.g. Boss Blues Driver, Fulltone OCD v1.3, Klon KTR, Walrus 385 MKII), each recorded across a 6 × 6 grid of gain and tone knob positions (0, 2, … 10), 36 combinations per pedal, ~98 h total. Chain latency is removed and polarity corrected. The input is about 6 min of mixed instrumental material from Yeh et al., plus sine sweeps. **Not pure DI guitar.**
- **Format:** WAV, 48 kHz, 24-bit, per-pedal zips with CSV. 45.1 GB in total, so download only the pedals we need.
- **Red flags:** NC. The rights in the Yeh et al. input material weren't checked (open question).

### ToneTwist AFx
- **URLs:** repo <https://github.com/mcomunita/tonetwist-afx-dataset> (MIT code). Per-device Zenodo records, e.g. <https://zenodo.org/records/10901357> (Custom Dynamic Fuzz) and <https://zenodo.org/records/10465454> (Ampeg Optocomp).
- **License:** CC BY-NC 4.0 on both records checked. **Confidence: medium.** Only 2 of about 40 records were checked, so check each record we download.
- **Attribution:** Comunità, Steinmetz, Phan, Reiss, ICASSP 2023 ("Modelling black-box audio effects with time-varying feature modulation"), and Comunità, Steinmetz, Reiss, NablAFx, arXiv:2502.11668.
- **Contents:** paired dry/wet audio for about 40 devices. Analog: amps/preamps, chorus, compressor, distortion, fuzz, overdrive, tremolo (e.g. EHX Big Muff, Fender Blues Jr, Blackstar HT1). Digital: plugins. Some devices are parametric. Wet outputs are 48 kHz / 32-bit float per the repo's contribution guide.
- **Dry inputs:** IDMT-SMT-GUITAR datasets 2 and 4, IDMT-SMT-Bass-Single-Track, NAM audio files, "private guitar data", and "YouTube bass recordings".
- **Red flags:** **inherited rights**. Part of the dry audio is CC BY-NC-ND (IDMT), and some comes from YouTube. Re-licensing those as CC BY-NC is questionable. Use for internal evaluation only and don't republish clips. The analog subset is still the best paired real-pedal data we found.

### IDMT-SMT-Guitar
- **URLs:** <https://zenodo.org/records/7544110>; <https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html>.
- **License:** CC BY-NC-ND 4.0 (Zenodo field). The IDMT page says it is "provided for evaluation purpose under the Creative Commons licence CC BY-NC-ND 4.0". **Confidence: high.**
- **Permits:** non-commercial use and verbatim redistribution with attribution. **No sharing of adapted material.** Private renders are fine, but published renders are not. Public weights trained on it are a grey area. The "evaluation purpose" wording sits next to the CC license without being part of it. Read conservatively, it suggests the authors intend evaluation, not training.
- **Attribution:** Kehling, Männchen, Eppler (Fraunhofer IDMT). Zenodo DOI 10.5281/zenodo.7544110. Reference paper: Kehling, Abeßer, Dittmar, Schuller, DAFx 2014.
- **Contents:** 7 guitars. Mostly fed directly from the guitar output (DI-like), with one condenser-mic recording. Subset 1: about 4,700 note events, techniques, licks (24-bit). Subset 2: 400 single notes. Subset 3: 5 short pieces. Subset 4: 64 genre pieces at 2 tempi. XML annotations.
- **Format:** mono WAV, 44.1 kHz, 24-bit (subset 1) or 16-bit (subsets 2–4). 1.3 GB zip, open download.
- **Red flags:** ND + NC + "evaluation purpose".

### IDMT-SMT-Audio-Effects
- **URLs:** <https://zenodo.org/records/7544032>; <https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html>.
- **License:** CC BY-NC-ND 4.0. Same "evaluation purpose" wording. **Confidence: high.**
- **Attribution:** Stein, Abeßer, Dittmar, Schuller, "Automatic Detection of Audio Effects in Guitar and Bass Recordings", AES 128th Convention, 2010. DOI 10.5281/zenodo.7544032.
- **Contents:** 55,044 files (about 30 h): 20,592 mono guitar notes, 20,592 mono bass notes, 13,860 polyphonic guitar sounds. 11 classes: no effect, feedback/slapback delay, reverb, chorus, flanger, phaser, tremolo, vibrato, distortion, overdrive. 2 guitars + 2 basses.
- **Format:** mono WAV, 44.1 kHz, 16-bit. 6.5 GB zip.
- **Red flags:** ND/NC. **Zenodo doesn't say how the effects were produced**, so don't count it as "real pedal" audio for `test_real` unless the AES paper confirms hardware. The "no effect" class is a usable DI-like source, under the same ND limits.

### GuitarSet
- **URLs:** <https://zenodo.org/records/3371780>; code <https://github.com/marl/GuitarSet> (MIT).
- **License:** CC BY 4.0 (Zenodo field, v1.1.0). **Confidence: high.** The mirdata table lists "MIT", which is the code repo's license, not the data's.
- **Attribution:** Xi, Bittner, Pauwels, Ye, Bello, "GuitarSet: A Dataset for Guitar Transcription", ISMIR 2018.
- **Contents:** 360 excerpts (mirdata count). **Acoustic guitar** recorded with a hexaphonic pickup and a microphone. Audio variants: hex original, hex debleeded, mono mic, mono pickup mix. JAMS annotations. 8.2 GB.
- **Sample rate:** not stated on Zenodo (not verified here).
- **Red flags:** none on licensing. **Wrong instrument for "clean electric DI"**: an acoustic body and pickup won't match electric pickups into a pedal. design.md lists it as a DI corpus, so that line should change. At most it's an out-of-distribution robustness check.

### EGDB (Electric Guitar DataBase)
- **URLs:** project <https://ss12f32v.github.io/Guitar-Transcription/> (Google Drive link); paper arXiv:2202.09907 (ICASSP 2022).
- **License:** **not stated.** The project page and paper only say "We make the dataset publicly available". **Confidence: high that no license is published** in these sources. The Drive folder itself wasn't inspected for a LICENSE file.
- **Attribution:** Chen, Hsiao, Hsieh, Jang, Yang, "Towards Automatic Transcription of Polyphonic Electric Guitar Music: A New Dataset and a Multi-Loss Transformer Model", ICASSP 2022.
- **Contents:** 240 tabs, 118 min of **DI** (Stratocaster-type with a hexaphonic pickup summed through a 6-out DI box), plus 5 amp renders from **Guitar Rig 5**: Mesa Mark V, Fender Twin, Marshall JCM2000, Roland JC120, Marshall Plexi. Six timbres in total.
- **Format:** not stated (check on download).
- **Red flags:** no license, so all rights are reserved by default. Distributed via Google Drive. The amp renders come from a commercial plugin, and we don't need them anyway.

### GOAT (Guitar On Audio and Tablature)
- **URLs:** <https://zenodo.org/records/15690894>; paper arXiv:2509.22655 (ISMIR 2025); code <https://github.com/JackJamesLoth/GOAT-Dataset>.
- **License:** CC BY-NC 4.0 (Zenodo field). Files are **restricted (request form)**, and the description says "for research purposes only and is not intended for use in any commercial product". **Confidence: high.**
- **Contents:** 5.9 h of unique DI from several guitars and players, 172 files, 44.1 kHz. Guitar Pro tabs, MIDI. Amp augmentation (29.5 h) uses about 7,000 public NAM profiles plus NeuralDSP Archetype Nolly IRs.
- **Red flags:** request-gated, research-only. The authors say they "do not exclusively own copyright for some content" (covers of popular songs). The renders use commercial IRs and third-party NAM captures. Not suitable for anything public.

### GUITAR-FX-DIST
- **URL:** <https://zenodo.org/records/4296040> (Mono Continuous; the other 3 subsets are separate records).
- **License:** labeled CC BY 4.0. Built from unprocessed IDMT-SMT-Audio-Effects recordings, which are **CC BY-NC-ND 4.0**. **Confidence in the label: high. Confidence that it's valid: low.**
- **Contents:** about 550k 2 s samples (~305 h) through 14 *digital emulations* of overdrive, distortion, and fuzz. 44.1 kHz / 16-bit WAV. Mono Continuous alone is 22.5 GB.
- **Red flags:** license conflict with the source audio. Plugins, not real pedals. Skip it.

### TONE3000 (formerly ToneHunt) — NAM captures
- **URLs:** ToS <https://www.tone3000.com/terms> (effective 2025-03-12; tonehunt.org redirects here); sharing policy <https://www.tone3000.com/policy>; API docs <https://www.tone3000.com/api>; API terms <https://www.tone3000.com/api/terms> (effective 2026-07-07). Example tone page with the T3K license: <https://www.tone3000.com/tones/rr-ac30-tb-75774>.
- **License:** **per tone**. The API `License` enum is `t3k`, `cc-by`, `cc-by-sa`, `cc-by-nc`, `cc-by-nc-sa`, `cc-by-nd`, `cc-by-nc-nd`, `cco` (presumably CC0; not defined on the page). The T3K text on tone pages reads: "Users may download and use the data file in software and publish the resulting outputs without royalties or restrictions. However, they may not upload, republish, or distribute the data file without the author's permission." **Confidence: high** for the T3K text and the enum.
- **Site ToS (verbatim points):**
  - "You may not use automated tools, scripts, bots, or other means to systematically download, scrape…"
  - "You may not collect or download all or a substantial portion of available tones for any purpose."
  - No "data mining, harvesting, or systematic extraction".
  - No access "that exceeds normal human browsing patterns".
  - No "package, bundle, or redistribute tones … through third-party platforms".
  - No competing services.
  - No selling or commercial distribution.
  - The privacy policy says bulk downloading is monitored. A 2025-09-25 changelog entry tightened the terms against "unauthorized model scraping and republishing".
- **AI/ML training:** **not mentioned** in the ToS, policy, API docs, or API terms. Not forbidden, not permitted.
- **API terms:** free tier for non-commercial products only. No bulk download, mirroring, or caching of the catalog. Creator and license metadata must be kept, with "Powered by TONE3000" shown. Downloads only on user request.
- **What this means for us:**
  - Rendering audio through a T3K capture produces "outputs", which T3K explicitly lets us publish. Training on those renders and publishing weights fits that reading.
  - The `.nam` file must never go into the repo, a HF Space, or a dataset release.
  - CC BY / CC0 captures are the cleanest. CC BY-NC is fine while we stay non-commercial. Avoid ND, since rendered audio is arguably adapted material.
  - Because of the ToS we have to **hand-pick a small set** (tens of captures, not hundreds). No scripted crawling.
- **Red flags:** mixed per-item licenses; anti-scraping ToS; the ML-training question is unaddressed. The sharing policy bans captures of current commercial plugins, but enforcement is after the fact, so a capture's provenance (real hardware vs plugin) is self-reported.

### GuitarML Proteus ToneLibrary
- **URLs:** <https://guitarml.com/tonelibrary/tonelib-pro.html>; <https://github.com/GuitarML/ToneLibrary> (GPL-3.0 LICENSE in repo); plugin <https://github.com/GuitarML/Proteus> (GPL-3.0).
- **License:** **not stated for the model files.** The ToneLibrary GitHub repo carries a GPL-3.0 LICENSE, but its README doesn't say the models are covered. The website footer says "© Untitled. All rights reserved." Models are community-contributed by email or PR. **Confidence: high that no explicit model license exists.**
- **Contents:** knob captures (models conditioned on one knob, e.g. TS-9 drive, MXR 78 distortion, Friedman SmallBox gain, Xotic BB gain) and snapshot captures (e.g. ProCo Rat, Boss MT-2, Little Big Muff, Xotic SP compressor, plus amps). JSON (Proteus/LSTM format), bulk `Proteus_Tone_Packs.zip` from a GitHub release. Open-Amp (Wright, Carson, Juvela, ICASSP 2025) used this pack: 59 amp + 101 pedal captures.
- **Red flags:** no explicit license. Contributor provenance unknown. **Knob-conditioned pedal captures are valuable for us**, since they give a real-pedal response across a parameter. Email to confirm before publishing anything trained on them.

### Neural Amp Modeler (software)
- **URL:** <https://github.com/sdatkinson/neural-amp-modeler> — MIT. TONE3000's FAQ also says the NAM architecture, trainer, and inference are MIT.
- The license of NAM's standard capture input file (`v3_0_0.wav`) wasn't verified. It only matters if we redistribute it.

## Recommendations

### P4 DI inputs (task 4.1)
1. **Primary: Guitar-TECHS DI tracks.** CC BY 4.0, real electric DI, techniques + chords + scales + music, 3 players. Resample 48 kHz → 44.1 kHz. Split by player.
2. **Supplement: EGFxSet `Clean.zip`.** CC BY 4.0, single notes across 5 Strat pickup positions. Good for note-level coverage and pickup diversity.
3. **Optional, private only: IDMT-SMT-Guitar.** CC BY-NC-ND. Use only for experiments whose renders and weights we don't publish, or keep it as an extra evaluation input. Don't ship renders from it. Before using it to train published weights, ask Fraunhofer (open question 1).
4. **Pending an answer: EGDB DI.** The best fit after Guitar-TECHS (118 min of real electric DI phrases), but don't use it until the authors confirm a license.
5. **Our own DI recordings** are always clean. Even 30–60 min recorded by the author would remove most of the risk and are worth planning in.
6. **Don't use GuitarSet as a DI source** (acoustic guitar). Update design.md accordingly.

### `test_real` set (task 4.4)
1. **EGFxSet** (CC BY 4.0) is the core: BD-2, TS Mini, RAT2, CE-3, Phase 45, E-Lady, DL4, amp reverbs. Every one is a real unit with a matching clean input. Listen for noise first.
2. **pOD-set** (CC BY-NC 4.0): 27 real overdrives with a **gain × tone knob grid**, which suits a parametric sim-to-real check on the drive and tone axes. Download only a handful of pedals.
3. **ToneTwist AFx analog subset** (CC BY-NC 4.0, inherited-rights caveat): paired real fuzz, distortion, overdrive, chorus, tremolo, compressor. Internal evaluation numbers only. Never publish its audio.
4. **IDMT-SMT-Audio-Effects** is **not** a real-pedal set until we confirm how its effects were made. At most a secondary private eval.

### Capture-based rendering (task 4.2, plan item 2.4 fallback)
1. **TONE3000 NAM captures, hand-curated.** Choose about 20–50 pedal captures by hand. Keep a manifest (`tone id, URL, creator, license, date`) **in the repo**, but keep the `.nam` files out (gitignore and a local cache). Prefer licenses in this order: CC0 / CC BY > T3K > CC BY-NC. Skip anything ND or SA, and anything whose page suggests it captures a plugin. Use only renders and weights publicly, which T3K explicitly allows for outputs.
2. **GuitarML Proteus knob captures:** use them privately now, and confirm with GuitarML before publishing weights trained on them.
3. In the HF Space, render only with our own white-box/grey-box sims (or captures whose license allows redistribution, e.g. CC BY/CC0, with attribution in the Space). Don't bundle T3K or unlicensed capture files in the Space.

### Avoid
- **GUITAR-FX-DIST** (license conflict, plugin effects).
- **GOAT** (request-gated, research-only, contains copyrighted song covers).
- **Scripted bulk download from TONE3000** (explicit ToS violation).
- **Publishing any audio rendered from IDMT sources**, including through ToneTwist dry inputs.
- **Redistributing any capture file** (`.nam`, Proteus `.json`) unless its license is CC BY/CC0.

## Open questions / follow-ups
1. **Fraunhofer IDMT:** does "provided for evaluation purpose" rule out training? Are model weights trained on IDMT-SMT-Guitar shareable under CC BY-NC-ND? (Contact via the IDMT dataset page.)
2. **EGDB authors (Yu-Hua Chen, f08946011@ntu.edu.tw / Yi-Hsuan Yang):** what license covers the DI tracks? Can we train on them and publish weights and a demo?
3. **EGFxSet:** confirm CC BY 4.0 (Zenodo) vs CC BY-SA 4.0 (mirdata index) with the authors or mirdata maintainers. Also check sample rate and bit depth on download.
4. **TONE3000 (support@tone3000.com):** is using a hand-picked set of captures to render ML training data, and publishing the resulting non-commercial model weights, acceptable under the ToS? Does "cco" mean CC0? Is there a way to filter by license in the UI?
5. **GuitarML (smartguitarml@gmail.com):** what license covers the ToneLibrary / Proteus Tone Pack model files? Does the repo's GPL-3.0 apply to them?
6. **IDMT-SMT-Audio-Effects:** were the effects hardware or software? (Check Stein et al., AES 2010.)
7. **pOD-set:** what are the rights in the Yeh et al. input material?
8. **ToneTwist AFx:** check the license on each Zenodo record we actually download, and whether the authors cleared the IDMT and YouTube dry inputs.
9. **GuitarSet:** verify the sample rate (not stated on Zenodo).
10. If the project ever goes commercial (ads, paid tier, sale), re-run this review. Every NC item and the TONE3000 API commercial terms change status.
