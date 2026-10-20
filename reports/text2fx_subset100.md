# P5 baselines

Generated 2026-10-20T09:53:00 at commit `34814e793c33d67d04e650f5ef6bff3dd99c8cd6` (dirty tree); run `eval_t2fx_l100/20261019-172245`; config `configs/eval/p5.yaml`.

- dataset sha256: `c4987b4ab834d3f7b4da87fcfc72d42fd995ad634f1939c7dfb7635291c93cfa`
- gold index sha256 (LF-normalized): `2fa0cfe5277523a26ac7dbb52a6c5b773c845bb3493f2f81caf314cdd5524e51`
- embedding cache files: 33 (sha256 in the JSON)

Every cell: mean [95 % bootstrap CI] (n items). Higher is better for 2AFC (chance 0.5) and gate accuracy/F1; lower is better for MR-STFT, L1 and descriptor errors.

> **Smoke run** (--limit 100): not the reported baseline results.

## Methods

| method | status | seeds | chosen (val) | selection criterion |
|---|---|---|---|---|
| text2fx | ok | 1 | - | none (no hyperparameters); harmonic descriptors off |
| noop | ok | 1 | - | none (no hyperparameters); harmonic descriptors off |
| mean | ok | 1 | - | none (no hyperparameters); harmonic descriptors off |
| knn_clap | ok | 1 | k=25 | runner.select_hparams: proxy.selection_score on val (with the val gold-proxy 2AFC); harmonic descriptors off |
| knn_bge | ok | 1 | k=25 | runner.select_hparams: proxy.selection_score on val (with the val gold-proxy 2AFC); harmonic descriptors off |
| p6_model | ok | 5 | - | none (no hyperparameters); harmonic descriptors off |

## Gold 2AFC, headline subset (per gold source)

Primary distance MR-STFT; LLM-written sources are reported separately ('on LLM-written test instructions'); `human` stays n/a until real friend items are paired.

### MR-STFT 2AFC (headline)

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| text2fx | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.601 [0.536, 0.664] (69) |
| noop | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.357 [0.278, 0.444] (69) |
| mean | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.616 [0.534, 0.693] (69) |
| knn_clap | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.686 [0.594, 0.773] (69) |
| knn_bge | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.710 [0.638, 0.780] (69) |
| p6_model | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.586 [0.498, 0.671] (69); seed SD 0.005 (5 seeds) |

### descriptor 2AFC (headline)

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| text2fx | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.597 [0.529, 0.662] (69) |
| noop | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.341 [0.266, 0.420] (69) |
| mean | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.514 [0.444, 0.585] (69) |
| knn_clap | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.558 [0.481, 0.633] (69) |
| knn_bge | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.597 [0.527, 0.667] (69) |
| p6_model | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.549 [0.476, 0.619] (69); seed SD 0.012 (5 seeds) |

### MR-STFT 2AFC by subset

**all**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| text2fx | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.580 [0.520, 0.637] (100) |
| noop | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.367 [0.295, 0.440] (100) |
| mean | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.633 [0.563, 0.700] (100) |
| knn_clap | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.675 [0.600, 0.747] (100) |
| knn_bge | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.707 [0.643, 0.770] (100) |
| p6_model | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.644 [0.572, 0.714] (100); seed SD 0.004 (5 seeds) |

**ambiguous**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| text2fx | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.511 [0.344, 0.667] (15) |
| noop | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.467 [0.300, 0.644] (15) |
| mean | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.711 [0.500, 0.900] (15) |
| knn_clap | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.678 [0.489, 0.844] (15) |
| knn_bge | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.700 [0.511, 0.867] (15) |
| p6_model | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.762 [0.573, 0.918] (15); seed SD 0.026 (5 seeds) |

**out_of_scope**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| text2fx | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.603 [0.423, 0.769] (13) |
| noop | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.385 [0.167, 0.615] (13) |
| mean | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.590 [0.397, 0.769] (13) |
| knn_clap | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.654 [0.449, 0.846] (13) |
| knn_bge | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.641 [0.449, 0.833] (13) |
| p6_model | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.767 [0.633, 0.892] (13); seed SD 0.025 (5 seeds) |

**unseen_pedal**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| text2fx | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.417 [0.000, 0.833] (4) |
| noop | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.125 [0.000, 0.375] (4) |
| mean | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.875 [0.625, 1.000] (4) |
| knn_clap | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.625 [0.250, 1.000] (4) |
| knn_bge | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.958 [0.875, 1.000] (4) |
| p6_model | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.825 [0.650, 1.000] (4); seed SD 0.075 (5 seeds) |

### MR-STFT 2AFC per distractor axis (headline)

| method | gold source | brightness | compression | gain | mids |
|---|---|---|---|---|---|
| text2fx | llm_draft | 0.558 [0.442, 0.675] (40) | 0.444 [0.222, 0.611] (6) | 0.687 [0.610, 0.764] (65) | 0.417 [0.222, 0.612] (12) |
| noop | llm_draft | 0.467 [0.317, 0.617] (40) | 0.167 [0.000, 0.389] (6) | 0.251 [0.154, 0.359] (65) | 0.611 [0.389, 0.833] (12) |
| mean | llm_draft | 0.475 [0.350, 0.600] (40) | 0.389 [0.167, 0.611] (6) | 0.749 [0.641, 0.846] (65) | 0.500 [0.333, 0.667] (12) |
| knn_clap | llm_draft | 0.675 [0.542, 0.800] (40) | 0.333 [0.111, 0.556] (6) | 0.769 [0.667, 0.867] (65) | 0.556 [0.333, 0.778] (12) |
| knn_bge | llm_draft | 0.567 [0.425, 0.700] (40) | 0.389 [0.222, 0.556] (6) | 0.821 [0.728, 0.903] (65) | 0.611 [0.389, 0.806] (12) |
| p6_model | llm_draft | 0.488 [0.357, 0.620] (40); seed SD 0.007 (5 seeds) | 0.333 [0.089, 0.600] (6); seed SD 0.056 (5 seeds) | 0.676 [0.566, 0.781] (65); seed SD 0.013 (5 seeds) | 0.478 [0.300, 0.661] (12); seed SD 0.050 (5 seeds) |

### Other gold metrics (headline)

| method | gold source | region L1 | MR-STFT to target |
|---|---|---|---|
| text2fx | human | n/a, 0 items | n/a, 0 items |
| text2fx | simulated_human | n/a, 0 items | n/a, 0 items |
| text2fx | llm_persona | n/a, 0 items | n/a, 0 items |
| text2fx | llm_draft | 0.283 [0.266, 0.300] (69) | 2.669 [2.505, 2.839] (69) |
| noop | human | n/a, 0 items | n/a, 0 items |
| noop | simulated_human | n/a, 0 items | n/a, 0 items |
| noop | llm_persona | n/a, 0 items | n/a, 0 items |
| noop | llm_draft | 0.251 [0.239, 0.264] (69) | 2.419 [2.187, 2.651] (69) |
| mean | human | n/a, 0 items | n/a, 0 items |
| mean | simulated_human | n/a, 0 items | n/a, 0 items |
| mean | llm_persona | n/a, 0 items | n/a, 0 items |
| mean | llm_draft | 0.098 [0.084, 0.112] (69) | 2.789 [2.416, 3.202] (69) |
| knn_clap | human | n/a, 0 items | n/a, 0 items |
| knn_clap | simulated_human | n/a, 0 items | n/a, 0 items |
| knn_clap | llm_persona | n/a, 0 items | n/a, 0 items |
| knn_clap | llm_draft | 0.108 [0.093, 0.123] (69) | 2.196 [1.932, 2.492] (69) |
| knn_bge | human | n/a, 0 items | n/a, 0 items |
| knn_bge | simulated_human | n/a, 0 items | n/a, 0 items |
| knn_bge | llm_persona | n/a, 0 items | n/a, 0 items |
| knn_bge | llm_draft | 0.099 [0.083, 0.115] (69) | 2.010 [1.763, 2.271] (69) |
| p6_model | human | n/a, 0 items | n/a, 0 items |
| p6_model | simulated_human | n/a, 0 items | n/a, 0 items |
| p6_model | llm_persona | n/a, 0 items | n/a, 0 items |
| p6_model | llm_draft | 0.126 [0.110, 0.142] (69); seed SD 0.003 (5 seeds) | 2.242 [2.015, 2.495] (69); seed SD 0.022 (5 seeds) |

## Synthetic `test`

| method | param_l1 | gate_acc | mrstft | centroid_oct | odd_even_db | hnr_db | archetype_l1 |
|---|---|---|---|---|---|---|---|
| text2fx | 0.373 [0.361, 0.385] (98) | 0.617 [0.578, 0.655] (100) | 2.944 [2.758, 3.137] (100) | 1.639 [1.456, 1.822] (100) | 5.565 [4.703, 6.502] (100) | 4.380 [3.855, 4.936] (100) | 0.508 [0.449, 0.571] (100) |
| noop | 0.227 [0.214, 0.240] (98) | 0.357 [0.327, 0.390] (100) | 2.500 [2.264, 2.734] (100) | 0.744 [0.596, 0.898] (100) | 5.198 [3.885, 6.601] (100) | 2.195 [1.792, 2.614] (100) | 0.322 [0.273, 0.375] (100) |
| mean | 0.178 [0.165, 0.191] (98) | 0.870 [0.820, 0.917] (100) | 2.729 [2.403, 3.068] (100) | 0.681 [0.562, 0.810] (100) | 3.687 [2.678, 4.878] (100) | 1.602 [1.290, 1.933] (100) | 0.242 [0.207, 0.280] (100) |
| knn_clap | 0.127 [0.108, 0.147] (98) | 0.937 [0.907, 0.963] (100) | 1.584 [1.361, 1.815] (100) | 0.576 [0.483, 0.672] (100) | 2.511 [1.938, 3.165] (100) | 1.205 [0.950, 1.476] (100) | 0.189 [0.157, 0.222] (100) |
| knn_bge | 0.112 [0.093, 0.132] (98) | 0.955 [0.932, 0.975] (100) | 1.269 [1.088, 1.457] (100) | 0.495 [0.410, 0.584] (100) | 1.989 [1.551, 2.464] (100) | 1.074 [0.862, 1.299] (100) | 0.165 [0.136, 0.194] (100) |
| p6_model | 0.126 [0.108, 0.144] (98); seed SD 4.99e-04 (5 seeds) | 0.945 [0.919, 0.968] (100); seed SD 0.001 (5 seeds) | 1.519 [1.308, 1.739] (100); seed SD 0.029 (5 seeds) | 0.563 [0.471, 0.660] (100); seed SD 0.006 (5 seeds) | 2.232 [1.792, 2.700] (100); seed SD 0.103 (5 seeds) | 1.192 [0.953, 1.444] (100); seed SD 0.011 (5 seeds) | 0.174 [0.145, 0.203] (100); seed SD 0.005 (5 seeds) |

Gate F1 (`test`, over queries)

| method | compressor | drive | eq | macro |
|---|---|---|---|---|
| text2fx | 0.293 (196) | 0.804 (196) | 0.926 (196) | 0.674 (196) |
| noop | 0.000 (196) | 0.000 (196) | 0.000 (196) | 0.000 (196) |
| mean | 0.000 (196) | 0.914 (196) | 0.963 (196) | 0.626 (196) |
| knn_clap | 0.512 (196) | 0.982 (196) | 0.971 (196) | 0.821 (196) |
| knn_bge | 0.558 (196) | 1.000 (196) | 0.978 (196) | 0.845 (196) |
| p6_model | 0.520 (196); seed SD 0.012 | 0.992 (196); seed SD 0.001 | 0.978 (196); seed SD 0.003 | 0.830 (196); seed SD 0.003 |

## Synthetic `test_unseen_pedal`

| method | param_l1 | gate_acc | mrstft | centroid_oct | odd_even_db | hnr_db | archetype_l1 |
|---|---|---|---|---|---|---|---|
| text2fx | 0.382 [0.367, 0.396] (100) | 0.633 [0.602, 0.665] (100) | 2.876 [2.718, 3.040] (100) | 1.645 [1.467, 1.826] (100) | 5.432 [4.723, 6.208] (100) | 3.959 [3.489, 4.466] (100) | 0.426 [0.373, 0.482] (100) |
| noop | 0.227 [0.218, 0.236] (100) | 0.333 [0.333, 0.333] (100) | 3.397 [3.192, 3.610] (100) | 0.879 [0.763, 1.002] (100) | 5.672 [4.362, 7.104] (100) | 3.403 [2.968, 3.849] (100) | 0.404 [0.350, 0.464] (100) |
| mean | 0.195 [0.189, 0.200] (100) | 1.000 [1.000, 1.000] (100) | 1.853 [1.729, 1.986] (100) | 0.615 [0.527, 0.706] (100) | 2.498 [2.101, 2.930] (100) | 2.331 [1.977, 2.692] (100) | 0.279 [0.236, 0.324] (100) |
| knn_clap | 0.201 [0.194, 0.208] (100) | 0.998 [0.995, 1.000] (100) | 1.755 [1.635, 1.886] (100) | 0.649 [0.564, 0.735] (100) | 2.665 [2.258, 3.070] (100) | 1.802 [1.530, 2.095] (100) | 0.269 [0.225, 0.315] (100) |
| knn_bge | 0.207 [0.201, 0.214] (100) | 1.000 [1.000, 1.000] (100) | 1.787 [1.695, 1.885] (100) | 0.679 [0.596, 0.766] (100) | 2.943 [2.507, 3.400] (100) | 1.854 [1.571, 2.150] (100) | 0.270 [0.230, 0.312] (100) |
| p6_model | 0.197 [0.191, 0.203] (100); seed SD 0.001 (5 seeds) | 0.987 [0.978, 0.994] (100); seed SD 0.004 (5 seeds) | 1.898 [1.755, 2.059] (100); seed SD 0.028 (5 seeds) | 0.721 [0.629, 0.813] (100); seed SD 0.028 (5 seeds) | 2.632 [2.234, 3.058] (100); seed SD 0.099 (5 seeds) | 1.848 [1.592, 2.115] (100); seed SD 0.064 (5 seeds) | 0.245 [0.208, 0.285] (100); seed SD 0.011 (5 seeds) |

Gate F1 (`test_unseen_pedal`, over queries)

| method | compressor | drive | eq | macro |
|---|---|---|---|---|
| text2fx | 0.000 (190) | 0.899 (190) | 0.965 (190) | 0.621 (190) |
| noop | n/a | 0.000 (190) | 0.000 (190) | 0.000 (190) |
| mean | n/a | 1.000 (190) | 1.000 (190) | 1.000 (190) |
| knn_clap | n/a | 1.000 (190) | 0.997 (190) | 0.999 (190) |
| knn_bge | n/a | 1.000 (190) | 1.000 (190) | 1.000 (190) |
| p6_model | 0.000 (190); seed SD 0.000 | 1.000 (190); seed SD 0.000 | 0.996 (190); seed SD 0.002 | 0.665 (190); seed SD 7.93e-04 |

## Val selection table (`val`; selection only, D8)

| method | param_l1 | gate_acc | mrstft | centroid_oct | odd_even_db | hnr_db | archetype_l1 |
|---|---|---|---|---|---|---|---|
| text2fx | - | - | - | - | - | - | - |
| noop | - | - | - | - | - | - | - |
| mean | - | - | - | - | - | - | - |
| knn_clap | - | - | - | - | - | - | - |
| knn_bge | - | - | - | - | - | - | - |
| p6_model | - | - | - | - | - | - | - |

## Paired comparisons

diff = A - B on paired items: mean [95 % paired bootstrap CI] (n); Wilcoxon signed-rank p; Cliff's delta; paired d (mean diff / SD).

| A | B | split / subset / source | metric | diff | p | Cliff's delta | d |
|---|---|---|---|---|---|---|---|
| knn_bge | noop | gold headline llm_draft | afc_mrstft | 0.353 [0.232, 0.469] (69) | 4.13e-06 | 0.515 | 0.690 |
| knn_bge | noop | test all | mrstft | -1.231 [-1.463, -0.992] (100) | 5.48e-13 | -0.594 | -1.010 |
| knn_bge | noop | test_unseen_pedal all | mrstft | -1.610 [-1.843, -1.375] (100) | 1.31e-15 | -0.826 | -1.334 |
| knn_clap | noop | gold headline llm_draft | afc_mrstft | 0.329 [0.196, 0.452] (69) | 5.71e-05 | 0.466 | 0.592 |
| knn_clap | noop | test all | mrstft | -0.916 [-1.182, -0.645] (100) | 1.54e-08 | -0.461 | -0.660 |
| knn_clap | noop | test_unseen_pedal all | mrstft | -1.642 [-1.869, -1.409] (100) | 5.02e-16 | -0.817 | -1.395 |
| mean | noop | gold headline llm_draft | afc_mrstft | 0.258 [0.106, 0.401] (69) | 0.002 | 0.392 | 0.409 |
| mean | noop | test all | mrstft | 0.230 [-0.201, 0.689] (100) | 0.498 | -0.019 | 0.101 |
| mean | noop | test_unseen_pedal all | mrstft | -1.544 [-1.728, -1.350] (100) | 4.88e-16 | -0.774 | -1.613 |
| p6_model | noop | gold headline llm_draft | afc_mrstft | 0.229 [0.117, 0.340] (69) | 8.25e-05 | 0.371 | 0.480 |
| p6_model | noop | test all | mrstft | -0.981 [-1.226, -0.738] (100) | 2.35e-10 | -0.497 | -0.774 |
| p6_model | noop | test_unseen_pedal all | mrstft | -1.499 [-1.753, -1.254] (100) | 1.42e-14 | -0.767 | -1.173 |
| text2fx | noop | gold headline llm_draft | afc_mrstft | 0.244 [0.116, 0.362] (69) | 0.001 | 0.410 | 0.460 |
| text2fx | noop | test all | mrstft | 0.445 [0.117, 0.779] (100) | 0.020 | 0.209 | 0.257 |
| text2fx | noop | test_unseen_pedal all | mrstft | -0.521 [-0.831, -0.207] (100) | 0.001 | -0.295 | -0.329 |
| knn_bge | mean | gold headline llm_draft | afc_mrstft | 0.094 [0.007, 0.186] (69) | 0.065 | 0.155 | 0.246 |
| knn_bge | mean | test all | mrstft | -1.460 [-1.817, -1.126] (100) | 8.36e-13 | -0.591 | -0.827 |
| knn_bge | mean | test_unseen_pedal all | mrstft | -0.066 [-0.185, 0.052] (100) | 0.273 | -0.005 | -0.107 |
| knn_clap | mean | gold headline llm_draft | afc_mrstft | 0.070 [-0.019, 0.159] (69) | 0.108 | 0.148 | 0.183 |
| knn_clap | mean | test all | mrstft | -1.146 [-1.453, -0.862] (100) | 3.03e-12 | -0.438 | -0.762 |
| knn_clap | mean | test_unseen_pedal all | mrstft | -0.098 [-0.219, 0.020] (100) | 0.120 | -0.088 | -0.159 |
| p6_model | mean | gold headline llm_draft | afc_mrstft | -0.029 [-0.132, 0.077] (69) | 0.559 | -0.035 | -0.065 |
| p6_model | mean | test all | mrstft | -1.211 [-1.552, -0.893] (100) | 5.56e-12 | -0.479 | -0.718 |
| p6_model | mean | test_unseen_pedal all | mrstft | 0.045 [-0.124, 0.210] (100) | 0.802 | 0.027 | 0.053 |
| text2fx | mean | gold headline llm_draft | afc_mrstft | -0.014 [-0.099, 0.072] (69) | 0.456 | -0.058 | -0.040 |
| text2fx | mean | test all | mrstft | 0.215 [-0.152, 0.564] (100) | 0.047 | 0.245 | 0.116 |
| text2fx | mean | test_unseen_pedal all | mrstft | 1.023 [0.821, 1.220] (100) | 4.45e-13 | 0.684 | 0.994 |
| p6_model | knn_bge | gold headline llm_draft | afc_mrstft | -0.124 [-0.215, -0.037] (69) | 0.014 | -0.184 | -0.326 |
| p6_model | knn_bge | test all | mrstft | 0.250 [0.117, 0.399] (100) | 7.79e-05 | 0.136 | 0.348 |
| p6_model | knn_bge | test_unseen_pedal all | mrstft | 0.110 [-0.044, 0.270] (100) | 0.608 | 0.033 | 0.137 |
| p6_model | knn_clap | gold headline llm_draft | afc_mrstft | -0.100 [-0.191, -0.007] (69) | 0.018 | -0.173 | -0.251 |
| p6_model | knn_clap | test all | mrstft | -0.065 [-0.206, 0.061] (100) | 0.858 | -0.026 | -0.094 |
| p6_model | knn_clap | test_unseen_pedal all | mrstft | 0.142 [0.033, 0.251] (100) | 0.007 | 0.122 | 0.256 |
| p6_model | text2fx | gold headline llm_draft | afc_mrstft | -0.015 [-0.114, 0.086] (69) | 0.790 | 0.025 | -0.035 |
| p6_model | text2fx | test all | mrstft | -1.425 [-1.724, -1.118] (100) | 9.22e-12 | -0.735 | -0.903 |
| p6_model | text2fx | test_unseen_pedal all | mrstft | -0.979 [-1.166, -0.781] (100) | 2.42e-14 | -0.702 | -1.004 |

## test_real: sim-to-real gap (D11)

Not run (`lstmabar eval sim-real` writes `reports/sim_real.json`).

## Notes

- v1 model (W266 submission, tag v1-w266): not comparable and not evaluated: it uses a different label scheme and the legacy DDSP engine, not the v2 free board (D4).
- No real human gold items yet (human column n/a). The gold headline is reported on each LLM-written set (simulated_human, llm_persona, llm_draft) separately, never pooled or relabeled, and claims are worded 'on LLM-written test instructions' (D9). The training paraphrases are also Claude-written (claude-haiku-5-5), so LLM-written test instructions are closer in style to the training captions than real people's; this weakens the template-circularity guard of design §5.
- Paired comparisons: Wilcoxon p-values are uncorrected for multiple comparisons (many method pairs x gold sources x splits); read them with the effect sizes and CIs, not as a family of significance tests.
- Headline gold subset = in scope, not ambiguous and not unseen-pedal; ambiguous, out-of-scope and unseen-pedal items are reported separately. Gain distractors also change the drive voicing (gain + voicing), and light-compression vs off contrasts are weak (targets file header).
- The secondary (descriptor) 2AFC distance averages the scaled centroid, odd/even and HNR errors with the unscaled archetype L1. The archetypes are a measured harmonic readout from the same analysis, so that term is not independent of the harmonic terms; MR-STFT is the primary 2AFC distance.
- CLAP score is a secondary metric only (spike A: mean word AUC ~0.6; only the drive-amount contrast is robust); it is not part of this table.
- test_unseen_pedal and test_real are supporting evidence, not text-adherence results.
- test_real: items marked internal_only are reported in aggregate only; no per-item numbers of those items appear in this report.
- Gate F1 is a split-level statistic over all queries (no item bootstrap: lo = hi = mean, n = queries). Every other cell is mean [95 % bootstrap CI over items] (n items).
- Failed renders (non-finite or silent output) count as a 2AFC loss on gold; on synthetic splits their audio metrics are NaN (dropped from n).
- Harmonic descriptors were on for every scored split (gold, test, test_unseen_pedal).
- Failed renders: none.
- Smoke run: at most 100 items per evaluated split (--limit); these numbers are not the reported baseline results.
