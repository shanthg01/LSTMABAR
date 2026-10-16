# P5 baselines

Generated 2026-10-15T18:37:58 at commit `e68770d9b87dd2f0a68b89aa1a3dc229604bab34`; run `eval_p5/20261014-204858`; config `configs/eval/p5.yaml`.

- dataset sha256: `c4987b4ab834d3f7b4da87fcfc72d42fd995ad634f1939c7dfb7635291c93cfa`
- gold index sha256 (LF-normalized): `2fa0cfe5277523a26ac7dbb52a6c5b773c845bb3493f2f81caf314cdd5524e51`
- embedding cache files: 33 (sha256 in the JSON)

Every cell: mean [95 % bootstrap CI] (n items). Higher is better for 2AFC (chance 0.5) and gate accuracy/F1; lower is better for MR-STFT, L1 and descriptor errors.

## Methods

| method | status | seeds | chosen (val) | selection criterion |
|---|---|---|---|---|
| noop | ok | 1 | - | none (no hyperparameters); harmonic descriptors off |
| mean | ok | 1 | - | none (no hyperparameters); harmonic descriptors off |
| knn_clap | ok | 1 | k=25 | runner.select_hparams: proxy.selection_score on val (with the val gold-proxy 2AFC); harmonic descriptors off |
| knn_bge | ok | 1 | k=25 | runner.select_hparams: proxy.selection_score on val (with the val gold-proxy 2AFC); harmonic descriptors off |

## Gold 2AFC, headline subset (per gold source)

Primary distance MR-STFT; LLM-written sources are reported separately ('on LLM-written test instructions'); `human` stays n/a until real friend items are paired.

### MR-STFT 2AFC (headline)

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| noop | n/a, 0 items | 0.412 [0.341, 0.485] (91) | 0.465 [0.393, 0.538] (111) | 0.363 [0.296, 0.430] (107) |
| mean | n/a, 0 items | 0.630 [0.557, 0.703] (91) | 0.565 [0.497, 0.634] (111) | 0.631 [0.565, 0.696] (107) |
| knn_clap | n/a, 0 items | 0.645 [0.571, 0.716] (91) | 0.545 [0.474, 0.616] (111) | 0.688 [0.620, 0.755] (107) |
| knn_bge | n/a, 0 items | 0.690 [0.619, 0.760] (91) | 0.646 [0.581, 0.707] (111) | 0.699 [0.637, 0.760] (107) |

### descriptor 2AFC (headline)

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| noop | n/a, 0 items | 0.370 [0.306, 0.434] (91) | 0.464 [0.396, 0.532] (111) | 0.346 [0.283, 0.408] (107) |
| mean | n/a, 0 items | 0.540 [0.480, 0.601] (91) | 0.520 [0.459, 0.581] (111) | 0.505 [0.447, 0.562] (107) |
| knn_clap | n/a, 0 items | 0.590 [0.527, 0.652] (91) | 0.495 [0.432, 0.559] (111) | 0.534 [0.472, 0.597] (107) |
| knn_bge | n/a, 0 items | 0.592 [0.527, 0.654] (91) | 0.590 [0.529, 0.649] (111) | 0.592 [0.530, 0.653] (107) |

### MR-STFT 2AFC by subset

**all**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| noop | n/a, 0 items | 0.476 [0.416, 0.537] (150) | 0.496 [0.430, 0.560] (150) | 0.372 [0.316, 0.432] (150) |
| mean | n/a, 0 items | 0.572 [0.510, 0.634] (150) | 0.530 [0.468, 0.592] (150) | 0.634 [0.577, 0.690] (150) |
| knn_clap | n/a, 0 items | 0.642 [0.581, 0.702] (150) | 0.539 [0.476, 0.603] (150) | 0.652 [0.592, 0.711] (150) |
| knn_bge | n/a, 0 items | 0.630 [0.569, 0.690] (150) | 0.621 [0.563, 0.678] (150) | 0.700 [0.647, 0.752] (150) |

**ambiguous**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| noop | n/a, 0 items | 0.533 [0.389, 0.678] (30) | 0.507 [0.340, 0.673] (25) | 0.413 [0.275, 0.565] (23) |
| mean | n/a, 0 items | 0.578 [0.428, 0.722] (30) | 0.533 [0.367, 0.700] (25) | 0.659 [0.493, 0.812] (23) |
| knn_clap | n/a, 0 items | 0.628 [0.478, 0.772] (30) | 0.653 [0.487, 0.807] (25) | 0.536 [0.370, 0.703] (23) |
| knn_bge | n/a, 0 items | 0.511 [0.367, 0.656] (30) | 0.560 [0.400, 0.713] (25) | 0.703 [0.551, 0.841] (23) |

**out_of_scope**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| noop | n/a, 0 items | 0.788 [0.621, 0.932] (22) | 0.769 [0.538, 1.000] (13) | 0.469 [0.260, 0.667] (16) |
| mean | n/a, 0 items | 0.227 [0.091, 0.379] (22) | 0.192 [0.000, 0.397] (13) | 0.562 [0.396, 0.729] (16) |
| knn_clap | n/a, 0 items | 0.606 [0.417, 0.788] (22) | 0.231 [0.051, 0.462] (13) | 0.604 [0.427, 0.781] (16) |
| knn_bge | n/a, 0 items | 0.515 [0.333, 0.689] (22) | 0.500 [0.231, 0.756] (13) | 0.698 [0.521, 0.865] (16) |

**unseen_pedal**

| method | human | simulated_human | llm_persona | llm_draft |
|---|---|---|---|---|
| noop | n/a, 0 items | 0.125 [0.000, 0.312] (8) | 0.000 [0.000, 0.000] (1) | 0.100 [0.000, 0.300] (5) |
| mean | n/a, 0 items | 0.875 [0.729, 0.979] (8) | 1.000 [1.000, 1.000] (1) | 0.900 [0.700, 1.000] (5) |
| knn_clap | n/a, 0 items | 0.750 [0.500, 0.958] (8) | 1.000 [1.000, 1.000] (1) | 0.633 [0.300, 0.933] (5) |
| knn_bge | n/a, 0 items | 0.750 [0.500, 0.938] (8) | 1.000 [1.000, 1.000] (1) | 0.767 [0.367, 1.000] (5) |

### MR-STFT 2AFC per distractor axis (headline)

| method | gold source | brightness | compression | gain | mids |
|---|---|---|---|---|---|
| noop | simulated_human | 0.756 [0.650, 0.854] (41) | 0.111 [0.037, 0.204] (18) | 0.351 [0.251, 0.459] (77) | 0.579 [0.404, 0.737] (19) |
| noop | llm_persona | 0.522 [0.413, 0.627] (67) | 0.375 [0.083, 0.667] (8) | 0.412 [0.321, 0.506] (106) | 0.690 [0.548, 0.833] (14) |
| noop | llm_draft | 0.492 [0.373, 0.610] (59) | 0.143 [0.000, 0.333] (7) | 0.224 [0.149, 0.307] (101) | 0.667 [0.514, 0.819] (24) |
| mean | simulated_human | 0.675 [0.561, 0.789] (41) | 0.537 [0.389, 0.685] (18) | 0.636 [0.532, 0.740] (77) | 0.281 [0.175, 0.386] (19) |
| mean | llm_persona | 0.557 [0.463, 0.652] (67) | 0.625 [0.458, 0.792] (8) | 0.588 [0.497, 0.676] (106) | 0.405 [0.262, 0.548] (14) |
| mean | llm_draft | 0.514 [0.412, 0.616] (59) | 0.381 [0.190, 0.571] (7) | 0.756 [0.670, 0.835] (101) | 0.486 [0.347, 0.625] (24) |
| knn_clap | simulated_human | 0.642 [0.520, 0.756] (41) | 0.463 [0.296, 0.630] (18) | 0.693 [0.593, 0.792] (77) | 0.386 [0.211, 0.561] (19) |
| knn_clap | llm_persona | 0.498 [0.393, 0.602] (67) | 0.583 [0.333, 0.833] (8) | 0.569 [0.478, 0.657] (106) | 0.643 [0.476, 0.810] (14) |
| knn_clap | llm_draft | 0.689 [0.582, 0.791] (59) | 0.333 [0.095, 0.571] (7) | 0.762 [0.680, 0.838] (101) | 0.556 [0.403, 0.694] (24) |
| knn_bge | simulated_human | 0.715 [0.585, 0.837] (41) | 0.426 [0.259, 0.593] (18) | 0.732 [0.632, 0.823] (77) | 0.544 [0.351, 0.737] (19) |
| knn_bge | llm_persona | 0.547 [0.448, 0.647] (67) | 0.583 [0.333, 0.792] (8) | 0.698 [0.616, 0.777] (106) | 0.595 [0.452, 0.738] (14) |
| knn_bge | llm_draft | 0.627 [0.514, 0.734] (59) | 0.381 [0.238, 0.524] (7) | 0.769 [0.686, 0.845] (101) | 0.583 [0.431, 0.736] (24) |

### Other gold metrics (headline)

| method | gold source | region L1 | MR-STFT to target |
|---|---|---|---|
| noop | human | n/a, 0 items | n/a, 0 items |
| noop | simulated_human | 0.246 [0.235, 0.258] (91) | 2.061 [1.827, 2.296] (91) |
| noop | llm_persona | 0.234 [0.226, 0.243] (111) | 2.182 [1.984, 2.389] (111) |
| noop | llm_draft | 0.249 [0.238, 0.259] (107) | 2.398 [2.202, 2.591] (107) |
| mean | human | n/a, 0 items | n/a, 0 items |
| mean | simulated_human | 0.124 [0.108, 0.141] (91) | 3.299 [2.917, 3.709] (91) |
| mean | llm_persona | 0.091 [0.081, 0.102] (111) | 3.327 [3.006, 3.656] (111) |
| mean | llm_draft | 0.097 [0.086, 0.108] (107) | 2.784 [2.486, 3.096] (107) |
| knn_clap | human | n/a, 0 items | n/a, 0 items |
| knn_clap | simulated_human | 0.138 [0.121, 0.156] (91) | 2.567 [2.230, 2.945] (91) |
| knn_clap | llm_persona | 0.119 [0.106, 0.133] (111) | 2.568 [2.313, 2.832] (111) |
| knn_clap | llm_draft | 0.110 [0.098, 0.122] (107) | 2.232 [2.009, 2.469] (107) |
| knn_bge | human | n/a, 0 items | n/a, 0 items |
| knn_bge | simulated_human | 0.124 [0.107, 0.143] (91) | 2.092 [1.774, 2.440] (91) |
| knn_bge | llm_persona | 0.112 [0.099, 0.126] (111) | 2.407 [2.126, 2.707] (111) |
| knn_bge | llm_draft | 0.106 [0.093, 0.121] (107) | 2.119 [1.896, 2.344] (107) |

## Synthetic `test`

| method | param_l1 | gate_acc | mrstft | centroid_oct | odd_even_db | hnr_db | archetype_l1 |
|---|---|---|---|---|---|---|---|
| noop | 0.231 [0.226, 0.236] (553) | 0.375 [0.362, 0.388] (560) | 2.573 [2.469, 2.680] (560) | 0.736 [0.679, 0.796] (560) | 4.942 [4.461, 5.449] (560) | 2.179 [1.975, 2.398] (560) | 0.313 [0.287, 0.341] (560) |
| mean | 0.186 [0.180, 0.192] (553) | 0.860 [0.838, 0.880] (560) | 2.779 [2.631, 2.932] (560) | 0.629 [0.586, 0.676] (560) | 3.452 [3.095, 3.837] (560) | 1.877 [1.697, 2.068] (560) | 0.290 [0.265, 0.317] (560) |
| knn_clap | 0.133 [0.125, 0.141] (553) | 0.917 [0.901, 0.931] (560) | 1.546 [1.453, 1.642] (560) | 0.487 [0.456, 0.520] (560) | 2.175 [1.962, 2.404] (560) | 1.414 [1.277, 1.560] (560) | 0.214 [0.194, 0.235] (560) |
| knn_bge | 0.121 [0.112, 0.129] (553) | 0.943 [0.931, 0.955] (560) | 1.263 [1.185, 1.344] (560) | 0.457 [0.424, 0.491] (560) | 1.975 [1.770, 2.197] (560) | 1.338 [1.212, 1.477] (560) | 0.201 [0.181, 0.222] (560) |

Gate F1 (`test`, over queries)

| method | compressor | drive | eq | macro |
|---|---|---|---|---|
| noop | 0.000 (1074) | 0.000 (1074) | 0.000 (1074) | 0.000 (1074) |
| mean | 0.000 (1074) | 0.911 (1074) | 0.940 (1074) | 0.617 (1074) |
| knn_clap | 0.396 (1074) | 0.981 (1074) | 0.948 (1074) | 0.775 (1074) |
| knn_bge | 0.602 (1074) | 0.997 (1074) | 0.960 (1074) | 0.853 (1074) |

## Synthetic `test_unseen_pedal`

| method | param_l1 | gate_acc | mrstft | centroid_oct | odd_even_db | hnr_db | archetype_l1 |
|---|---|---|---|---|---|---|---|
| noop | 0.224 [0.218, 0.229] (280) | 0.333 [0.333, 0.333] (280) | 3.339 [3.217, 3.459] (280) | 0.758 [0.694, 0.826] (280) | 6.524 [5.710, 7.413] (280) | 2.710 [2.446, 2.971] (280) | 0.444 [0.403, 0.487] (280) |
| mean | 0.193 [0.190, 0.196] (280) | 1.000 [1.000, 1.000] (280) | 1.850 [1.754, 1.957] (280) | 0.585 [0.535, 0.638] (280) | 2.461 [2.217, 2.734] (280) | 2.154 [1.941, 2.376] (280) | 0.305 [0.276, 0.336] (280) |
| knn_clap | 0.202 [0.198, 0.206] (280) | 0.997 [0.993, 0.999] (280) | 1.825 [1.737, 1.921] (280) | 0.691 [0.638, 0.747] (280) | 2.634 [2.370, 2.921] (280) | 2.007 [1.817, 2.210] (280) | 0.324 [0.292, 0.358] (280) |
| knn_bge | 0.209 [0.206, 0.213] (280) | 0.991 [0.983, 0.998] (280) | 1.904 [1.819, 1.995] (280) | 0.740 [0.683, 0.798] (280) | 2.870 [2.589, 3.180] (280) | 2.021 [1.828, 2.232] (280) | 0.342 [0.309, 0.377] (280) |

Gate F1 (`test_unseen_pedal`, over queries)

| method | compressor | drive | eq | macro |
|---|---|---|---|---|
| noop | n/a | 0.000 (531) | 0.000 (531) | 0.000 (531) |
| mean | n/a | 1.000 (531) | 1.000 (531) | 1.000 (531) |
| knn_clap | 0.000 (531) | 0.999 (531) | 0.999 (531) | 0.666 (531) |
| knn_bge | 0.000 (531) | 0.996 (531) | 0.992 (531) | 0.663 (531) |

## Val selection table (`val`; selection only, D8)

| method | param_l1 | gate_acc | mrstft | centroid_oct | odd_even_db | hnr_db | archetype_l1 | afc_descriptor | afc_descriptor/brightness | afc_descriptor/compression | afc_descriptor/gain | afc_descriptor/mids | afc_mrstft | afc_mrstft/brightness | afc_mrstft/compression | afc_mrstft/gain | afc_mrstft/mids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noop | 0.229 [0.226, 0.233] (1424) | 0.380 [0.372, 0.387] (1444) | 2.882 [2.796, 2.965] (1444) | 0.948 [0.904, 0.992] (1444) | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.376 [0.354, 0.397] (1444) | 0.552 [0.515, 0.590] (678) | 0.333 [0.237, 0.430] (93) | 0.313 [0.289, 0.338] (1317) | 0.436 [0.357, 0.521] (140) | 0.351 [0.330, 0.372] (1444) | 0.507 [0.471, 0.544] (678) | 0.161 [0.086, 0.237] (93) | 0.279 [0.255, 0.303] (1317) | 0.571 [0.493, 0.650] (140) |
| mean | 0.182 [0.179, 0.186] (1424) | 0.849 [0.835, 0.864] (1444) | 3.177 [3.052, 3.307] (1444) | 0.736 [0.710, 0.761] (1444) | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.589 [0.566, 0.611] (1444) | 0.482 [0.445, 0.519] (678) | 0.538 [0.441, 0.645] (93) | 0.620 [0.595, 0.646] (1317) | 0.321 [0.250, 0.400] (140) | 0.617 [0.596, 0.639] (1444) | 0.493 [0.454, 0.531] (678) | 0.419 [0.323, 0.516] (93) | 0.678 [0.653, 0.704] (1317) | 0.464 [0.379, 0.550] (140) |
| knn_clap | 0.130 [0.125, 0.136] (1424) | 0.916 [0.906, 0.925] (1444) | 1.591 [1.526, 1.657] (1444) | 0.449 [0.430, 0.467] (1444) | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.749 [0.730, 0.767] (1444) | 0.600 [0.563, 0.637] (678) | 0.495 [0.398, 0.591] (93) | 0.817 [0.796, 0.838] (1317) | 0.486 [0.407, 0.571] (140) | 0.817 [0.801, 0.832] (1444) | 0.593 [0.556, 0.630] (678) | 0.398 [0.301, 0.495] (93) | 0.923 [0.908, 0.937] (1317) | 0.557 [0.479, 0.636] (140) |
| knn_bge | 0.112 [0.107, 0.117] (1424) | 0.944 [0.937, 0.951] (1444) | 1.249 [1.205, 1.293] (1444) | 0.391 [0.373, 0.410] (1444) | n/a, 0 items | n/a, 0 items | n/a, 0 items | 0.783 [0.765, 0.801] (1444) | 0.664 [0.628, 0.699] (678) | 0.495 [0.387, 0.591] (93) | 0.845 [0.825, 0.864] (1317) | 0.536 [0.457, 0.621] (140) | 0.868 [0.854, 0.881] (1444) | 0.661 [0.625, 0.696] (678) | 0.570 [0.473, 0.667] (93) | 0.961 [0.951, 0.971] (1317) | 0.700 [0.621, 0.771] (140) |

## Paired comparisons

diff = A - B on paired items: mean [95 % paired bootstrap CI] (n); Wilcoxon signed-rank p; Cliff's delta; paired d (mean diff / SD).

| A | B | split / subset / source | metric | diff | p | Cliff's delta | d |
|---|---|---|---|---|---|---|---|
| knn_bge | noop | gold headline simulated_human | afc_mrstft | 0.278 [0.178, 0.379] (91) | 2.73e-06 | 0.424 | 0.561 |
| knn_bge | noop | gold headline llm_persona | afc_mrstft | 0.180 [0.086, 0.276] (111) | 5.73e-04 | 0.265 | 0.347 |
| knn_bge | noop | gold headline llm_draft | afc_mrstft | 0.336 [0.235, 0.436] (107) | 6.40e-08 | 0.491 | 0.638 |
| knn_bge | noop | test all | mrstft | -1.309 [-1.420, -1.198] (560) | 1.16e-66 | -0.606 | -0.962 |
| knn_bge | noop | test_unseen_pedal all | mrstft | -1.435 [-1.595, -1.271] (280) | 3.98e-33 | -0.752 | -1.035 |
| knn_clap | noop | gold headline simulated_human | afc_mrstft | 0.233 [0.119, 0.344] (91) | 1.97e-04 | 0.364 | 0.421 |
| knn_clap | noop | gold headline llm_persona | afc_mrstft | 0.080 [-0.027, 0.188] (111) | 0.162 | 0.119 | 0.135 |
| knn_clap | noop | gold headline llm_draft | afc_mrstft | 0.326 [0.220, 0.428] (107) | 4.13e-07 | 0.473 | 0.595 |
| knn_clap | noop | test all | mrstft | -1.026 [-1.153, -0.897] (560) | 2.08e-41 | -0.483 | -0.656 |
| knn_clap | noop | test_unseen_pedal all | mrstft | -1.514 [-1.671, -1.353] (280) | 4.55e-35 | -0.772 | -1.114 |
| mean | noop | gold headline simulated_human | afc_mrstft | 0.218 [0.088, 0.346] (91) | 0.002 | 0.338 | 0.344 |
| mean | noop | gold headline llm_persona | afc_mrstft | 0.099 [-0.030, 0.227] (111) | 0.203 | 0.150 | 0.142 |
| mean | noop | gold headline llm_draft | afc_mrstft | 0.268 [0.151, 0.383] (107) | 9.87e-05 | 0.409 | 0.426 |
| mean | noop | test all | mrstft | 0.206 [0.002, 0.414] (560) | 0.766 | -0.013 | 0.083 |
| mean | noop | test_unseen_pedal all | mrstft | -1.488 [-1.624, -1.346] (280) | 2.46e-35 | -0.743 | -1.242 |
| knn_bge | mean | gold headline simulated_human | afc_mrstft | 0.060 [-0.024, 0.145] (91) | 0.171 | 0.100 | 0.147 |
| knn_bge | mean | gold headline llm_persona | afc_mrstft | 0.081 [7.20e-18, 0.161] (111) | 0.051 | 0.117 | 0.185 |
| knn_bge | mean | gold headline llm_draft | afc_mrstft | 0.069 [-0.002, 0.139] (107) | 0.097 | 0.112 | 0.185 |
| knn_bge | mean | test all | mrstft | -1.516 [-1.669, -1.370] (560) | 6.85e-68 | -0.579 | -0.841 |
| knn_bge | mean | test_unseen_pedal all | mrstft | 0.053 [-0.030, 0.136] (280) | 0.268 | 0.106 | 0.077 |
| knn_clap | mean | gold headline simulated_human | afc_mrstft | 0.015 [-0.053, 0.081] (91) | 0.700 | 0.024 | 0.045 |
| knn_clap | mean | gold headline llm_persona | afc_mrstft | -0.020 [-0.095, 0.056] (111) | 0.769 | -0.028 | -0.048 |
| knn_clap | mean | gold headline llm_draft | afc_mrstft | 0.058 [-0.009, 0.123] (107) | 0.094 | 0.112 | 0.165 |
| knn_clap | mean | test all | mrstft | -1.233 [-1.366, -1.105] (560) | 8.95e-63 | -0.450 | -0.781 |
| knn_clap | mean | test_unseen_pedal all | mrstft | -0.025 [-0.103, 0.052] (280) | 0.493 | 0.014 | -0.039 |

## test_real: sim-to-real gap (D11)

From `reports/sim_real.json`, aggregate sections only (per-item sections omitted: kb_knob_sweeps, per_item). internal_only groups appear only as group aggregates. Text-driven methods are not evaluated on test_real in P5 (no instruction text, D11).

### Groups

| group | items | evaluated | internal_only |
|---|---|---|---|
| egfxset:BluesDriver | 690 | 100 | False |
| egfxset:RAT | 690 | 100 | False |
| egfxset:TubeScreamer | 689 | 100 | False |
| podset:bluesdriver | 144 | 36 | True |
| podset:kingoftone | 144 | 36 | True |
| podset:ktr | 144 | 36 | True |
| podset:ocd | 144 | 36 | True |
| podset:tubedreamer | 144 | 36 | True |

### KB render vs real wet (loudness-matched)

| group | KB pedal | engine | input dBFS | MR-STFT | |d centroid| oct | |d odd/even| dB | |d HNR| dB |
|---|---|---|---|---|---|---|---|
| egfxset:RAT | rat | dry | - | 5.185 [5.122, 5.250] (100) | 1.120 [1.017, 1.223] (100) | 14.246 [12.145, 16.409] (100) | 5.542 [4.799, 6.337] (100) |
| egfxset:RAT | rat | greybox | -24 | 2.358 [2.318, 2.401] (100) | 0.189 [0.160, 0.221] (100) | 7.352 [6.041, 8.722] (100) | 3.886 [3.382, 4.411] (100) |
| egfxset:RAT | rat | greybox | -18 | 1.948 [1.918, 1.980] (100) | 0.172 [0.144, 0.202] (100) | 7.851 [6.478, 9.312] (100) | 3.543 [3.092, 4.028] (100) |
| egfxset:RAT | rat | greybox | -12 | 1.667 [1.645, 1.687] (100) | 0.258 [0.235, 0.281] (100) | 7.910 [6.442, 9.480] (100) | 2.658 [2.270, 3.078] (100) |
| egfxset:RAT | rat | greybox | -6 | 1.529 [1.506, 1.552] (100) | 0.346 [0.331, 0.361] (100) | 7.989 [6.470, 9.594] (100) | 2.047 [1.743, 2.374] (100) |
| egfxset:RAT | rat | greybox | best | 1.526 [1.504, 1.549] (100) | 0.342 [0.327, 0.359] (100) | 7.976 [6.456, 9.584] (100) | 2.069 [1.764, 2.397] (100) |
| egfxset:RAT | rat | greybox_default | -24 | 3.519 [3.466, 3.574] (100) | 0.576 [0.508, 0.646] (100) | 9.870 [8.376, 11.432] (100) | 4.775 [4.209, 5.358] (100) |
| egfxset:RAT | rat | greybox_default | -18 | 2.933 [2.884, 2.982] (100) | 0.333 [0.282, 0.385] (100) | 8.031 [6.787, 9.386] (100) | 4.168 [3.651, 4.710] (100) |
| egfxset:RAT | rat | greybox_default | -12 | 2.413 [2.372, 2.457] (100) | 0.206 [0.174, 0.241] (100) | 7.356 [6.088, 8.708] (100) | 3.862 [3.359, 4.390] (100) |
| egfxset:RAT | rat | greybox_default | -6 | 1.997 [1.965, 2.029] (100) | 0.210 [0.179, 0.243] (100) | 7.690 [6.331, 9.133] (100) | 3.607 [3.142, 4.102] (100) |
| egfxset:RAT | rat | greybox_default | best | 1.997 [1.965, 2.029] (100) | 0.210 [0.179, 0.243] (100) | 7.690 [6.331, 9.133] (100) | 3.607 [3.142, 4.102] (100) |
| egfxset:TubeScreamer | ts808 | dry | - | 2.166 [2.110, 2.225] (100) | 0.488 [0.428, 0.548] (100) | 13.980 [11.566, 16.618] (100) | 3.711 [2.674, 5.178] (100) |
| egfxset:TubeScreamer | ts808 | whitebox | -24 | 1.833 [1.801, 1.867] (100) | 0.398 [0.365, 0.432] (100) | 10.569 [8.793, 12.565] (100) | 3.403 [2.426, 4.894] (100) |
| egfxset:TubeScreamer | ts808 | whitebox | -18 | 1.568 [1.541, 1.597] (100) | 0.282 [0.257, 0.308] (100) | 7.243 [5.867, 8.971] (100) | 3.036 [2.161, 4.457] (100) |
| egfxset:TubeScreamer | ts808 | whitebox | -12 | 1.277 [1.254, 1.303] (100) | 0.181 [0.164, 0.199] (100) | 4.235 [3.244, 5.675] (100) | 2.676 [1.854, 4.082] (100) |
| egfxset:TubeScreamer | ts808 | whitebox | -6 | 0.971 [0.951, 0.998] (100) | 0.104 [0.094, 0.114] (100) | 2.203 [1.471, 3.388] (100) | 2.206 [1.464, 3.518] (100) |
| egfxset:TubeScreamer | ts808 | whitebox | best | 0.971 [0.951, 0.998] (100) | 0.104 [0.094, 0.114] (100) | 2.203 [1.471, 3.388] (100) | 2.206 [1.464, 3.518] (100) |
| egfxset:TubeScreamer | ts808 | greybox | -24 | 2.171 [2.136, 2.207] (100) | 0.616 [0.576, 0.654] (100) | 12.683 [10.660, 14.852] (100) | 5.101 [4.155, 6.519] (100) |
| egfxset:TubeScreamer | ts808 | greybox | -18 | 1.915 [1.886, 1.945] (100) | 0.443 [0.411, 0.474] (100) | 9.188 [7.643, 10.956] (100) | 4.830 [3.927, 6.212] (100) |
| egfxset:TubeScreamer | ts808 | greybox | -12 | 1.624 [1.601, 1.648] (100) | 0.280 [0.258, 0.302] (100) | 5.259 [4.179, 6.727] (100) | 4.423 [3.553, 5.793] (100) |
| egfxset:TubeScreamer | ts808 | greybox | -6 | 1.335 [1.315, 1.359] (100) | 0.143 [0.126, 0.160] (100) | 3.157 [2.461, 4.112] (100) | 4.101 [3.262, 5.407] (100) |
| egfxset:TubeScreamer | ts808 | greybox | best | 1.335 [1.315, 1.359] (100) | 0.143 [0.126, 0.160] (100) | 3.157 [2.461, 4.112] (100) | 4.101 [3.262, 5.407] (100) |
| egfxset:TubeScreamer | ts808 | greybox_default | -24 | 2.471 [2.427, 2.517] (100) | 0.861 [0.812, 0.909] (100) | 14.882 [12.366, 17.606] (100) | 5.197 [4.173, 6.675] (100) |
| egfxset:TubeScreamer | ts808 | greybox_default | -18 | 2.316 [2.277, 2.357] (100) | 0.752 [0.704, 0.800] (100) | 14.635 [12.217, 17.244] (100) | 5.147 [4.149, 6.612] (100) |
| egfxset:TubeScreamer | ts808 | greybox_default | -12 | 2.116 [2.082, 2.152] (100) | 0.583 [0.539, 0.626] (100) | 13.268 [11.136, 15.584] (100) | 5.125 [4.157, 6.566] (100) |
| egfxset:TubeScreamer | ts808 | greybox_default | -6 | 1.862 [1.832, 1.892] (100) | 0.383 [0.347, 0.418] (100) | 10.444 [8.731, 12.366] (100) | 4.886 [3.949, 6.302] (100) |
| egfxset:TubeScreamer | ts808 | greybox_default | best | 1.862 [1.832, 1.892] (100) | 0.383 [0.347, 0.418] (100) | 10.444 [8.731, 12.366] (100) | 4.886 [3.949, 6.302] (100) |

### Family-level descriptor shifts (wet - dry): real vs synthetic family

| group | family | term | real | synthetic family |
|---|---|---|---|---|
| egfxset:BluesDriver | overdrive | centroid_oct | 1.732 [1.630, 1.832] (100) | 0.202 [0.192, 0.212] (3388) |
| egfxset:BluesDriver | overdrive | odd_even_db | 15.086 [12.010, 18.236] (100) | 3.991 [3.702, 4.277] (3369) |
| egfxset:BluesDriver | overdrive | hnr_db | -6.329 [-7.110, -5.558] (100) | 0.490 [0.364, 0.614] (3369) |
| egfxset:BluesDriver | overdrive | crest_db | -15.633 [-16.241, -15.014] (100) | -5.527 [-5.622, -5.433] (3388) |
| egfxset:RAT | distortion | centroid_oct | 1.118 [1.014, 1.220] (100) | 2.047 [2.021, 2.074] (3394) |
| egfxset:RAT | distortion | odd_even_db | 13.373 [11.054, 15.713] (100) | 7.161 [6.757, 7.585] (3360) |
| egfxset:RAT | distortion | hnr_db | -5.476 [-6.301, -4.692] (100) | -3.864 [-4.065, -3.669] (3360) |
| egfxset:RAT | distortion | crest_db | -16.595 [-17.206, -16.017] (100) | -6.103 [-6.280, -5.925] (3394) |
| egfxset:TubeScreamer | overdrive | centroid_oct | 0.461 [0.393, 0.527] (100) | 0.202 [0.192, 0.212] (3388) |
| egfxset:TubeScreamer | overdrive | odd_even_db | 12.660 [9.901, 15.512] (100) | 3.991 [3.702, 4.277] (3369) |
| egfxset:TubeScreamer | overdrive | hnr_db | -2.976 [-4.487, -1.842] (100) | 0.490 [0.364, 0.614] (3369) |
| egfxset:TubeScreamer | overdrive | crest_db | -5.714 [-6.132, -5.309] (100) | -5.527 [-5.622, -5.433] (3388) |
| podset:bluesdriver | overdrive | centroid_oct | -0.114 [-0.261, 0.027] (36) | 0.202 [0.192, 0.212] (3388) |
| podset:bluesdriver | overdrive | odd_even_db | -0.094 [-0.386, 0.217] (36) | 3.991 [3.702, 4.277] (3369) |
| podset:bluesdriver | overdrive | hnr_db | -0.483 [-0.716, -0.246] (36) | 0.490 [0.364, 0.614] (3369) |
| podset:bluesdriver | overdrive | crest_db | -2.309 [-3.422, -1.154] (36) | -5.527 [-5.622, -5.433] (3388) |
| podset:kingoftone | overdrive | centroid_oct | -1.083 [-1.226, -0.940] (36) | 0.202 [0.192, 0.212] (3388) |
| podset:kingoftone | overdrive | odd_even_db | -1.065 [-1.425, -0.742] (36) | 3.991 [3.702, 4.277] (3369) |
| podset:kingoftone | overdrive | hnr_db | 0.940 [0.572, 1.308] (36) | 0.490 [0.364, 0.614] (3369) |
| podset:kingoftone | overdrive | crest_db | -1.725 [-2.061, -1.378] (36) | -5.527 [-5.622, -5.433] (3388) |
| podset:ktr | overdrive | centroid_oct | -0.800 [-0.946, -0.645] (36) | 0.202 [0.192, 0.212] (3388) |
| podset:ktr | overdrive | odd_even_db | -0.339 [-0.538, -0.143] (36) | 3.991 [3.702, 4.277] (3369) |
| podset:ktr | overdrive | hnr_db | 0.668 [0.275, 1.062] (36) | 0.490 [0.364, 0.614] (3369) |
| podset:ktr | overdrive | crest_db | -2.255 [-3.128, -1.331] (36) | -5.527 [-5.622, -5.433] (3388) |
| podset:ocd | overdrive | centroid_oct | -1.169 [-1.320, -1.024] (36) | 0.202 [0.192, 0.212] (3388) |
| podset:ocd | overdrive | odd_even_db | -0.510 [-0.664, -0.354] (36) | 3.991 [3.702, 4.277] (3369) |
| podset:ocd | overdrive | hnr_db | 2.066 [1.765, 2.370] (36) | 0.490 [0.364, 0.614] (3369) |
| podset:ocd | overdrive | crest_db | -3.984 [-4.921, -3.078] (36) | -5.527 [-5.622, -5.433] (3388) |
| podset:tubedreamer | overdrive | centroid_oct | -0.624 [-0.682, -0.563] (36) | 0.202 [0.192, 0.212] (3388) |
| podset:tubedreamer | overdrive | odd_even_db | -1.230 [-1.355, -1.111] (36) | 3.991 [3.702, 4.277] (3369) |
| podset:tubedreamer | overdrive | hnr_db | 0.394 [0.189, 0.587] (36) | 0.490 [0.364, 0.614] (3369) |
| podset:tubedreamer | overdrive | crest_db | -1.813 [-2.331, -1.271] (36) | -5.527 [-5.622, -5.433] (3388) |

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
- Harmonic descriptors (odd/even, HNR, archetype terms of predicted renders) were OFF for: val (eval.metrics.harmonic_descriptors=false, eval.select_harmonic=false for val selection); those terms are n/a there and the descriptor 2AFC uses the centroid (and archetype where available) only. They were on for: gold, test, test_unseen_pedal.
- Failed renders: none.
