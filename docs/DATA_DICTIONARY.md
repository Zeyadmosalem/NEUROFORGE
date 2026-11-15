# NEUROFORGE — Data dictionary and provenance

## 1. Training file

`data/generated/attempts.csv` is **synthetic simulation data**, produced by `python -m data.make_data --seed 42`. It does not contain real humans' outcomes. Each row is one stochastic completion outcome for one simulated player profile on one search-verified CSP-generated level.

| Column | Type / units | Provenance | Meaning |
|---|---|---|---|
| `persona_id` | int | generated | Identifies simulated player; used for group-disjoint split |
| `level_id` | int | generated | ID of physics-verified level template; may recur across personas |
| `data_origin` | string | generated constant | `synthetic_simulation`, never human data |
| `latent_skill_simulator_only` | float [0,1] | sampled | Private simulated skill used to generate labels. **Never a model input** |
| `p_win_simulator_only` | float [0,1] | calculated | Latent probability used for simulation. **Never a model input** |
| `recent_win_rate` | float [0,1] | generated noisy observation | Simulated history of previous attempts, not current label |
| `jump_accuracy` | float [0,1] | generated noisy observation | Proxy for proportion of jumps safely landed |
| `deaths_per_attempt` | float | generated noisy observation | Mean recent deaths per attempt |
| `recent_progress` | float [0,1] | generated noisy observation | Mean recent level progress |
| `recent_time` | seconds | generated noisy observation | Mean recent attempt duration |
| `length` | tiles | derived from CSP-generated map | Tile map width |
| `gap_tiles` | tiles | derived | Number of missing ground cells at row `height-2` |
| `max_gap` | tiles | derived | Longest consecutive run of missing ground cells |
| `hazards` | count | derived | Spikes `^` in level |
| `coins` | count | derived | Coins `o` in level |
| `platforms` | count | derived | Elevated tiles with empty space below |
| `difficulty` | float [0,1] | derived | Realized level difficulty: sum of the chosen chunks' difficulty scores divided by `3.4 × (chunks − 2)`, capped at 1 (`csp/generator.py`). It is not the requested CSP target and not a measure of human-perceived difficulty |
| `label_win` | binary | **synthetic** stochastic Bernoulli draw | Outcome used as NN target |

`Level.features()` also reports `moving_platforms`, `moving_enemies` and `timed_gates`. These three counts are **not** columns of `attempts.csv` and are not NN inputs; dynamic objects reach the model only through the `difficulty` score of the chunks that contain them.

The simulator uses hidden `latent_skill_simulator_only`, level challenge, an interaction with jump accuracy, and normally distributed stochastic noise to create a completion probability. There is no claim that this formula reproduces human gameplay. Model inputs are ONLY the 12 columns in `FEATURES`; no simulator-only columns enter training. Player splits: shuffle unique persona IDs with the specified seed; 70% train, 15% validation, 15% test. Scalers are fit on train only. Details in `nn/train.py`.

## 2. Real external level corpus (downloaded)

`python -m data.make_data --seed 42` also downloads four **real geometry** text maps from [TheVGLC](https://github.com/TheVGLC/TheVGLC) (Super Mario Bros, Processed). The URLs are pinned to upstream commit `0e86a8f31f20ecad4eaa5741ff061af88767f7fb` and each file is verified against a SHA-256 stored in `data/make_data.py`, so every run fetches byte-identical data; a mismatch stops the script. It saves the source files under `data/generated/vglc_raw`, the manifest `vglc_sources.json` (file, URL, SHA-256), and the preprocessed table `vglc_features.csv`. Internet is required for this step: a network failure is reported and the script exits with an error after writing the synthetic file; no substitute data is invented. Pass `--no-download` to skip the step offline.

| External column | Meaning |
|---|---|
| `level_name` | Source map filename |
| `width`, `height` | Map dimensions in characters |
| `solid_tile_fraction` | Fraction of all grid symbols from the documented approximate solid-set |
| `enemy_count` | Number of source `E` symbols |
| `longest_ground_gap` | Longest `-` run in bottom row |

VGLC geometry is a structural research resource, **not** the source of the synthetic outcome labels and **not** automatically a playable level in our different physics engine. Our current candidate chunks are original handcrafted templates rather than copies of Nintendo level maps.

## 3. Actual game telemetry

Manual play appends each completed or failed *attempt* to `data/generated/human_gameplay.jsonl`; the graphical four-agent arena appends its attempts to `data/generated/algorithm_gameplay.jsonl`. Both are real logs of what happened in the game (generated at run time, one JSON object per line), not part of the training file.

| Field | Type / units | Written for | Meaning |
|---|---|---|---|
| `won` | 0/1 | both | Attempt reached the goal |
| `elapsed` | seconds | both | Attempt duration (arena: logical ticks × 0.10) |
| `jumps` | count | both | Jumps started from the ground |
| `landings` | count | both | Safe landings |
| `progress` | float [0,1] | both | Furthest fraction of the level reached |
| `seed` | int | both | Generation seed of the level |
| `difficulty` | float [0,1] | both | Realized level difficulty (same definition as the training column) |
| `chunks` | list of strings | both | Chunk template names, in order |
| `origin` | `human` / `algorithm` | both | Who played the attempt |
| `owner` | string | both | `human`, or `arena_bfs` / `arena_dfs` / `arena_astar` / `arena_greedy` |
| `surprise_count` | count | arena only | World surprises accepted during the attempt |
| `unexpected_events` | list of objects | arena only | One entry per surprise: `tick`, `kind`, `description`, `x`, `new_plan_cost`, `search_expanded` |
| `replans` | count | arena only | Times the agent's own search replanned |

The player profile is based on the last 8 attempts. Each owner's online adapter state (13 weights, pending mini-batch, update and forgetting history) is saved separately in `models/online/*.json`. These logs are ignored by git and are never used to retrain or evaluate the shipped base model; the online adapter learns only from fresh attempts in batches of ten. Do not claim real-human experiments unless they are conducted with consent, documented and properly evaluated.
