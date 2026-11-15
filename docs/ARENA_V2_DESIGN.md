# NEUROFORGE V2: Four Minds, Adaptive Worlds, Controlled Forgetting

## Who is the player?

**Manual mode:** the human is the player, controlling actions using arrow keys and jump. Their attempts update their own telemetry and adaptation model. The four search algorithms remain an inspectable AI Lab.

**Arena mode:** the four graph-search algorithms are the **players/controllers**. Each algorithm plans a sequence of physically valid actions using the game's shared `step` function: BFS/DFS/A*/Greedy. When an execution slip changes a state, that *same agent algorithm* replans from the actual state. A frozen pretrained neural network is the **difficulty coach**, NOT the movement controller. The CSP generator creates candidates; search verifies them; the NN + a small per-agent online calibration head estimates the win probability.

Each agent owns a distinct pseudo-player profile, procedural world, history, search trace, challenge target, residual weights and ten-outcome update schedule. Distinct motor-noise probabilities introduce different simulated execution styles; this **confounds** causal comparisons of search algorithms in the arena. For a fair pure-search comparison, use `eval/run_experiments.py`, which gives every algorithm the same world and removes motor-noise execution from the benchmark.

## The update process

1. Log `won`, elapsed time, jumps, landings, progress, difficulty and chosen level after every attempt. Save human and algorithm simulation to *separate* local JSONL files if persistence is enabled.
2. Snapshot the **pre-outcome** feature vector before the attempt; never train using the current outcome among the predictor inputs.
3. The base NumPy MLP, originally trained offline on 6,500 synthetic outcomes (250 simulated personas × 26 attempts), is **frozen during live play**.
4. After **exactly 10 new labelled attempts**, do 55 small SGD steps on those **10 samples only** to update a single `(12+1)`-parameter logistic residual that corrects the frozen network's log-odds. No older outcomes enter the gradient, and the base network is not retrained.
5. During those SGD steps, apply a 12% **train-only inverted feature dropout mask** to the adapter inputs; it is a regularizer, not forgetting.
6. By default, after every **three** ten-attempt updates (30 attempts), a deterministic random schedule **zeros ~35% of currently learned adapter feature coefficients**. The bias, original trained MLP and unconsumed samples remain intact. This is *episodic partial adapter forgetting* (not standard neural dropout, and not LoRA).
7. The next 10 fresh outcomes can relearn the affected coefficients. To assist recovery, the first two **new-only** mini-batches after a glitch use a 1.4× adapter learning rate; the original model remains unchanged. The record includes weight norms before/after, chosen features, update count, and subsequent **prequential** log-loss on each fresh batch.
8. Neither recovery nor improvement is guaranteed. Compare to a never-forget ablation using independently seeded simulations. Persist adapter state per player to JSON, including incomplete batches and forgetting history, without touching the MLP checkpoint.

## Difficulty progression

Winning raises an individual challenge budget (accelerates with streaks), losing reduces it; level length grows from six to ten chunks over attempts. A chunk vocabulary of safe zones, gaps, spike runs, raised shelves, blockers and airborne hazards creates different movement patterns. AC-3/Forward Checking constrain chunk choices; A* physics-checks resulting levels. Neural win probability and layout novelty help rank candidate levels, but challenge progression deliberately prevents always choosing the easiest levels.

## Controls

Launch manual game with `python run_demo.py`, or four-panel arena with `python run_demo.py --arena`.

- Manual: arrows/A/D = move; Space/W/Up = jump; Tab = search Lab; 1-4 = search; N = NN toggle; F/C = CSP toggles; G = regenerate; X = **forget now** (manual adapter); V = enter arena; Esc = exit.
- Arena: Space = pause; `+/-` = 2× speed up/down; H = search route overlay; D = toggle scheduled forgetting; X = trigger partial forgetting in all four adapters; M = manual mode; R = restart runners; S = save screenshot; Esc = exit.
- Runtime info per lane: independently generated level, agent position and planned route, cumulative wins/losses, predicted win chance, difficulty budget, search expansions, count of online updates, observations pending (0/10 through 9/10) and countdown to the next scheduled partial reset.

## Reproducible commands

```bash
python -m data.make_data --seed 42 --personas 250 --attempts 26
python -m nn.train --seed 42
python run_demo.py --arena
python run_demo.py --arena-headless --rounds 34 --forget-every 3
python run_demo.py --arena-headless --rounds 34 --forget-every 3 --no-forgetting
python -m eval.forgetting_study --seeds 7 --rounds 22 --forget-every 1
# Optional five-seed supplemental study (can take a few minutes)
python -m eval.forgetting_study --seeds 7 17 27 37 47 --rounds 31 --forget-every 2
python -m pytest tests -v
```

In headless mode, persistence is disabled by default and a generated JSON summary is written to `results/arena_demo_summary.json`. Graphical mode saves per-agent incremental JSON and outcome telemetry locally. The synthetic dataset is a *simulation*, not real human performance evidence.

## Experimental interpretation and remaining limitations

- `results/forgetting_ablation.*` was generated as a **one-seed exploratory smoke comparison** (22 rounds, forgetting every one update). It is not evidence that forgetting helps. Rerun the script on five or more seeds before making empirical claims. During the study, the no-forgetting and forgetting versions have identical random seeds but may diverge because subsequent level selection changes, which is part of the closed-loop treatment.
- Search algorithms all solve the same discrete static physics. A* and BFS can find identical shortest paths; DFS and Greedy have no optimality guarantee. Policy differences can be subtle on simple geometry. Comparisons on an agent-specific adaptive sequence cannot be treated as paired pure-search benchmarks.
- High-skill adaptive challenge reflects simulator assumptions and chunk difficulty estimates; it has not been psychometrically validated on humans. Moving enemies, real-world human datasets and a fully timed difficulty-calibration study remain future work.
- Pygame CE is required for graphics; this build was tested headlessly in a sandbox that could not install Pygame CE. `assets/arena_preview.png` is an **offline render from actual level and agent state**, not a screenshot of a running Pygame window.
