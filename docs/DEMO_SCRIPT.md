# Presentation script — 10-minute talk + 5-minute live demo

## Presentation (10 minutes)

- **0:00–1:30 Problem** — Static difficulty frustrates beginners and experts; we hypothesize candidate-level search plus learned player modeling can personalize constraints.
- **1:30–3:00 Shared mechanics and search** — Explain state `(x,y,vy)`, real movement transitions, four strategies, admissible horizontal-distance heuristic, counters.
- **3:00–4:30 CSP** — Explain chunk variables, domains, adjacency, global difficulty budget, Backtracking, FC and AC-3 as independently toggleable propagation.
- **4:30–6:00 Neural model** — 12 input features, hidden 32/16 ReLU, binary output and logistic baseline; simulated labels explicitly identified.
- **6:00–7:30 Integration** — Player telemetry → trained predictor → CSP candidates → A* physical validation → selection by closeness to target P(win).
- **7:30–9:00 Experiments** — Show actual means ± std in results; compare algorithms, pruning, synthetic adaptation and acknowledge NN baseline result.
- **9:00–10:00 Limitations** — synthetic labels, coarse time-discretized physics, optional external VGLC geometry, no verified human transfer; next steps.

## Live demo (5 minutes)

1. `python run_demo.py --seed 42` → play with A/D + SPACE, show coins, spikes, finish portal. If a display is unavailable use `python run_demo.py --headless --seed 42`.
2. Press `TAB` → AI lab; `1`, `2`, `3`, `4` → see BFS, DFS, A*, Greedy traces and counters on the *same* generated map.
3. Press `F`, `C` to independently toggle Forward Checking and AC-3; press `G` to regenerate the level with current switches and inspect backtracks/pruning.
4. Press `N` to toggle the trained NN vs rule-based baseline. Play/complete a level to change the player profile and select the next level; compare predictions.
5. Show `results/experiment_summary.json` and `results/*_plot.png` from reproducible benchmarks. To demonstrate impossibility run the text demo and point to `IMPOSSIBLE CSP: No solution`, and the explicit impossible tests.

**Demo logistics:** All Python processes should start from the project root. The GUI is optional for algorithmic testing but Pygame CE is required for gameplay. The laboratory overlay visualizes search expansion and a found route on a common deterministic simulation; all counters are real. Toggle `F`/`C` affects subsequent regeneration (`G`), not the level already on-screen. NN toggle affects subsequent level selections.
