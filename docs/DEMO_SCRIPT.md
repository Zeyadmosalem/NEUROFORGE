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

1. `python run_demo.py --seed 42` → show the printed demo and `results/demo_integration.png`, then choose **Play** on the title screen. Play with A/D + SPACE, show coins, spikes, the goal arrow and the finish portal. If a display is unavailable use `python run_demo.py --headless --seed 42`.
2. Press `Tab` → AI Lab dashboard. The search chart compares nodes expanded by BFS, DFS, A* and Greedy on the *same* level; press `1`, `2`, `3`, `4` to animate each one's exploration and route.
3. Press `F`, `C` to independently toggle Forward Checking and AC-3; the CSP chart highlights the current setting next to the other three. Press `G` to regenerate the level with those switches and read its backtracks/pruning line.
4. Press `N` to switch the trained NN and the rule-based baseline; the two gauges show both estimates and which one now picks your next level. Complete or fail a level to change the player profile and compare predictions.
5. Press `V` (or `Esc` → **AI Arena**) for the four-agent arena; press `E` to force a world surprise and point at the replan counters.
6. Show `results/experiment_summary.json` and `results/*_plot.png` from reproducible benchmarks. To demonstrate impossibility run the text demo and point to `IMPOSSIBLE CSP: No solution`, and the explicit impossible tests.

**Demo logistics:** All Python processes should start from the project root. The GUI is optional for algorithmic testing but Pygame CE is required for gameplay. The laboratory overlay visualizes search expansion and a found route on a common deterministic simulation; all counters are real. Toggle `F`/`C` affects subsequent regeneration (`G`), not the level already on-screen. NN toggle affects subsequent level selections.
