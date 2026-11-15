# NEUROFORGE V3 — Dynamic Physics & Reactive Surprise Worlds

## What is different from V2?

- The **same** `game.physics.step()` simulates humans, BFS, DFS, A*, and Greedy. A state includes a cycle phase (`0..7` when hazards move), so planning must account for *when* the character crosses a tile. The `W` (wait) action advances hazard time, and riding a platform can move the player.
- CSP contains moving-ferry, shuttle, patrol-enemy and pulse-gate chunks, as well as static challenges. Generated levels must pass a time-aware A* playability check.
- Platform movement, enemy positions, and gate status are now drawn in **both** arena and human graphics using the same phase used by the physics engine.
- The arena features **seeded, unexpected changes within a level**. Events are triggered on unpredictable per-agent intervals (typically every 11–18 simulation ticks, up to 3 accepted events per stage). The agents cannot plan ahead for future event *occurrence* or *choice*.

## Surprise events

The new `integration/world_events.py` supports:

| Change | How it can affect play |
| --- | --- |
| Spike bloom | A previously safe tile becomes dangerous; the agent may need a new jump |
| New patrol | A moving enemy appears ahead in a safe, occupied-ground corridor |
| Patrol switch | Existing enemy changes its patrol timing |
| Gate shuffle | A timed gate changes its activation phase |
| Platform shuffle | A future moving platform changes its timing (not under the current player) |
| Rising bridge | A gap gains a solid platform |
| Coin shower | Collectible coins appear ahead |

A proposed world update is rejected if it immediately kills the actor, is too close, or if **that same agent's algorithm** fails to find a route from its exact current position within a finite search budget. Accepted updates immediately invalidate the old plan, record a telemetry event, and **replan** using BFS, DFS, A*, or Greedy respectively. Randomness is reproducible from each lane's independent RNG seed, but is unknown to the planner until it happens.

**Important limits:** This is reactive model-based planning with full access to the *current* world map, **not reinforcement learning**, **not fog-of-war**, **not a learned movement policy**, and not robust control against completely unknowable hazards. The neural network coaches **level selection and outcome calibration**, and the per-agent small residual head updates after each 10 *new* attempts; it does not control jumps. New hazards remain deterministic for prediction after they are observed. A future extension would use limited-range sensors and uncertain obstacle beliefs.

## GUI and headless usage

```bash
python run_demo.py --arena                   # V3 four-agent arena, surprises enabled
python run_demo.py                           # human 2D platformer
python run_demo.py --arena-headless --rounds 5 --surprises --seed 23
python -m pytest -q tests/test_world_surprises.py
python -m eval.reactivity_study --seeds 11 22 33 44 55 --rounds 3
```

**Arena controls:** `U` toggles automatic surprises; `E` forces one new surprise attempt per lane; `H` shows path, `Space` pauses, `+/-` sets speed, `D` toggles scheduled adapter forgetting, `X` triggers deliberate partial memory erasure, `M` switches to human play, `S` saves a screenshot.

`--arena-headless` remains with surprises **off by default** for backwards-compatible evaluation. Enable them explicitly using `--surprises`. GUI arena has surprise events **on** by default.

## Research questions we can actually measure

1. Does a change force a different search plan? Unit tests verify this using a known flat level and a sudden spike.
2. Can the four search strategies plan in a time-varying environment? Unit tests verify crossing a gate only during a safe phase and holding position on moving platforms.
3. How do mid-level changes alter time-to-goal, win rate, replans, and node expansions? `eval/reactivity_study.py` writes raw and aggregate CSVs across five deterministic seeds, using independent trials for each search algorithm.
4. How does outcome model calibration recover following deliberate adapter forgetting? Existing `eval/forgetting_study.py` supports this experiment. Forgetting is not assured to improve outcomes.

## Caveats for the professor

The solver **pre-validates** each accepted surprise from the actor's exact current state. This deliberately avoids unwinnable surprise traps, but means the experiment measures reaction to *solvable* changes, not every possible world perturbation. Headless agents are synthetic players, not humans. Descriptive before/after statistics cannot establish that an agent has learned a generalized motor skill. The genuinely learned part is the online completion-probability adapter; the movement strategy is classical replanning.

## Exploratory five-seed measurements

The `reactivity_study.py` script was run with seeds **11, 22, 33, 44, 55**, three attempts per lane, with the usual surrogate motor noise. On average, agents encountered **6.4–7.6 accepted changes** per three attempts in surprise mode. Their routes were recalculated more often. BFS averaged 2.6 replans without surprises versus 8.4 with surprises; A* averaged 1.6 versus 8.4; DFS 6.2 versus 13.8; Greedy 9.6 versus 16.2. Greedy's mean completion proportion changed from 100% to 93.3% in this small exploratory study; other methods mostly stayed at or near 100%. **This is not a statistical claim of superiority.** See `results/reactivity_raw.csv`, `results/reactivity_summary.csv`, and `results/reactivity_comparison.png` for auditable measurements.
