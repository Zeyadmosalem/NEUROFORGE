# Expedition mode — design

*Approved 16 November 2026. Adds a new game mode; the platformer, arena, tests and report results stay as they are.*

## Goal

An original isometric, pixel-art, top-down alien shooter ("Expedition") in which the same three AI parts cooperate on a **2D map** instead of a left-to-right level. It is inspired by the top-down shooter genre; it does not copy any existing game's name, art, characters or levels.

## Player experience

- Isometric diamond tiles; walls, crates and pillars are raised blocks with a lit top and two shaded faces; all sprites and blocks are depth-sorted.
- The world renders at 640×360 and is scaled ×2 with nearest-neighbour to 1280×720 (crisp pixels). The HUD is drawn at full resolution.
- Darkness with light from the player's flashlight cone, muzzle flashes and acid pools; particles, screen shake, hit flashes.
- Controls: WASD move, mouse aim, left click / hold to shoot, R reload, Tab AI Lab, F1 help, Esc menu.
- Goal: reach the exit beacon alive. Alien nests spawn waves; ammo and health crates are in armory rooms.
- Hero: the user's flat white skull logo (`assets/hero_skull.png`) reduced to a pixel sprite with a small body, legs, gun, walk bob and facing flip.
- Aliens (original, drawn as pixel grids in code): Glob (slow), Skitter (fast), Brute (tough).

## AI parts

| Part | Role in Expedition | Module |
|---|---|---|
| CSP | Builds the R×C room grid. Variables = room cells; domains = room templates. Binary constraints: doors line up between neighbours, border rooms have no outward doors, the start room is safe. Global: difficulty budget and at least one exit. Backtracking with switchable Forward Checking and AC-3, counting assignments, backtracks and pruned values. BFS then verifies the exit is reachable. | `csp/rooms.py` |
| Search | `graph_search()` runs BFS/DFS/A*/Greedy on any graph with the same frontier, tie-break and counter rules as the platformer's `solve()`; the platformer keeps its separately profiled specialised loop (changed during implementation: routing it through callbacks would slow the arena). Skitters plan with Greedy, Brutes with A*, Globs follow a shared BFS distance field from the player. The AI Lab shows all four routes from the player to the exit with counters. | `search/algorithms.py` |
| NN | The trained player model scores several CSP candidate maps through a documented mapping of map features onto its 12 inputs; the controller picks the map closest to the target challenge. Trained on platformer data, so this is a demonstration, not a validated prediction. | `integration/expedition.py` |

## Constraints

- Refactoring the search core must not change any platformer result: the existing tests pass and `eval/run_experiments.py` reproduces identical node counts and path costs.
- Simulation logic (`integration/expedition.py`) imports no Pygame and is testable headlessly.

## Tests

CSP door consistency and all four FC/AC-3 modes; an impossible map returns `No solution`; every generated map is reachable; all four algorithms on a small grid with a known shortest path; aliens close distance to the player; a headless simulation runs; a UI smoke test draws Expedition frames; the full existing suite still passes.
