# Expedition Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Expedition", an isometric pixel-art top-down alien shooter mode in which a room-grid CSP builds the map, BFS/DFS/A*/Greedy move the aliens and power the AI Lab, and the trained NN coaches map difficulty.

**Architecture:** Pure-logic layers (shared search core, room CSP, expedition simulation + NN director) have no Pygame imports and are unit-tested headlessly. A Pygame renderer draws the simulation isometrically at 640×360, scales it ×2 with nearest-neighbour, then draws the HUD at full resolution with the existing `game/ui.py` kit.

**Tech Stack:** Python 3.14, pygame-ce 2.5.8, numpy (existing MLP), pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-expedition-mode-design.md`

## Global Constraints

- The platformer, arena, NN training and report results must not change: the existing 49 tests pass, and `python -m eval.run_experiments --seeds 11 22 33 44 55` reproduces identical node counts and path costs in `results/search_summary.csv`.
- `integration/expedition.py`, `csp/rooms.py` and `search/algorithms.py` import no Pygame.
- Original art only; no name, art, characters or levels from any existing game.
- Hero sprite comes from `assets/hero_skull.png` (copy of the user's flat white skull logo, 256×256 RGBA, transparent background); a built-in pixel skull is the fallback if the file is missing.
- Screens are 1280×720 logical (`game.ui.WIDTH/HEIGHT`); Esc returns `'title'`, closing the window returns `None`.

---

## File Structure

| File | Responsibility |
|---|---|
| `search/algorithms.py` (modify) | `graph_search()` shared core; `solve()` becomes a thin platformer wrapper with identical results; new `solve_grid()` and `bfs_distance_field()` for 2D grids. |
| `csp/rooms.py` (create) | Room templates, `RoomCSP` (Backtracking + switchable FC / AC-3, counters), `compile_rooms()` → `ExpeditionMap`, `generate_map()` with BFS reachability check. |
| `integration/expedition.py` (create) | `map_features()` NN mapping, `ExpeditionDirector` (candidate maps + NN + online adapter), `Expedition` simulation (player, bullets, aliens, nests, pickups, win/lose, events), `lab_routes()`. |
| `game/pixel_art.py` (create) | Pixel sprites: skull hero built from the logo, aliens / nest / pickups / beacon from character grids, hit-flash and flip variants. |
| `game/expedition_app.py` (create) | Isometric renderer, lighting, input, HUD, minimap, AI Lab, help. `run_expedition(seed, persist)`. |
| `game/title.py`, `run_demo.py` (modify) | EXPEDITION menu entry and mode loop. |
| `tests/test_expedition.py` (create), `tests/test_ui_smoke.py` (modify) | Tests below. |
| `README.md`, `docs/DEMO_SCRIPT.md` (modify) | Controls and how Expedition uses the three AI parts. |

---

### Task 1: Shared search core + grid search

**Files:** Modify `search/algorithms.py`; Test `tests/test_expedition.py`

**Interfaces — Produces:**
- `graph_search(start, is_goal, successors, h, algorithm, max_expansions=18000, trace_limit=3500, dfs_successors=None) -> SearchResult` — `successors(s)` yields `(action, next_state)`.
- `solve(level, algorithm='astar', max_expansions=18000, trace_limit=3500, start=None) -> SearchResult` — unchanged signature and results.
- `solve_grid(passable, start, goal, algorithm='astar', max_expansions=20000, trace_limit=3500) -> SearchResult` — `passable(x, y) -> bool`, states are `(x, y)` tuples, actions `'N','E','S','W'`, Manhattan heuristic.
- `bfs_distance_field(passable, source, width, height) -> dict[(x, y), int]`.

- [ ] **Step 1: Failing tests**

```python
from search.algorithms import solve_grid, bfs_distance_field
GRID = ["......",
        ".####.",
        ".#..#.",
        ".#..#.",
        "......"]
def open_cell(x, y):
    return 0 <= y < len(GRID) and 0 <= x < len(GRID[0]) and GRID[y][x] == '.'

def test_grid_search_known_shortest_path():
    for algo in ('bfs', 'dfs', 'astar', 'greedy'):
        r = solve_grid(open_cell, (0, 0), (5, 4), algo)
        assert r.found and r.path[0] == (0, 0) and r.path[-1] == (5, 4)
        assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(r.path, r.path[1:]))
    assert solve_grid(open_cell, (0, 0), (5, 4), 'bfs').cost == 9
    assert solve_grid(open_cell, (0, 0), (5, 4), 'astar').cost == 9
    assert solve_grid(open_cell, (0, 0), (2, 2), 'bfs').reason == 'No solution'

def test_bfs_distance_field():
    field = bfs_distance_field(open_cell, (0, 0), 6, 5)
    assert field[(0, 0)] == 0 and field[(5, 0)] == 5 and field[(5, 4)] == 9
    assert (2, 2) not in field
```

- [ ] **Step 2:** `pytest tests/test_expedition.py -v` → FAIL (ImportError).
- [ ] **Step 3:** Move the body of `solve()` into `graph_search()` unchanged in order of operations (FIFO/LIFO/heap, stale-record skip, `nxt is None or nxt == cur` skip, `frontier_max` update after each expansion). `solve()` builds `successors` from `ACTIONS` and `dfs_successors` from `('W','L','JL','J','R','JR')`, `h = lambda s: heuristic(s, level)`, `is_goal = level.complete`. `solve_grid()` builds 4-neighbour successors over `passable`. `bfs_distance_field()` is a plain deque BFS.
- [ ] **Step 4:** Run the new tests and the full suite: all pass. Run the experiments and `git diff results/search_summary.csv` — only `elapsed_ms` columns may differ.
- [ ] **Step 5:** Commit `search: shared graph-search core with 2D grid entry points`.

### Task 2: Room CSP and map compiler

**Files:** Create `csp/rooms.py`; Test `tests/test_expedition.py`

**Interfaces — Produces:**
- `ROOM = 10` tiles per room side; doors are the two middle tiles of a side.
- `RoomTemplate(name, kind, difficulty)`; `Room(template, doors: frozenset[str])` (`doors ⊆ {'N','E','S','W'}`).
- `RoomCSP(rows=4, cols=4, target=0.5, seed=42, impossible=False).solve(forward_checking=True, arc_consistency=True) -> CSPResult` (reuses `csp.solver.CSPStats/CSPResult`; `.chunks` holds the row-major `list[Room]` or `None` with reason `'No solution'`).
- `ExpeditionMap` dataclass: `tiles: tuple[str, ...]`, `rows`, `cols`, `rooms`, `seed`, `difficulty`, `start: (x, y)`, `exit: (x, y)`, `nests: tuple[(x, y), ...]`; methods `tile(x, y)`, `passable(x, y)`, properties `width`, `height`, `chunks` (room template names).
- Tile codes: `#` wall, `.` floor, `c` crate, `p` pillar, `~` acid (walkable, harmful), `N` nest (solid), `a` ammo, `h` health, `S` start, `E` exit.
- `compile_rooms(rooms, rows, cols, seed) -> ExpeditionMap`; `generate_map(seed=42, target=0.5, rows=4, cols=4, forward_checking=True, arc_consistency=True, attempts=12) -> MapResult(map, csp, search, candidates, reason)`.

Constraints: start cell `(0,0)` domain = `landing`, exit cell `(rows-1, cols-1)` domain = `beacon`; border rooms have no outward doors; every room has ≥1 door; neighbours' facing doors match; two nests never adjacent; summed difficulty within `budget±2` where `budget = round(target * (rows*cols-2) * 1.6)`; complete assignments must connect start and exit through doors. Interiors keep the central cross (room rows/cols 4–5) clear so rooms are internally connected.

- [ ] **Step 1: Failing tests**

```python
from csp.rooms import RoomCSP, generate_map
from search.algorithms import solve_grid
import pytest

@pytest.mark.parametrize('fc,ac', [(False, False), (True, False), (False, True), (True, True)])
def test_room_csp_modes_give_consistent_doors(fc, ac):
    r = RoomCSP(4, 4, 0.5, seed=7).solve(fc, ac)
    rooms = r.chunks
    assert rooms is not None and r.stats.assignments >= 16
    grid = [rooms[i*4:(i+1)*4] for i in range(4)]
    for y in range(4):
        for x in range(4):
            d = grid[y][x].doors
            assert ('W' in d) == (x > 0 and 'E' in grid[y][x-1].doors)
            assert ('N' in d) == (y > 0 and 'S' in grid[y-1][x].doors)
            assert d and (x < 3 or 'E' not in d) and (y < 3 or 'S' not in d)
    assert grid[0][0].template.kind == 'start' and grid[3][3].template.kind == 'exit'

def test_room_csp_impossible_says_no_solution():
    r = RoomCSP(impossible=True).solve()
    assert r.chunks is None and r.reason == 'No solution'

@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5])
def test_generated_maps_are_reachable(seed):
    m = generate_map(seed=seed, target=0.2 * seed).map
    assert m is not None
    assert solve_grid(m.passable, m.start, m.exit, 'bfs').found
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** run → PASS. **Step 5:** commit `csp: room-grid CSP for Expedition maps`.

### Task 3: Expedition simulation and NN director

**Files:** Create `integration/expedition.py`; Test `tests/test_expedition.py`

**Interfaces — Consumes:** Task 1 `solve_grid`, `bfs_distance_field`; Task 2 `generate_map`, `ExpeditionMap`; existing `integration.adaptive.AdaptiveController`, `data.make_data.FEATURES`.
**Produces:**
- `ALIENS = {'glob': {...'algorithm': 'bfs'}, 'skitter': {...'greedy'}, 'brute': {...'astar'}}` with `speed`, `hp`, `damage`, `radius`.
- `map_features(m) -> dict` with the 7 level keys of `FEATURES` (length = BFS path length start→exit, gap_tiles = acid tiles, max_gap = acid tiles / 4, hazards = nests, coins = pickups, platforms = (crates + pillars) / 4, difficulty).
- `ExpeditionDirector(seed=42, persist=False)`: `.next_map(n_candidates=4) -> MapChoice(map, predicted_win, score)`; `.record(won, elapsed, shots, hits, progress) -> dict`; `.controller` (an `AdaptiveController`, owner `'expedition'`).
- `Expedition(m, seed=42, spawning=True)`: fields `player` (`x, y, hp, ammo, mag, angle`), `aliens`, `bullets`, `pickups`, `nests`, `kills`, `shots`, `hits`, `time`, `won`, `dead`; `update(dt, move=(0, 0), aim=(1, 0), shoot=False, reload=False) -> list[tuple]` events (`'shot'|'hit'|'kill'|'hurt'|'pickup'|'spawn'|'nest_down'|'win'|'dead'`, x, y); `spawn(kind, x, y)`; `lab_routes() -> dict[str, SearchResult]`.

- [ ] **Step 1: Failing tests**

```python
from integration.expedition import Expedition, ExpeditionDirector, map_features
from csp.rooms import generate_map
from search.algorithms import solve_grid

def test_aliens_close_the_distance():
    m = generate_map(seed=3, target=0.4).map
    ex = Expedition(m, seed=3, spawning=False)
    route = solve_grid(m.passable, m.start, m.exit, 'bfs').path
    for kind, cell in zip(('glob', 'skitter', 'brute'), route[8:11]):
        ex.spawn(kind, cell[0] + 0.5, cell[1] + 0.5)
    before = [abs(a.x - ex.player.x) + abs(a.y - ex.player.y) for a in ex.aliens]
    for _ in range(90):
        ex.update(1 / 30)
    after = [abs(a.x - ex.player.x) + abs(a.y - ex.player.y) for a in ex.aliens]
    assert all(b < a for a, b in zip(before, after))

def test_player_following_route_reaches_exit():
    m = generate_map(seed=5, target=0.3).map
    ex = Expedition(m, seed=5, spawning=False)
    route = solve_grid(m.passable, m.start, m.exit, 'astar').path
    for cx, cy in route:
        for _ in range(60):
            dx, dy = cx + 0.5 - ex.player.x, cy + 0.5 - ex.player.y
            if abs(dx) + abs(dy) < 0.1 or ex.won:
                break
            ex.update(1 / 30, move=(dx, dy))
    assert ex.won and not ex.dead

def test_director_scores_maps_and_learns():
    d = ExpeditionDirector(seed=9)
    choice = d.next_map(n_candidates=3)
    assert 0 <= choice.predicted_win <= 1 and choice.map is not None
    feats = map_features(choice.map)
    assert set(feats) >= {'length', 'hazards', 'difficulty'}
    before = d.controller.challenge
    d.record(True, 30.0, shots=20, hits=12, progress=1.0)
    assert d.controller.challenge > before
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement (axis-separated circle-vs-tile collision; bullets with sub-steps; globs step down the shared BFS field refreshed every 0.3 s; skitters / brutes replan with `solve_grid` Greedy / A* every 0.4–0.6 s, staggered; nests spawn within 10 tiles of the player, max 4 alive each, have 14 HP; acid deals 12 HP/s; ammo pickup +30, health +35; win within 0.9 tiles of `exit`). **Step 4:** PASS. **Step 5:** commit `integration: Expedition simulation and NN map director`.

### Task 4: Pixel art and isometric renderer

**Files:** Create `game/pixel_art.py`, `game/expedition_app.py`, `assets/hero_skull.png`; Modify `game/title.py`, `run_demo.py`, `tests/test_ui_smoke.py`

**Interfaces — Consumes:** Tasks 2–3; `game.ui` (`Kit`, `open_window`, `toggle_fullscreen`, `Particles`, `Toasts`, `PALETTE`).
**Produces:** `run_expedition(seed=42, persist=True) -> 'title' | None`; title menu returns `'expedition'`.

Rendering rules: tile diamond 32×16 px at 640×360; `iso(x, y) = ((x - y) * 16, (x + y) * 8)`; floor layer pre-rendered once per map; walls (height 22), pillars (30), crates (11) and the nest are pre-rendered block sprites; blocks and sprites drawn in `x + y` depth order; blocks within 2 tiles in front of the player drawn translucent; darkness surface with flashlight cone (rotated cached texture), muzzle, acid and beacon lights subtracted; ×2 nearest scale; HUD: health, ammo, kills, NN predicted win, minimap, toasts, AI Lab (four routes to exit drawn in the world + counters, room-CSP stats under four FC/AC-3 settings, NN vs rule gauges), help overlay, crosshair.

- [ ] **Step 1: Failing smoke tests** (append to `tests/test_ui_smoke.py`; update the title test so two Downs then Enter reaches `'arena'`)

```python
def test_expedition_draws_lab_help_and_escape_returns_to_title(monkeypatch):
    from game.expedition_app import run_expedition
    drive(monkeypatch, 90, {5: pg.K_TAB, 20: pg.K_F1, 30: pg.K_F1, 40: pg.K_ESCAPE})
    assert run_expedition(42, persist=False) == 'title'

def test_title_enter_starts_expedition(monkeypatch):
    from game.title import run_title
    drive(monkeypatch, 30, {3: pg.K_RETURN})
    assert run_title(42) == 'expedition'
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement; copy the logo. **Step 4:** PASS; capture screenshots and fix visual issues. **Step 5:** commit `game: Expedition isometric pixel-art mode`.

### Task 5: Docs, full verification, commit

- [ ] README: Expedition section (controls, the three AI parts, NN-mapping caveat, screenshot); `docs/DEMO_SCRIPT.md`: one demo step.
- [ ] Full suite passes; experiments reproduce identical search node counts / path costs.
- [ ] Commit `docs: Expedition mode`.
