"""Deterministic tile physics and periodic hazards shared by human and AI agents.

The *same* step() resolves moves for the graphical game and all four search
algorithms. Dynamic objects follow an 8-tick periodic schedule. A search state
includes the current phase, so waiting can be necessary for a valid solution.
These are logical, grid-quantized platformer physics, not continuous rigid bodies.
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

Action = Literal['L', 'R', 'W', 'JL', 'JR', 'J']
ACTIONS: tuple[Action, ...] = ('R', 'JR', 'J', 'L', 'JL', 'W')
CYCLE = (0, 1, 2, 3, 4, 3, 2, 1)  # periodic and one tile maximum per phase


class State(NamedTuple):
    """Immutable search state. A named tuple of ints hashes and compares in C
    and is not tracked by the garbage collector, which matters because the
    planners create and store hundreds of thousands of these."""
    x: int
    y: int  # feet row; for landing, it is a solid support row
    vy: int = 0
    phase: int = 0


@dataclass(frozen=True, slots=True)
class MovingPlatform:
    x: int
    y: int
    width: int = 2
    travel: int = 3
    axis: str = 'x'
    offset: int = 0

    def position(self, phase: int) -> tuple[int, int]:
        # 0..travel oscillation; speed max one cell per logical step.
        delta = min(self.travel, CYCLE[(phase + self.offset) % 8])
        if self.axis == 'x':
            return self.x + delta, self.y
        return self.x, self.y - delta

    def contains(self, x: int, y: int, phase: int) -> bool:
        px, py = self.position(phase)
        return py == y and px <= x < px + self.width


@dataclass(frozen=True, slots=True)
class MovingEnemy:
    x: int
    y: int  # occupied character-body row
    travel: int = 3
    offset: int = 0
    kind: str = 'patrol'

    def position(self, phase: int) -> tuple[int, int]:
        delta = min(self.travel, CYCLE[(phase + self.offset) % 8])
        return self.x + delta, self.y


@dataclass(frozen=True, slots=True)
class TimedGate:
    x: int
    y_top: int
    y_bottom: int
    offset: int = 0
    active_ticks: int = 4

    def active(self, phase: int) -> bool:
        return (phase + self.offset) % 8 < self.active_ticks

    def blocks(self, x: int, y: int, phase: int) -> bool:
        return x == self.x and self.y_top <= y <= self.y_bottom and self.active(phase)


@dataclass(frozen=True)
class Level:
    rows: tuple[str, ...]
    seed: int = 0
    difficulty: float = 0.0
    chunks: tuple[str, ...] = ()
    platforms: tuple[MovingPlatform, ...] = ()
    enemies: tuple[MovingEnemy, ...] = ()
    gates: tuple[TimedGate, ...] = ()

    # A Level is immutable, so everything derived from it is computed once and
    # kept in the instance __dict__ (cached_property bypasses the frozen
    # __setattr__ and is ignored by the generated __eq__/__hash__). World
    # surprises build a new Level via dataclasses.replace(), which starts with
    # an empty cache.
    @cached_property
    def width(self): return len(self.rows[0])

    @cached_property
    def height(self): return len(self.rows)

    @cached_property
    def period(self) -> int:
        return 8 if self.platforms or self.enemies or self.gates else 1

    @cached_property
    def _platform_cells(self) -> tuple[dict, ...]:
        """Per phase 0..7: cell -> first moving platform covering that cell."""
        tables = []
        for phase in range(8):
            cells = {}
            for p in self.platforms:
                px, py = p.position(phase)
                for x in range(px, px + p.width):
                    cells.setdefault((x, py), p)
            tables.append(cells)
        return tuple(tables)

    @cached_property
    def _hazard_cells(self) -> tuple[frozenset, ...]:
        """Per phase 0..7: every lethal cell (spikes, patrol enemies, active gates)."""
        spikes = frozenset((x, y) for y, row in enumerate(self.rows)
                           for x, c in enumerate(row) if c == '^')
        if not (self.enemies or self.gates):
            return (spikes,) * 8
        tables = []
        for phase in range(8):
            cells = {e.position(phase) for e in self.enemies}
            for g in self.gates:
                if g.active(phase):
                    cells.update((g.x, y) for y in range(g.y_top, g.y_bottom + 1))
            tables.append(spikes | cells)
        return tuple(tables)

    @cached_property
    def _enemy_positions(self) -> tuple[tuple, ...]:
        """Per phase 0..7: the position of every patrol enemy, in order."""
        return tuple(tuple(e.position(phase) for e in self.enemies) for phase in range(8))

    @cached_property
    def _solid_cells(self) -> tuple[frozenset, ...]:
        """Per phase 0..7: in-bounds '#' tiles plus moving-platform cells.

        Out-of-bounds solidity (side walls and ceiling) is not stored here;
        callers add `x < 0 or x >= width or y < 0`, exactly like tile().
        """
        blocks = frozenset((x, y) for y, row in enumerate(self.rows)
                           for x, c in enumerate(row) if c == '#')
        if not self.platforms:
            return (blocks,) * 8
        return tuple(blocks | frozenset(cells) for cells in self._platform_cells)

    @cached_property
    def _step_cache(self) -> dict:
        """Memoized step() transitions, one {state: next state} table per action.

        Replanning revisits the same states over and over. Keying by the state
        itself (no per-lookup key object) keeps the planners allocation-light.
        """
        return {action: {} for action in ACTIONS}

    def tile(self, x: int, y: int) -> str:
        if x < 0 or x >= self.width: return '#'
        if y < 0: return '#'
        if y >= self.height: return '.'
        return self.rows[y][x]

    def solid(self, x: int, y: int, phase: int = 0) -> bool:
        return (x < 0 or x >= self.width or y < 0
                or (x, y) in self._solid_cells[phase % 8])

    def support_platform(self, x: int, y: int, phase: int) -> MovingPlatform | None:
        if self.tile(x, y) == '#': return None
        return self._platform_cells[phase % 8].get((x, y))

    @cached_property
    def start(self) -> State:
        return State(1, self.height - 2, 0, 0)

    @cached_property
    def goal_x(self) -> int:
        return self.width - 2

    def complete(self, state: State) -> bool:
        return state.x >= self.goal_x and 0 <= state.y <= self.height

    def features(self) -> dict[str, float]:
        ground = self.height-2
        gaps = [x for x in range(self.width) if self.tile(x, ground) != '#']
        max_run = 0; current = 0
        for x in range(self.width):
            if self.tile(x, ground) != '#':
                current += 1; max_run = max(max_run, current)
            else: current = 0
        hazards = sum(row.count('^') for row in self.rows)
        coins = sum(row.count('o') for row in self.rows)
        platforms = sum(1 for y in range(1, ground) for x in range(self.width)
                        if self.tile(x,y) == '#' and self.tile(x,y+1) != '#')
        return {'length': float(self.width), 'gap_tiles': float(len(gaps)),
                'max_gap': float(max_run), 'hazards': float(hazards),
                'coins': float(coins), 'platforms': float(platforms),
                'difficulty': float(self.difficulty),
                'moving_platforms': float(len(self.platforms)),
                'moving_enemies': float(len(self.enemies)),
                'timed_gates': float(len(self.gates))}


def body_rows(state: State) -> tuple[int, ...]:
    return (state.y - 1,)


def hazard_at(level: Level, x: int, body_y: int, phase: int) -> bool:
    return (x, body_y) in level._hazard_cells[phase % 8]


def is_dead(level: Level, state: State) -> bool:
    if state.y > level.height: return True
    return any(hazard_at(level, state.x, y, state.phase) for y in body_rows(state))


def step(level: Level, state: State, action: Action) -> State | None:
    """One exact logical tick; all searches and GUI use these same rules.

    - WAIT advances obstacle phase, even if the player does not move.
    - A player standing on a moving platform is carried one tile per tick.
    - Patrol enemies and timed gates are lethal on contact or crossed paths.
    - Falls and jumps sweep intermediate rows (no tunneling through spikes).
    """
    if action not in ACTIONS: raise ValueError(f'Invalid action: {action}')
    table = level._step_cache[action]
    try:
        return table[state]
    except KeyError:
        result = table[state] = _step(level, state, action)
        return result


def _step(level: Level, state: State, action: Action) -> State | None:
    """The transition rules. Same logic as always, with the level's solid and
    hazard cells read from precomputed per-phase sets instead of being
    re-derived from tiles and moving objects on every query."""
    sx = state.x; sy = state.y; current = state.phase
    width = level.width; height = level.height
    # is_dead(level, state)
    if sy > height or (sx, sy-1) in level._hazard_cells[current % 8]: return None
    new_phase = (current + 1) % level.period
    hazards = level._hazard_cells[new_phase]
    blocks = level._solid_cells[new_phase]
    dx = -1 if action in ('L','JL') else 1 if action in ('R','JR') else 0
    grounded = state.vy == 0 and (sx < 0 or sx >= width or sy < 0
                                  or (sx, sy) in level._solid_cells[current % 8])
    support = (level.support_platform(sx, sy, current)
               if grounded and level.platforms else None)

    old_x = sx; old_y = sy
    if support is not None:
        ox, oy = support.position(current)
        nx, ny = support.position(new_phase)
        old_x += nx - ox
        old_y += ny - oy
        if old_x < 0 or old_x >= width: return None
        if level.tile(old_x, old_y-1) == '#': return None

    vy = -3 if grounded and action in ('JL', 'JR', 'J') else state.vy
    x = old_x + dx
    if x < 0 or x >= width: return None
    # From here 0 <= x < width (old_x is in range too unless the caller passed
    # an out-of-level state), so "solid" needs only the ceiling test + the set.
    if x != old_x and (old_y-1 < 0 or (x, old_y-1) in blocks):
        x = old_x
    wall = x < 0 or x >= width
    if grounded and vy == 0 and not (wall or old_y < 0 or (x, old_y) in blocks): vy = 1
    y = old_y
    if vy < 0:
        proposed = max(-1, old_y + vy)
        blocked = None
        for r in range(old_y-2, proposed-2, -1):
            if wall or r < 0 or (x, r) in blocks:
                blocked = r; break
        if blocked is not None:
            y = blocked+2
            vy = 0
        else:
            y = proposed
            vy += 1
    else:
        proposed = old_y + vy
        landing = None
        for r in range(old_y, proposed+1):
            if wall or r < 0 or (x, r) in blocks:
                landing = r; break
        if landing is not None:
            y = landing
            vy = 0
        else:
            y = proposed
            vy = min(vy+1, 3)

    # The body's vertical sweep also detects moving hazards active upon arrival.
    for row in range(min(old_y,y)-1, max(old_y,y)):
        if (x, row) in hazards: return None
    # A platform can move a player horizontally through a hazard; disallow it.
    if x != sx:
        for col in range(min(sx,x), max(sx,x)+1):
            if (col, y-1) in hazards: return None
    # Head-on enemy crossing can collide without sharing a final grid cell.
    patrols = level._enemy_positions
    for before, after in zip(patrols[current % 8], patrols[new_phase]):
        if before[1] == sy-1 and after[1] == y-1:
            if (sx-after[0])*(x-before[0]) <= 0 and (
                (sx == after[0] and x == before[0]) or
                (sx == before[0] and x == after[0])
            ):
                return None
    # is_dead(level, result)
    if y > height or (x, y-1) in hazards: return None
    return State(x, y, vy, new_phase)


def parse_level(lines: list[str] | tuple[str, ...]) -> Level:
    lines = tuple(line.rstrip('\n') for line in lines)
    if not lines or any(len(r)!=len(lines[0]) for r in lines):
        raise ValueError('Level rows must have identical lengths')
    return Level(lines)


def replay(level: Level, actions: list[Action]):
    path=[level.start]
    for action in actions:
        nxt=step(level,path[-1],action)
        if nxt is None: return path, False
        path.append(nxt)
    return path, level.complete(path[-1])
