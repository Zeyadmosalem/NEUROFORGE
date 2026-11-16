"""Room-grid CSP that builds Expedition maps (a 2D map instead of a 1D level).

Variables: the cells of a rows x cols grid of rooms.
Domains:   Room(template, doors) values; doors is a subset of {N, E, S, W}.
Unary:     the start cell is the landing room, the exit cell is the beacon room,
           border rooms have no doors that lead off the map, every room has a door.
Binary:    neighbours' facing doors match (an east door needs a west door next
           door), and two nest rooms are never adjacent.
Global:    summed room difficulty within budget +/- 2, and every room is
           connected to the start through doors (checked on complete assignments).

Backtracking with MRV, plus independently switchable Forward Checking and AC-3,
with the same counters as the platformer CSP. A* then verifies the compiled tile
map, exactly like the platformer generator.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from random import Random
from time import perf_counter
from csp.solver import CSPStats, CSPResult
from search.algorithms import solve_grid, SearchResult

ROOM = 10                      # tiles per room side (walls on the room's edge)
DOOR = (4, 5)                  # the two door tiles in the middle of a side
SIDES = ('N', 'E', 'S', 'W')
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
STEP = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
PASSABLE = frozenset('.~ahSE')  # floor, acid, ammo, health, start, exit
MAX_ASSIGNMENTS = 6000          # search budget for one CSP run; generate_map retries a new seed


@dataclass(frozen=True)
class RoomTemplate:
    name: str
    kind: str          # start, exit, open, cover, hazard, supply, nest
    difficulty: int


TEMPLATES = (
    RoomTemplate('landing', 'start', 0),
    RoomTemplate('beacon', 'exit', 1),
    RoomTemplate('hall', 'open', 0),
    RoomTemplate('armory', 'supply', 0),
    RoomTemplate('pillars', 'cover', 1),
    RoomTemplate('cargo', 'cover', 1),
    RoomTemplate('maze', 'cover', 2),
    RoomTemplate('acid', 'hazard', 2),
    RoomTemplate('nest', 'nest', 3),
    RoomTemplate('hive', 'nest', 4),
)
BY_NAME = {t.name: t for t in TEMPLATES}


@dataclass(frozen=True)
class Room:
    template: RoomTemplate
    doors: frozenset

    @property
    def name(self):
        return self.template.name


class RoomCSP:
    def __init__(self, rows=4, cols=4, target=0.5, seed=42, impossible=False):
        if rows < 2 or cols < 2:
            raise ValueError('Need at least a 2x2 room grid')
        self.rows, self.cols, self.n = rows, cols, rows * cols
        self.target = max(0.0, min(1.0, float(target)))
        self.seed = seed
        self.stats = CSPStats()
        self.start_cell, self.exit_cell = 0, self.n - 1
        self.budget = round(self.target * (self.n - 2) * 1.6) + 1   # +1: the beacon room
        self.budget_lo, self.budget_hi = max(0, self.budget - 2), self.budget + 2
        # Harder maps must contain alien nests (like dynamic chunks in the platformer).
        self.min_nests = 0 if self.target < 0.3 else 1 if self.target < 0.6 else 2 if self.target < 0.85 else 3
        self.domains = {i: self._unary_domain(i, impossible) for i in range(self.n)}

    # ---- structure --------------------------------------------------------
    def cell(self, i):
        return divmod(i, self.cols)            # (row, col)

    def neighbour(self, i, side):
        r, c = self.cell(i)
        dc, dr = STEP[side]
        r, c = r + dr, c + dc
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r * self.cols + c
        return None

    def side_between(self, i, j):
        for side in SIDES:
            if self.neighbour(i, side) == j:
                return side
        return None

    def _unary_domain(self, i, impossible):
        inward = [s for s in SIDES if self.neighbour(i, s) is not None]
        door_sets = [frozenset(c) for k in range(1, len(inward) + 1) for c in combinations(inward, k)]
        if i == self.start_cell:
            templates = [BY_NAME['landing']]
        elif i == self.exit_cell:
            templates = [BY_NAME['beacon']]
            if impossible:
                # A sealed exit room: no doors at all contradicts "every room has
                # a door", so the domain is empty and no map can exist.
                door_sets = [d for d in door_sets if not d]
        else:
            templates = [t for t in TEMPLATES if t.kind not in ('start', 'exit')]
        return [Room(t, d) for t in templates for d in door_sets]

    # ---- constraints ------------------------------------------------------
    def compatible(self, i, a, j, b):
        self.stats.constraint_checks += 1
        side = self.side_between(i, j)
        if side is None:
            return True
        if (side in a.doors) != (OPPOSITE[side] in b.doors):
            return False
        if a.template.kind == 'nest' and b.template.kind == 'nest':
            return False
        return True

    def feasible_budget(self, assigned, domains):
        total = sum(r.template.difficulty for r in assigned.values())
        nests = sum(r.template.kind == 'nest' for r in assigned.values())
        rest = [i for i in range(self.n) if i not in assigned]
        if not rest:
            return self.budget_lo <= total <= self.budget_hi and nests >= self.min_nests
        if any(not domains[i] for i in rest):
            return False
        could_nest = sum(any(r.template.kind == 'nest' for r in domains[i]) for i in rest)
        if nests + could_nest < self.min_nests:
            return False
        if not self.could_connect(assigned, domains):
            return False
        # Each nest still required raises a 0-difficulty minimum to at least 3.
        needed = max(0, self.min_nests - nests)
        low = (total + sum(min(r.template.difficulty for r in domains[i]) for i in rest)
               + 3 * needed)
        high = total + sum(max(r.template.difficulty for r in domains[i]) for i in rest)
        return low <= self.budget_hi and high >= self.budget_lo

    def connected(self, assigned):
        seen = {self.start_cell}
        queue = deque([self.start_cell])
        while queue:
            i = queue.popleft()
            for side in assigned[i].doors:
                j = self.neighbour(i, side)
                if j is not None and j not in seen:
                    seen.add(j); queue.append(j)
        return len(seen) == self.n

    def could_connect(self, assigned, domains):
        """Global-constraint check on partial assignments: can every room still be
        reached from the start, if each unassigned room may open any door its
        current domain still allows? Pruning here avoids discovering a cut-off
        room only after all sixteen rooms are placed."""
        def may_open(i, side):
            if i in assigned:
                return side in assigned[i].doors
            return any(side in r.doors for r in domains[i])
        seen = {self.start_cell}
        queue = deque([self.start_cell])
        while queue:
            i = queue.popleft()
            for side in SIDES:
                j = self.neighbour(i, side)
                if j is not None and j not in seen and may_open(i, side) and may_open(j, OPPOSITE[side]):
                    seen.add(j); queue.append(j)
        return len(seen) == self.n

    def ac3(self, domains, arcs=None):
        queue = deque(arcs if arcs is not None else
                      [(i, j) for i in range(self.n) for s in SIDES
                       if (j := self.neighbour(i, s)) is not None])
        while queue:
            i, j = queue.popleft()
            before = len(domains[i]); self.stats.arc_revisions += 1
            domains[i] = [a for a in domains[i] if any(self.compatible(i, a, j, b) for b in domains[j])]
            self.stats.pruned += before - len(domains[i])
            if not domains[i]:
                return False
            if len(domains[i]) < before:
                for s in SIDES:
                    k = self.neighbour(i, s)
                    if k is not None and k != j:
                        queue.append((k, i))
        return True

    # ---- search -----------------------------------------------------------
    def solve(self, forward_checking=True, arc_consistency=True) -> CSPResult:
        started = perf_counter()
        rnd = Random(self.seed)
        domains = {i: list(v) for i, v in self.domains.items()}
        per_room = self.budget / self.n
        result = None
        if not (arc_consistency and not self.ac3(domains)):
            result = self._backtrack({}, domains, rnd, per_room, forward_checking, arc_consistency)
        self.stats.elapsed_ms = round((perf_counter() - started) * 1000, 3)
        if result is None:
            reason = 'search budget exceeded' if self.stats.assignments >= MAX_ASSIGNMENTS else 'No solution'
            return CSPResult(None, self.stats, reason)
        return CSPResult([result[i] for i in range(self.n)], self.stats)

    def _backtrack(self, assigned, domains, rnd, per_room, fc, ac):
        if len(assigned) == self.n:
            ok = self.feasible_budget(assigned, domains) and self.connected(assigned)
            return dict(assigned) if ok else None
        if self.stats.assignments >= MAX_ASSIGNMENTS:
            return None
        i = min((k for k in range(self.n) if k not in assigned), key=lambda k: (len(domains[k]), k))
        values = list(domains[i]); rnd.shuffle(values)
        # Prefer rooms near the per-room difficulty share with a few doors; the
        # random term keeps maps varied and the seed keeps them reproducible.
        values.sort(key=lambda r: abs(r.template.difficulty - per_room) - 0.2 * len(r.doors)
                    + rnd.random() * 2.6)
        for value in values:
            self.stats.assignments += 1
            if any(not self.compatible(i, value, j, other) for j, other in assigned.items()
                   if self.side_between(i, j)):
                self.stats.backtracks += 1
                continue
            nxt_assigned = {**assigned, i: value}
            nxt_domains = {k: list(v) for k, v in domains.items()}
            nxt_domains[i] = [value]
            good = True
            if fc:
                for s in SIDES:
                    j = self.neighbour(i, s)
                    if j is not None and j not in assigned:
                        before = len(nxt_domains[j])
                        nxt_domains[j] = [b for b in nxt_domains[j] if self.compatible(i, value, j, b)]
                        self.stats.pruned += before - len(nxt_domains[j])
                        if not nxt_domains[j]:
                            good = False; break
            if good and ac:
                good = self.ac3(nxt_domains, [(j, i) for s in SIDES
                                              if (j := self.neighbour(i, s)) is not None])
            if good and self.feasible_budget(nxt_assigned, nxt_domains):
                found = self._backtrack(nxt_assigned, nxt_domains, rnd, per_room, fc, ac)
                if found is not None:
                    return found
            self.stats.backtracks += 1
            if self.stats.assignments >= MAX_ASSIGNMENTS:
                return None
        return None


@dataclass
class ExpeditionMap:
    tiles: tuple
    rows: int
    cols: int
    rooms: tuple
    seed: int
    difficulty: float
    start: tuple
    exit: tuple
    nests: tuple = field(default_factory=tuple)

    @property
    def width(self):
        return len(self.tiles[0])

    @property
    def height(self):
        return len(self.tiles)

    @property
    def chunks(self):
        """Room template names, row by row (mirrors the platformer's Level.chunks)."""
        return tuple(r.name for r in self.rooms)

    def tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return '#'

    def passable(self, x, y):
        return self.tile(x, y) in PASSABLE

    def count(self, codes):
        return sum(row.count(c) for row in self.tiles for c in codes)


def _quadrant_cells():
    return [(x, y) for x in range(1, ROOM - 1) for y in range(1, ROOM - 1)
            if x not in DOOR and y not in DOOR]


def _furnish(room, rng):
    """Interior layout of one room in local coordinates (the central cross stays clear)."""
    marks = {}
    q = _quadrant_cells()
    name = room.name
    corners = [(2, 2), (7, 2), (2, 7), (7, 7)]
    if name == 'landing':
        marks[(4, 4)] = 'S'
        marks[(1, 1)] = 'c'; marks[(8, 8)] = 'c'
    elif name == 'beacon':
        marks[(4, 4)] = 'E'
        for c in corners:
            marks[c] = 'p'
    elif name == 'hall':
        for cell in rng.sample(q, 2):
            marks[cell] = 'c'
    elif name == 'armory':
        marks[(2, 2)] = 'a'; marks[(7, 7)] = 'h'; marks[(7, 2)] = 'a'
        for cell in rng.sample([c for c in q if c not in marks], 3):
            marks[cell] = 'c'
    elif name == 'pillars':
        for c in corners:
            marks[c] = 'p'
    elif name == 'cargo':
        for cell in rng.sample(q, 9):
            marks[cell] = 'c'
    elif name == 'maze':
        for x in (1, 2, 3):
            marks[(x, 2)] = '#'
        for y in (6, 7, 8):
            marks[(7, y)] = '#'
        marks[(2, 7)] = 'p'
    elif name == 'acid':
        for x, y in q:
            if (2 <= x <= 3 and 1 <= y <= 3) or (6 <= x <= 7 and 6 <= y <= 8):
                marks[(x, y)] = '~'
        marks[(7, 2)] = 'h'
    elif name == 'nest':
        marks[(7, 2)] = 'N'
        for x, y in ((6, 1), (8, 1), (6, 3), (8, 3)):
            marks[(x, y)] = '~'
        marks[(2, 7)] = 'c'
    elif name == 'hive':
        marks[(2, 7)] = 'N'; marks[(7, 2)] = 'N'
        for x, y in ((2, 2), (3, 2), (2, 3), (6, 7), (7, 7)):
            marks[(x, y)] = '~'
    return marks


def compile_rooms(rooms, rows, cols, seed=0) -> ExpeditionMap:
    rng = Random(seed)
    width, height = cols * ROOM, rows * ROOM
    grid = [['#'] * width for _ in range(height)]
    start = exit_ = None
    nests = []
    for index, room in enumerate(rooms):
        r, c = divmod(index, cols)
        ox, oy = c * ROOM, r * ROOM
        for y in range(1, ROOM - 1):
            for x in range(1, ROOM - 1):
                grid[oy + y][ox + x] = '.'
        for side in room.doors:
            for d in DOOR:
                x, y = {'N': (d, 0), 'S': (d, ROOM - 1), 'W': (0, d), 'E': (ROOM - 1, d)}[side]
                grid[oy + y][ox + x] = '.'
        for (x, y), mark in _furnish(room, rng).items():
            grid[oy + y][ox + x] = mark
            if mark == 'S':
                start = (ox + x, oy + y)
            elif mark == 'E':
                exit_ = (ox + x, oy + y)
            elif mark == 'N':
                nests.append((ox + x, oy + y))
    total = sum(r.template.difficulty for r in rooms)
    difficulty = min(1.0, total / max(1.0, (rows * cols - 2) * 1.6))
    return ExpeditionMap(tuple(''.join(row) for row in grid), rows, cols, tuple(rooms), seed,
                         difficulty, start, exit_, tuple(nests))


@dataclass
class MapResult:
    map: ExpeditionMap | None
    csp: CSPResult | None
    search: SearchResult | None
    candidates: int
    reason: str = ''


def generate_map(seed=42, target=0.5, rows=4, cols=4, forward_checking=True,
                 arc_consistency=True, attempts=12, impossible=False) -> MapResult:
    last_csp = last_search = None
    for attempt in range(attempts):
        problem = RoomCSP(rows, cols, target, seed + attempt * 997, impossible)
        last_csp = problem.solve(forward_checking, arc_consistency)
        if last_csp.chunks is None:
            if last_csp.reason == 'No solution':
                return MapResult(None, last_csp, None, attempt + 1, 'No solution')
            continue
        m = compile_rooms(last_csp.chunks, rows, cols, seed + attempt)
        last_search = solve_grid(m.passable, m.start, m.exit, 'astar')
        if last_search.found:
            return MapResult(m, last_csp, last_search, attempt + 1)
    return MapResult(None, last_csp, last_search, attempts,
                     'No solution: every candidate map failed verification')
