"""Expedition: the headless simulation behind the isometric alien shooter.

The three AI parts cooperate on a 2D map:
- CSP (csp.rooms) builds the room-grid map; A* verifies the exit is reachable.
- Search moves the aliens: Globs follow one shared BFS distance field from the
  player, Skitters replan with Greedy best-first, Brutes replan with A*.
- The NN (the platformer's trained model plus this mode's own online adapter)
  scores candidate maps through `map_features`, a documented mapping onto its
  12 inputs. The model was trained on platformer data, so in this mode its
  numbers are a demonstration, not a validated prediction.

No Pygame here: everything is unit-testable. Coordinates are in tiles; a tile
(x, y) covers [x, x+1) x [y, y+1).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import hypot, exp, floor, ceil
from random import Random
from csp.rooms import generate_map, ExpeditionMap, MapResult
from search.algorithms import solve_grid, bfs_distance_field
from integration.adaptive import AdaptiveController
from data.make_data import FEATURES

ALIENS = {
    'glob':    {'speed': 1.7, 'hp': 3, 'damage': 9, 'radius': 0.32, 'algorithm': 'bfs', 'replan': 0.3},
    'skitter': {'speed': 3.3, 'hp': 1, 'damage': 5, 'radius': 0.25, 'algorithm': 'greedy', 'replan': 0.45},
    'brute':   {'speed': 1.25, 'hp': 9, 'damage': 18, 'radius': 0.42, 'algorithm': 'astar', 'replan': 0.6},
}
PLAYER_SPEED = 4.2
PLAYER_RADIUS = 0.28
BODY_RADIUS = 0.22          # collision radius used for every body against walls
MAG_SIZE = 12
START_AMMO = 60
RELOAD_TIME = 1.1
FIRE_DELAY = 0.11
BULLET_SPEED = 18.0
BULLET_LIFE = 0.8
AGGRO_STEPS = 13            # field distance at which a dormant alien notices the player
ATTACK_COOLDOWN = 0.8
ACID_DPS = 12.0
NEST_HP = 14
NEST_RANGE = 10.0
NEST_MAX_ALIVE = 4
FIELD_REFRESH = 0.3
TARGET_WIN = 0.70


@dataclass
class Player:
    x: float
    y: float
    hp: float = 100.0
    ammo: int = START_AMMO
    mag: int = MAG_SIZE
    aim: tuple = (1.0, 0.0)
    reload_left: float = 0.0
    fire_cd: float = 0.0
    moving: bool = False


@dataclass
class Alien:
    kind: str
    x: float
    y: float
    hp: float
    algorithm: str
    speed: float
    damage: float
    radius: float
    active: bool = True
    frozen: bool = False
    path: list = field(default_factory=list)
    replan_in: float = 0.0
    cooldown: float = 0.0
    hit_t: float = 0.0
    nest: int | None = None
    heading: tuple = (1.0, 0.0)
    last_search: object = None


@dataclass
class Bullet:
    x: float
    y: float
    dx: float
    dy: float
    life: float = BULLET_LIFE


@dataclass
class Pickup:
    kind: str           # 'ammo' or 'health'
    x: float
    y: float
    taken: bool = False


@dataclass
class Nest:
    x: int
    y: int
    hp: float = NEST_HP
    timer: float = 1.5
    alive: int = 0

    @property
    def dead(self):
        return self.hp <= 0


TRAINING_LENGTH = 48.0      # every training level had 6 chunks x 8 tiles


def map_features(m: ExpeditionMap) -> dict:
    """Map an Expedition map onto the 7 level inputs the trained model expects.

    The model only saw platformer levels, so each value is scaled into the range
    it was trained on (data/generated/attempts.csv):
    length      = 48 always: it was constant in training, so it carries no signal
                  and any other value would be extrapolation
    gap_tiles   = acid tiles / 2, at most 7   (ground you should not stand on)
    max_gap     = acid tiles / 6, at most 4
    hazards     = 2 per alien nest, at most 4
    coins       = 13 + ammo and health pickups, at most 24
    platforms   = crates and pillars / 6, at most 8   (cover)
    difficulty  = realized CSP room difficulty (already 0..1)
    """
    acid = m.count('~')
    return {'length': TRAINING_LENGTH, 'gap_tiles': float(min(7.0, acid / 2)),
            'max_gap': float(min(4.0, acid / 6)), 'hazards': float(min(4, 2 * len(m.nests))),
            'coins': float(min(24, 13 + m.count('ah'))),
            'platforms': float(min(8.0, m.count('cp') / 6)), 'difficulty': float(m.difficulty)}


def rule_predict(profile, m: ExpeditionMap) -> float:
    """Rule baseline: player skill against map difficulty (no learning)."""
    p = profile.features()
    skill = 0.65 * p['recent_win_rate'] + 0.35 * p['jump_accuracy']
    return 1 / (1 + exp(-4 * (skill - m.difficulty)))


@dataclass
class MapChoice:
    map: ExpeditionMap
    predicted_win: float
    score: float
    result: MapResult
    candidates: list = field(default_factory=list)


class ExpeditionDirector:
    """Chooses the next map: CSP candidates around the player's challenge level,
    scored by the NN (or the rule baseline) for closeness to a 70% win chance."""

    def __init__(self, seed=42, persist=False):
        self.seed = seed
        self.persist = persist
        self.controller = AdaptiveController(seed=seed, use_nn=True, owner='expedition',
                                             persist=persist, initial_challenge=0.35)
        self.last = None

    def vector(self, m):
        fs = {**self.controller.profile.features(), **map_features(m)}
        return [fs[f] for f in FEATURES]

    def predict(self, m, use_nn=None):
        c = self.controller
        use_nn = c.use_nn if use_nn is None else use_nn
        if use_nn and c.model is not None:
            return float(c.adapter.predict(c.model, self.vector(m)))
        return float(rule_predict(c.profile, m))

    def next_map(self, n_candidates=4, forward_checking=True, arc_consistency=True) -> MapChoice:
        c = self.controller
        options = []
        for k in range(n_candidates):
            target = max(0.05, min(0.98, c.challenge + (k - (n_candidates - 1) / 2) * 0.12))
            seed = self.seed + k * 101 + len(c.profile.attempts) * 131
            result = generate_map(seed=seed, target=target, forward_checking=forward_checking,
                                  arc_consistency=arc_consistency)
            if result.map is None:
                continue
            p = self.predict(result.map)
            score = 1.4 * abs(result.map.difficulty - c.challenge) + 0.22 * abs(TARGET_WIN - p)
            options.append(MapChoice(result.map, p, score, result))
        if not options:
            raise RuntimeError('No reachable candidate map was generated')
        options.sort(key=lambda o: o.score)
        choice = options[0]
        choice.candidates = options
        self.seed += 1009
        c.last_selection = {'level_seed': choice.map.seed, 'features': self.vector(choice.map)}
        self.last = choice
        return choice

    def record(self, won, elapsed, shots, hits, progress):
        """Shots and hits fill the profile's 'jumps' and 'landings' slots, so
        'jump_accuracy' reads as shooting accuracy in this mode."""
        if self.last is None:
            raise RuntimeError('record() needs a map chosen by next_map()')
        return self.controller.record(self.last.map, won, elapsed, shots, hits, progress,
                                      save=self.persist, origin='human',
                                      extra={'mode': 'expedition'})


class Expedition:
    """One run through one map. Call update(dt, ...) every frame."""

    def __init__(self, m: ExpeditionMap, seed=42, spawning=True):
        self.map = m
        self.rng = Random(seed)
        self.spawning = spawning
        sx, sy = m.start
        self.player = Player(sx + 0.5, sy + 0.5)
        self.aliens: list[Alien] = []
        self.bullets: list[Bullet] = []
        self.pickups = [Pickup('ammo' if m.tile(x, y) == 'a' else 'health', x + 0.5, y + 0.5)
                        for y in range(m.height) for x in range(m.width) if m.tile(x, y) in 'ah']
        self.nests = [Nest(x, y, timer=1.0 + self.rng.random() * 2) for x, y in m.nests]
        self.kills = self.shots = self.hits = 0
        self.time = 0.0
        self.won = self.dead = False
        self.field = {}
        self.field_age = FIELD_REFRESH
        self.acid_tick = 0.0
        self.exit_field = bfs_distance_field(m.passable, m.exit, m.width, m.height)
        self.route_length = max(1, self.exit_field.get(m.start, 1))
        self._solid = {(x, y) for y in range(m.height) for x in range(m.width) if not m.passable(x, y)}
        if spawning:
            self._place_dormant_aliens()

    # ---- geometry -----------------------------------------------------------
    def solid(self, x, y):
        return (x, y) in self._solid or not (0 <= x < self.map.width and 0 <= y < self.map.height)

    def blocked(self, x, y, r=BODY_RADIUS):
        for ty in range(floor(y - r), floor(y + r) + 1):
            for tx in range(floor(x - r), floor(x + r) + 1):
                if self.solid(tx, ty):
                    return True
        return False

    def move_body(self, body, dx, dy):
        if dx and not self.blocked(body.x + dx, body.y):
            body.x += dx
        if dy and not self.blocked(body.x, body.y + dy):
            body.y += dy

    def tile_of(self, body):
        return int(floor(body.x)), int(floor(body.y))

    def progress(self):
        d = self.exit_field.get(self.tile_of(self.player), self.route_length)
        return max(0.0, min(1.0, 1 - d / self.route_length))

    # ---- spawning -----------------------------------------------------------
    def spawn(self, kind, x, y, active=True, nest=None):
        spec = ALIENS[kind]
        alien = Alien(kind, x, y, spec['hp'], spec['algorithm'], spec['speed'], spec['damage'],
                      spec['radius'], active=active, nest=nest,
                      replan_in=self.rng.random() * spec['replan'])
        self.aliens.append(alien)
        return alien

    def _place_dormant_aliens(self):
        m = self.map
        for index, room in enumerate(m.rooms):
            if room.template.kind in ('start', 'exit', 'supply'):
                continue
            count = self.rng.randint(0, 1 + room.template.difficulty // 2)
            r, c = divmod(index, m.cols)
            cells = [(c * 10 + x, r * 10 + y) for x in range(2, 8) for y in range(2, 8)
                     if m.passable(c * 10 + x, r * 10 + y) and m.tile(c * 10 + x, r * 10 + y) != '~']
            for cell in self.rng.sample(cells, min(count, len(cells))):
                kind = self.rng.choices(('glob', 'skitter', 'brute'),
                                        (0.55, 0.3, 0.15 + 0.3 * m.difficulty))[0]
                self.spawn(kind, cell[0] + 0.5, cell[1] + 0.5, active=False)

    def _nest_spawn(self, index, nest, events):
        m = self.map
        around = [(nest.x + dx, nest.y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if (dx or dy) and m.passable(nest.x + dx, nest.y + dy)]
        if not around:
            return
        x, y = self.rng.choice(around)
        kind = self.rng.choices(('glob', 'skitter', 'brute'), (0.5, 0.35, 0.1 + 0.25 * m.difficulty))[0]
        self.spawn(kind, x + 0.5, y + 0.5, nest=index)
        nest.alive += 1
        events.append(('spawn', x + 0.5, y + 0.5))

    # ---- AI ---------------------------------------------------------------------
    def lab_routes(self, max_expansions=20000):
        """The four searches from the player to the exit, for the AI Lab."""
        start = self.tile_of(self.player)
        return {algo: solve_grid(self.map.passable, start, self.map.exit, algo, max_expansions)
                for algo in ('bfs', 'dfs', 'astar', 'greedy')}

    def _steer(self, alien, dt):
        p = self.player
        dist = hypot(p.x - alien.x, p.y - alien.y)
        if dist < 1.1:
            tx, ty = p.x, p.y
        elif alien.algorithm == 'bfs':
            # Glob: step to the neighbouring tile closest to the player on the BFS field.
            ax, ay = self.tile_of(alien)
            best = None
            for nx, ny in ((ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1), (ax, ay)):
                d = self.field.get((nx, ny))
                if d is not None and (best is None or d < best[0]):
                    best = (d, nx, ny)
            if best is None:
                return
            tx, ty = best[1] + 0.5, best[2] + 0.5
        else:
            alien.replan_in -= dt
            if alien.replan_in <= 0 or not alien.path:
                alien.replan_in = ALIENS[alien.kind]['replan']
                budget = 1500 if alien.algorithm == 'greedy' else 2500
                result = solve_grid(self.map.passable, self.tile_of(alien), self.tile_of(p),
                                    alien.algorithm, budget, trace_limit=0)
                alien.last_search = result
                alien.path = list(result.path[1:]) if result.found else []
            while alien.path and hypot(alien.path[0][0] + 0.5 - alien.x, alien.path[0][1] + 0.5 - alien.y) < 0.15:
                alien.path.pop(0)
            if not alien.path:
                return
            tx, ty = alien.path[0][0] + 0.5, alien.path[0][1] + 0.5
        dx, dy = tx - alien.x, ty - alien.y
        length = hypot(dx, dy)
        if length > 1e-6:
            step = min(length, alien.speed * dt)
            alien.heading = (dx / length, dy / length)
            self.move_body(alien, dx / length * step, dy / length * step)

    # ---- main update -----------------------------------------------------------
    def update(self, dt, move=(0.0, 0.0), aim=(1.0, 0.0), shoot=False, reload=False):
        if self.won or self.dead:
            return []
        events = []
        self.time += dt
        p = self.player
        # player movement
        mx, my = move
        length = hypot(mx, my)
        p.moving = length > 1e-6
        if p.moving:
            scale = PLAYER_SPEED * dt / max(1.0, length) if length > 1 else PLAYER_SPEED * dt
            self.move_body(p, mx * scale, my * scale)
        ax, ay = aim
        alen = hypot(ax, ay)
        if alen > 1e-6:
            p.aim = (ax / alen, ay / alen)
        # weapon
        p.fire_cd = max(0.0, p.fire_cd - dt)
        if p.reload_left > 0:
            p.reload_left -= dt
            if p.reload_left <= 0:
                take = min(MAG_SIZE - p.mag, p.ammo)
                p.mag += take; p.ammo -= take
                events.append(('reloaded', p.x, p.y))
        elif (reload and p.mag < MAG_SIZE and p.ammo > 0) or (shoot and p.mag == 0 and p.ammo > 0):
            p.reload_left = RELOAD_TIME
            events.append(('reload', p.x, p.y))
        elif shoot and p.mag > 0 and p.fire_cd <= 0:
            p.fire_cd = FIRE_DELAY
            p.mag -= 1; self.shots += 1
            spread = (self.rng.random() - 0.5) * 0.06
            dx, dy = p.aim[0] - p.aim[1] * spread, p.aim[1] + p.aim[0] * spread
            self.bullets.append(Bullet(p.x + dx * 0.4, p.y + dy * 0.4, dx * BULLET_SPEED, dy * BULLET_SPEED))
            events.append(('shot', p.x + dx * 0.5, p.y + dy * 0.5))
        # bullets
        alive = []
        for b in self.bullets:
            b.life -= dt
            steps = max(1, ceil(hypot(b.dx, b.dy) * dt / 0.2))
            hit = False
            for _ in range(steps):
                b.x += b.dx * dt / steps; b.y += b.dy * dt / steps
                tx, ty = int(floor(b.x)), int(floor(b.y))
                if self.solid(tx, ty):
                    for index, nest in enumerate(self.nests):
                        if (nest.x, nest.y) == (tx, ty) and not nest.dead:
                            nest.hp -= 1; self.hits += 1
                            events.append(('hit', b.x, b.y))
                            if nest.dead:
                                events.append(('nest_down', nest.x + 0.5, nest.y + 0.5))
                            break
                    else:
                        events.append(('spark', b.x, b.y))
                    hit = True
                    break
                for alien in self.aliens:
                    if hypot(alien.x - b.x, alien.y - b.y) < alien.radius + 0.08:
                        alien.hp -= 1; alien.hit_t = 0.12; alien.active = True
                        self.hits += 1
                        events.append(('hit', b.x, b.y))
                        hit = True
                        break
                if hit:
                    break
            if not hit and b.life > 0:
                alive.append(b)
        self.bullets = alive
        survivors = []
        for alien in self.aliens:
            if alien.hp <= 0:
                self.kills += 1
                if alien.nest is not None:
                    self.nests[alien.nest].alive -= 1
                events.append(('kill', alien.x, alien.y, alien.kind))
            else:
                survivors.append(alien)
        self.aliens = survivors
        # shared BFS field for the globs (and for waking dormant aliens)
        self.field_age += dt
        if self.field_age >= FIELD_REFRESH:
            self.field_age = 0.0
            self.field = bfs_distance_field(self.map.passable, self.tile_of(p), self.map.width, self.map.height)
        # aliens
        for alien in self.aliens:
            alien.hit_t = max(0.0, alien.hit_t - dt)
            alien.cooldown = max(0.0, alien.cooldown - dt)
            if not alien.active:
                d = self.field.get(self.tile_of(alien))
                if d is not None and d <= AGGRO_STEPS:
                    alien.active = True
                    events.append(('alert', alien.x, alien.y))
                continue
            if alien.frozen:
                continue
            self._steer(alien, dt)
            if hypot(p.x - alien.x, p.y - alien.y) < alien.radius + PLAYER_RADIUS + 0.12 and alien.cooldown <= 0:
                alien.cooldown = ATTACK_COOLDOWN
                p.hp -= alien.damage
                events.append(('hurt', p.x, p.y))
        # keep aliens from stacking on one spot
        for i, a in enumerate(self.aliens):
            for b in self.aliens[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = hypot(dx, dy)
                gap = a.radius + b.radius
                if 1e-6 < d < gap:
                    push = (gap - d) / 2 / d
                    self.move_body(a, -dx * push, -dy * push)
                    self.move_body(b, dx * push, dy * push)
        # nests
        if self.spawning:
            for index, nest in enumerate(self.nests):
                if nest.dead or nest.alive >= NEST_MAX_ALIVE:
                    continue
                if hypot(nest.x + 0.5 - p.x, nest.y + 0.5 - p.y) > NEST_RANGE:
                    continue
                nest.timer -= dt
                if nest.timer <= 0:
                    nest.timer = (3.2 - 1.6 * self.map.difficulty) * (0.7 + self.rng.random() * 0.6)
                    self._nest_spawn(index, nest, events)
        # pickups and acid
        for item in self.pickups:
            if not item.taken and hypot(item.x - p.x, item.y - p.y) < 0.6:
                item.taken = True
                if item.kind == 'ammo':
                    p.ammo += 30
                else:
                    p.hp = min(100.0, p.hp + 35)
                events.append(('pickup', item.x, item.y, item.kind))
        if self.map.tile(*self.tile_of(p)) == '~':
            p.hp -= ACID_DPS * dt
            self.acid_tick -= dt
            if self.acid_tick <= 0:
                self.acid_tick = 0.4
                events.append(('acid', p.x, p.y))
        # outcome
        ex, ey = self.map.exit
        if hypot(ex + 0.5 - p.x, ey + 0.5 - p.y) < 0.9:
            self.won = True
            events.append(('win', p.x, p.y))
        elif p.hp <= 0:
            p.hp = 0.0
            self.dead = True
            events.append(('dead', p.x, p.y))
        return events
