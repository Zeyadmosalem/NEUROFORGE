"""Seeded, *unannounced* environment changes for live search-driven agents.

World events are not planned in advance by an agent. At each event boundary the
simulator changes the observed world, verifies a path from the CURRENT state,
and asks that agent's own search algorithm to plan again. In other words the
algorithm cannot memorize a path through the entire run.

Events are reproducible given each agent's RNG seed. The verifier rejects changes
that immediately kill the actor or leave it unable to complete the level within
the search budget. This is a fully observed replanning problem, NOT reinforcement
learning or a partially observed environment.
"""
from dataclasses import dataclass, replace
from random import Random
from game.physics import Level, State, MovingEnemy, is_dead
from search.algorithms import SearchResult, solve


@dataclass(frozen=True)
class WorldEvent:
    name: str
    description: str
    level: Level
    plan: SearchResult
    x: int | None = None


def _open_ground(level: Level, x: int) -> bool:
    base = level.height - 2
    return (2 < x < level.goal_x-2 and
            level.tile(x, base) == '#' and
            level.tile(x, base-1) == '.' and
            not any(g.x == x for g in level.gates) and
            not any(e.x <= x <= e.x + e.travel for e in level.enemies))


def _ahead(level: Level, state: State, rng: Random, lead: int = 4) -> list[int]:
    xs = list(range(state.x+lead, min(level.goal_x-2, state.x+19)))
    rng.shuffle(xs)
    return xs


def _propose(level: Level, state: State, rng: Random, kind: str):
    """Build an unverified candidate; no midair teleporting of the player."""
    base = level.height - 2
    ahead = _ahead(level, state, rng)
    if kind == 'spike_bloom':
        for x in ahead:
            if _open_ground(level, x):
                rows = list(level.rows)
                row = list(rows[base-1]);row[x] = '^';rows[base-1]=''.join(row)
                return replace(level, rows=tuple(rows)), x, f'NEW SPIKE AT X={x}'
    elif kind == 'bridge_rises':
        for x in ahead:
            if (level.tile(x,base)=='.' and level.tile(x,base-1)=='.'
                and level.tile(x,base+1)=='.'):
                rows = list(level.rows)
                for y in (base,base+1):
                    row = list(rows[y]);row[x]='#';rows[y]=''.join(row)
                return replace(level, rows=tuple(rows)), x, f'BRIDGE RISES AT X={x}'
    elif kind == 'new_patrol':
        for x in ahead:
            if all(_open_ground(level, px) for px in range(x, x+3)):
                enemy = MovingEnemy(x, base-1, travel=2,
                                    offset=rng.randrange(8), kind='surprise')
                return replace(level, enemies=level.enemies+(enemy,)),x,f'NEW PATROL AT X={x}'
    elif kind == 'patrol_switch':
        indices=[i for i,e in enumerate(level.enemies) if e.x > state.x+2]
        rng.shuffle(indices)
        if indices:
            idx=indices[0];enemies=list(level.enemies);e=enemies[idx]
            enemies[idx]=replace(e,offset=(e.offset+rng.randint(2,5))%8)
            return replace(level,enemies=tuple(enemies)),e.x,'PATROL CHANGES TIMING'
    elif kind == 'gate_shuffle':
        indices=[i for i,g in enumerate(level.gates) if g.x > state.x+2]
        rng.shuffle(indices)
        if indices:
            idx=indices[0];gates=list(level.gates);g=gates[idx]
            gates[idx]=replace(g,offset=(g.offset+rng.randint(2,5))%8)
            return replace(level,gates=tuple(gates)),g.x,'GATE CYCLE SHIFTED'
    elif kind == 'platform_shuffle':
        if level.support_platform(state.x,state.y,state.phase) is not None:
            return None
        indices=[i for i,p in enumerate(level.platforms) if p.x > state.x+2]
        rng.shuffle(indices)
        if indices:
            idx=indices[0];platforms=list(level.platforms);p=platforms[idx]
            platforms[idx]=replace(p,offset=(p.offset+rng.randint(2,5))%8)
            return replace(level,platforms=tuple(platforms)),p.x,'PLATFORM TIMING SHIFTED'
    elif kind == 'coin_shower':
        candidates=[x for x in ahead if level.tile(x,base-2)=='.']
        if candidates:
            rows=list(level.rows);row=list(rows[base-2]);chosen=candidates[:3]
            for x in chosen:row[x]='o'
            rows[base-2]=''.join(row)
            return replace(level,rows=tuple(rows)),chosen[0],'SURPRISE COINS APPEAR'
    return None


def trigger_surprise(level: Level, state: State, algorithm: str, rng: Random,
                     kinds: tuple[str,...] | None = None,
                     max_expansions: int = 11000) -> WorldEvent | None:
    """Make one safe but unforeseen change and immediately recalculate a route.

    One proposal per type caps computational cost. The returned path starts from
    the observed CURRENT state. No future event schedules are exposed to solve().
    """
    if level.complete(state) or is_dead(level,state):return None
    possibilities=list(kinds or (
        'spike_bloom','new_patrol','patrol_switch','gate_shuffle',
        'platform_shuffle','bridge_rises','coin_shower'))
    rng.shuffle(possibilities)
    for kind in possibilities:
        proposed=_propose(level,state,rng,kind)
        if proposed is None:continue
        changed,x,description=proposed
        if is_dead(changed,state):continue
        # Search with the actor's OWN algorithm, rather than secretly replacing
        # BFS/DFS/Greedy with A*. We reject unreachable or over-budget edits.
        plan=solve(changed,algorithm,start=state,max_expansions=max_expansions)
        if plan.found:
            return WorldEvent(kind,description,changed,plan,x)
    return None
