"""CSP-based level compiler + time-aware physics A* verification.

Dynamic hazard types are selected by the CSP as named chunk domains and
instantiated deterministically from the chosen chunks. Failed candidates are
rejected; no world can silently bypass the search verifier.
"""
from dataclasses import dataclass
from game.physics import Level, MovingPlatform, MovingEnemy, TimedGate
from search.algorithms import solve, SearchResult
from csp.solver import Chunk, ChunkCSP, CSPResult

W=8; H=12


def compile_chunks(chunks:list[Chunk],seed=0,target=0.5)->Level:
    width=len(chunks)*W
    rows=[['.' for _ in range(width)] for _ in range(H)]
    base=H-2
    moving=[];enemies=[];gates=[]
    for k,chunk in enumerate(chunks):
        for i,c in enumerate(chunk.floor):
            x=k*W+i
            if c=='#':rows[base][x]='#'; rows[base+1][x]='#'
        for i in chunk.spikes:
            x=k*W+i
            if rows[base][x]=='#': rows[base-1][x]='^'
        for i in chunk.coins:
            x=k*W+i
            if rows[base-2][x]=='.':rows[base-2][x]='o'
        for i,raise_by in chunk.shelves:
            x=k*W+i; y=base-raise_by
            if rows[y][x]=='.':rows[y][x]='#'
        for i,height in chunk.blockers:
            x=k*W+i
            for dy in range(1,height+1):
                y=base-dy
                if rows[y][x]=='.':rows[y][x]='#'
        # Every moving entity stays strictly within its own chunk.
        # Phase offsets differ across worlds, changing which waiting strategy
        # succeeds, but the objects all share the same short 8-tick cycle.
        phase=(seed + k*3) % 8
        if chunk.name=='ferry_bridge':
            moving.append(MovingPlatform(k*W+2,base-2,2,2,'x',phase))
        elif chunk.name=='shuttle_pit':
            moving.append(MovingPlatform(k*W+1,base-2,2,3,'x',phase))
        elif chunk.name=='patrol_lane':
            enemies.append(MovingEnemy(k*W+2,base-1,3,phase,'walker'))
        elif chunk.name=='hunter_duo':
            enemies.extend((MovingEnemy(k*W+1,base-1,2,phase,'walker'),
                            MovingEnemy(k*W+4,base-1,2,(phase+4)%8,'walker')))
        elif chunk.name=='pulse_gate':
            gates.append(TimedGate(k*W+4,1,base-1,phase,4))
        elif chunk.name=='laser_hall':
            gates.append(TimedGate(k*W+5,1,base-1,phase,5))
    for k,chunk in enumerate(chunks):
        for i,height in chunk.air_spikes:
            x=k*W+i; y=base-height
            if 0<=y<H and rows[y][x]=='.':rows[y][x]='^'
    actual = min(1.0, sum(c.difficulty for c in chunks) / max(1., 3.4*(len(chunks)-2)))
    return Level(tuple(''.join(r) for r in rows),seed,actual,tuple(c.name for c in chunks),
                 tuple(moving),tuple(enemies),tuple(gates))


@dataclass
class GenerationResult:
    level: Level | None
    csp: CSPResult | None
    search: SearchResult | None
    candidates: int
    reason: str=''


def generate(seed=42,target=0.5,n_chunks=6,forward_checking=True,
             arc_consistency=True,attempts=12,impossible=False)->GenerationResult:
    last_csp=None; last_search=None
    for attempt in range(attempts):
        problem=ChunkCSP(n_chunks,target,seed+attempt*997,impossible)
        last_csp=problem.solve(forward_checking,arc_consistency)
        if last_csp.chunks is None:
            return GenerationResult(None,last_csp,None,attempt+1,'No solution')
        level=compile_chunks(last_csp.chunks,seed+attempt,target)
        last_search=solve(level,'astar',max_expansions=18000)
        if last_search.found:
            return GenerationResult(level,last_csp,last_search,attempt+1)
    return GenerationResult(None,last_csp,last_search,attempts,
                            'No solution: time-aware search rejected every candidate')
