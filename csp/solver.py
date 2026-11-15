"""Actual finite-domain constraint solving with independent FC / AC-3 toggles.
AC-3 handles binary adjacency constraints; backtracking also handles global budgets.
"""
from dataclasses import dataclass
from time import perf_counter

@dataclass(frozen=True)
class Chunk:
    name: str
    difficulty: int
    kind: str
    floor: str
    spikes: tuple[int,...] = ()
    coins: tuple[int,...] = ()
    shelves: tuple[tuple[int,int],...] = ()  # (relative x, height above ground)
    blockers: tuple[tuple[int,int],...] = ()  # (relative x, wall height in tiles)
    air_spikes: tuple[tuple[int,int],...] = ()  # deadly airborne crystals (x, height)

CHUNKS: tuple[Chunk,...] = (
    # Calm sections are not all clones: quiet recovery sections make pacing visible.
    Chunk('meadow',0,'safe','########',coins=(2,4,6)),
    Chunk('coins',0,'safe','########',coins=(1,2,3,5,6)),
    Chunk('sanctuary',0,'safe','########',coins=(2,5)),
    Chunk('hops',1,'platform','########',coins=(3,5),shelves=((3,3),(4,3))),
    Chunk('gap1',1,'gap','###.####',coins=(4,5)),
    Chunk('gap2',2,'gap','##..####',coins=(4,5)),
    Chunk('gap3',3,'gap','##...###',coins=(5,6)),
    Chunk('gap4',4,'gap','##....##',coins=(6,)),
    Chunk('gap5',5,'gap','#.....##',coins=(6,)),
    Chunk('thorns',2,'hazard','########',spikes=(3,),coins=(5,)),
    Chunk('twin_thorns',3,'hazard','########',spikes=(2,4),coins=(6,)),
    Chunk('spike_run',4,'hazard','########',spikes=(2,3,4),coins=(6,)),
    Chunk('criss_cross',5,'hazard','########',spikes=(1,3,5),coins=(6,)),
    Chunk('pinnacle',2,'platform','########',shelves=((2,2),(3,2),(5,4)),coins=(3,6)),
    Chunk('stairway',3,'platform','########',shelves=((2,2),(3,3),(4,4),(5,3)),coins=(3,6)),
    Chunk('zigzag',4,'platform','########',shelves=((1,2),(3,4),(5,2),(6,4)),coins=(6,)),
    Chunk('springboard',2,'platform','########',shelves=((3,4),(4,4)),coins=(1,6)),
    Chunk('broken_bridge',4,'mixed','##..##..',coins=(4,)),
    Chunk('split_path',3,'mixed','##...###',shelves=((3,3),(4,3)),coins=(4,)),
    Chunk('razors_edge',5,'mixed','##..####',spikes=(5,6),coins=(3,)),
    Chunk('gatekeeper',2,'wall','########',blockers=((4,1),),coins=(6,)),
    Chunk('higher_wall',3,'wall','########',blockers=((4,2),),coins=(6,)),
    Chunk('twin_gates',4,'wall','########',blockers=((2,1),(5,2)),coins=(6,)),
    Chunk('gauntlet',5,'mixed','########',spikes=(3,6),blockers=((2,2),(5,1))),
    Chunk('edge_jump',5,'mixed','###...##',blockers=((1,2),),coins=(6,)),
    Chunk('sentry_corridor',4,'mixed','########',spikes=(2,5),blockers=((4,2),)),
    Chunk('needle_arch',3,'mixed','########',coins=(6,),air_spikes=((3,3),(4,3))),
    Chunk('briar_canopy',4,'mixed','########',spikes=(2,6),air_spikes=((3,4),(5,3))),
    Chunk('pitfall',4,'mixed','##...###',shelves=((3,4),(5,3)),air_spikes=((4,4),)),
    Chunk('death_gate',5,'mixed','###..###',spikes=(6,),air_spikes=((4,3),(5,3))),
    Chunk('jagged_stair',4,'mixed','##.#####',blockers=((5,2),),air_spikes=((3,4),(6,3))),
    Chunk('sky_saw',5,'mixed','##...###',shelves=((2,3),(5,3)),air_spikes=((3,4),(4,4))),
    # Time-dependent chunks: the solver decides whether to include these,
    # while the physics/search verifier checks if their timed solution exists.
    Chunk('ferry_bridge',4,'moving','##....##',coins=(3,6)),
    Chunk('shuttle_pit',5,'moving','##.....#',coins=(3,6)),
    Chunk('patrol_lane',3,'enemy','########',coins=(2,6)),
    Chunk('hunter_duo',5,'enemy','########',coins=(4,)),
    Chunk('pulse_gate',3,'timed','########',coins=(5,)),
    Chunk('laser_hall',5,'timed','########',coins=(6,),blockers=((2,1),)),
)

BY_NAME={c.name:c for c in CHUNKS}

@dataclass
class CSPStats:
    assignments: int=0
    backtracks: int=0
    pruned: int=0
    constraint_checks: int=0
    arc_revisions: int=0
    elapsed_ms: float=0.0

@dataclass
class CSPResult:
    chunks: list[Chunk] | None
    stats: CSPStats
    reason: str = ''

class ChunkCSP:
    def __init__(self,n_chunks=7,target=0.5, seed=42, impossible=False):
        if n_chunks<3: raise ValueError('Need >=3 chunks')
        self.n=n_chunks; self.seed=seed; self.target=max(0.,min(1.,float(target)))
        self.impossible=impossible
        self.stats=CSPStats()
        # Ends are protected: the player's start and goal must be safe.
        mid=max(1,n_chunks-2)
        self.budget=round(self.target * (mid*3.4))
        self.budget_lo=max(0,self.budget-1)
        self.budget_hi=self.budget+1
        self.domains={i: list(CHUNKS) if 0<i<n_chunks-1 else
                      [BY_NAME['meadow']] if i==0 else [BY_NAME['coins']]
                      for i in range(n_chunks)}
        if impossible:
            # Contradictory domain test: mandatory first element must also be gap3.
            self.domains[0] = [BY_NAME['gap3']]

    def compatible(self,i:int,a:Chunk,j:int,b:Chunk)->bool:
        self.stats.constraint_checks+=1
        if abs(i-j)!=1:return True
        if (i==0 and a.kind!='safe') or (j==0 and b.kind!='safe'):return False
        if a.name==b.name:return False
        if a.difficulty>=4 and b.difficulty>=4:return False
        if a.kind=='gap' and b.kind=='gap' and a.difficulty+b.difficulty>5:return False
        return True

    def feasible_budget(self, assigned:dict[int,Chunk], domains:dict[int,list[Chunk]])->bool:
        assigned_sum=sum(c.difficulty for c in assigned.values())
        unassigned=[i for i in range(self.n) if i not in assigned]
        if not unassigned:
            dynamic=sum(c.kind in ('moving','enemy','timed') for c in assigned.values())
            if self.target>=0.52 and dynamic<1: return False
            if self.target>=0.81 and dynamic<2: return False
            return self.budget_lo<=assigned_sum<=self.budget_hi
        if any(not domains[i] for i in unassigned):return False
        low=assigned_sum+sum(min(c.difficulty for c in domains[i]) for i in unassigned)
        high=assigned_sum+sum(max(c.difficulty for c in domains[i]) for i in unassigned)
        return low<=self.budget_hi and high>=self.budget_lo

    def ac3(self, domains:dict[int,list[Chunk]], arcs=None)->bool:
        from collections import deque
        queue=deque(arcs if arcs is not None else
                    [(i,j) for i in range(self.n) for j in (i-1,i+1) if 0<=j<self.n])
        while queue:
            i,j=queue.popleft(); before=len(domains[i]); self.stats.arc_revisions+=1
            domains[i]=[a for a in domains[i] if any(self.compatible(i,a,j,b) for b in domains[j])]
            self.stats.pruned+=before-len(domains[i])
            if not domains[i]: return False
            if len(domains[i])<before:
                for k in (i-1,i+1):
                    if 0<=k<self.n and k!=j:queue.append((k,i))
        return True

    def solve(self, forward_checking=True, arc_consistency=True) -> CSPResult:
        import random
        started=perf_counter(); rnd=random.Random(self.seed)
        domains={i:list(cs) for i,cs in self.domains.items()}
        if arc_consistency and not self.ac3(domains):
            self.stats.elapsed_ms=round((perf_counter()-started)*1000,3)
            return CSPResult(None,self.stats,'No solution')

        def rec(assigned, domains):
            if len(assigned)==self.n:
                return dict(assigned) if self.feasible_budget(assigned,domains) else None
            # MRV, then left to right to keep adjacencies local.
            i=min((i for i in range(self.n) if i not in assigned),
                  key=lambda j:(len(domains[j]),j))
            values=list(domains[i]); rnd.shuffle(values)
            values.sort(key=lambda c:abs(c.difficulty-(self.budget/max(1,self.n-2))) + rnd.random()*1.7)
            for value in values:
                self.stats.assignments+=1
                if any(not self.compatible(i,value,j,c) for j,c in assigned.items() if abs(i-j)==1):
                    continue
                next_assigned={**assigned,i:value}
                next_domains={j:list(items) for j,items in domains.items()}
                next_domains[i]=[value]
                good=True
                if forward_checking:
                    for j in (i-1,i+1):
                        if j not in assigned and j>=0 and j<self.n:
                            old=len(next_domains[j])
                            next_domains[j]=[b for b in next_domains[j] if self.compatible(i,value,j,b)]
                            self.stats.pruned+=old-len(next_domains[j])
                            if not next_domains[j]:good=False;break
                if good and arc_consistency:
                    good=self.ac3(next_domains,
                        [(j,i) for j in (i-1,i+1) if j>=0 and j<self.n])
                if good and self.feasible_budget(next_assigned,next_domains):
                    found=rec(next_assigned,next_domains)
                    if found is not None:return found
                self.stats.backtracks+=1
            return None
        found=rec({},domains)
        # rec refers to itself through its closure; dropping the name breaks that
        # reference cycle so each solve is freed immediately instead of waiting
        # for (and triggering) a garbage-collector pass.
        rec=None
        self.stats.elapsed_ms=round((perf_counter()-started)*1000,3)
        if found is None:return CSPResult(None,self.stats,'No solution')
        return CSPResult([found[i] for i in range(self.n)],self.stats)
