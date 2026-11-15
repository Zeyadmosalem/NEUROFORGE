"""Four independent graph-search strategies on exactly the same physics graph.
A* correctly RELAXES discovered states and discards stale frontier records.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter
from game.physics import Level, State, ACTIONS, step

_UNKNOWN=object()  # "transition not memoized yet" (None already means "lethal")

@dataclass
class SearchResult:
    algorithm: str
    found: bool
    actions: list[str]
    path: list[State]
    expanded: int
    generated: int
    frontier_max: int
    elapsed_ms: float
    trace: list[State]
    reason: str = ''

    @property
    def cost(self):return len(self.actions) if self.found else None


def heuristic(s: State, level: Level) -> int:
    """Admissible: a carried player can move at most two horizontal tiles/tick."""
    return (max(0,level.goal_x-s.x) + (1 if level.platforms else 0)) // (2 if level.platforms else 1)


def reconstruct(end: State, parents: dict[State, tuple[State, str]]) -> tuple[list[str],list[State]]:
    path=[end];actions=[]
    while path[-1] in parents:
        prev,act=parents[path[-1]];actions.append(act);path.append(prev)
    path.reverse();actions.reverse()
    return actions,path


def solve(level: Level, algorithm: str = 'astar', max_expansions: int = 18000,
          trace_limit: int = 3500, start: State | None = None) -> SearchResult:
    if algorithm not in {'bfs','dfs','astar','greedy'}:
        raise ValueError('Select bfs, dfs, astar or greedy')
    start=level.start if start is None else start;begun=perf_counter()
    parents={};seen={start};trace=[];expanded=0;generated=1;frontier_max=1
    # Per-search constants are hoisted out of the hot loop. The traversal order,
    # tie-breaking and every counter are unchanged.
    bfs=algorithm=='bfs';dfs=algorithm=='dfs';astar=algorithm=='astar'
    uninformed=bfs or dfs
    goal_x=level.goal_x;height=level.height;carried=bool(level.platforms)
    # DFS intentionally explores the JUMP-RIGHT branch first (LIFO),
    # unlike the right-biased Greedy priority. This reveals distinct
    # traversal paths rather than making DFS and Greedy look identical.
    order=('W','L','JL','J','R','JR') if dfs else ACTIONS
    # step() memoizes transitions per level; reading its tables directly skips
    # a function call for every transition that is already known.
    transitions=[(action,level._step_cache[action]) for action in order]
    infinity=float('inf')
    # Greedy: h, A*: g+h, and FIFO/LIFO for BFS/DFS. Equal f uses FIFO order.
    if uninformed:
        frontier=deque([start]) if bfs else [start]
    else:
        frontier=[(heuristic(start,level),0,0,start)]
        best_g={start:0};sequence=0
    while frontier and expanded < max_expansions:
        if bfs:cur=frontier.popleft()
        elif dfs:cur=frontier.pop()
        else:
            _,_,g,cur=heappop(frontier)
            if astar and g!=best_g[cur]:
                continue  # obsolete heap record after a better path was found
        expanded+=1
        if len(trace)<trace_limit:trace.append(cur)
        if cur.x>=goal_x and 0<=cur.y<=height:  # == level.complete(cur)
            actions,path=reconstruct(cur,parents)
            return SearchResult(algorithm,True,actions,path,expanded,generated,
                                frontier_max,round((perf_counter()-begun)*1000,3),trace)
        if not uninformed and not astar:g=best_g[cur]
        for action,known in transitions:
            nxt=known.get(cur,_UNKNOWN)
            if nxt is _UNKNOWN:nxt=step(level,cur,action)
            if nxt is None or nxt==cur:continue
            if astar:
                new_g=g+1
                if new_g>=best_g.get(nxt,infinity):continue
                best_g[nxt]=new_g
                parents[nxt]=(cur,action);generated+=1;sequence+=1
                # inlined heuristic(nxt, level)
                remaining=goal_x-nxt.x
                if remaining<0:remaining=0
                heappush(frontier,(new_g+((remaining+1)//2 if carried else remaining),
                                   sequence,new_g,nxt))
            else:
                if nxt in seen:continue
                seen.add(nxt);parents[nxt]=(cur,action);generated+=1
                if uninformed:
                    frontier.append(nxt)
                else:
                    new_g=g+1;best_g[nxt]=new_g;sequence+=1
                    remaining=goal_x-nxt.x
                    if remaining<0:remaining=0
                    heappush(frontier,((remaining+1)//2 if carried else remaining,
                                       sequence,new_g,nxt))
        if len(frontier)>frontier_max:frontier_max=len(frontier)
    return SearchResult(algorithm,False,[],[],expanded,generated,frontier_max,
                        round((perf_counter()-begun)*1000,3),trace,
                        'expansion limit' if frontier else 'No solution')
