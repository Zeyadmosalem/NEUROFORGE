"""Explicit rubric matrix: >=5 normal, >=3 edge, >=2 impossible cases."""
import pytest
from game.physics import Level,State,parse_level,step,replay
from search.algorithms import solve
from csp.solver import ChunkCSP
from csp.generator import generate


def flat(w=12):return Level(tuple(['.'*w for _ in range(10)]+['#'*w,'#'*w]))

@pytest.mark.parametrize('seed,difficulty',[(1,.1),(2,.3),(3,.5),(4,.7),(5,.9)])
def test_normal_generated_playable(seed,difficulty):
    gen=generate(seed=seed,target=difficulty)
    assert gen.level is not None
    res=solve(gen.level,'astar')
    assert res.found and replay(gen.level,res.actions)[1]


def test_edge_player_already_at_goal():
    level=flat(2)
    assert solve(level,'bfs').cost==0


def test_edge_obstacle_blocks_horizontal_but_not_jumping():
    rows=list(flat().rows);row=list(rows[9]);row[3]='#';rows[9]=''.join(row)
    level=Level(tuple(rows));start=State(2,10,0)
    result=step(level,start,'R')
    assert result is not None and result.x==2
    assert step(level,start,'JR').y<start.y


def test_edge_repeated_jump_midair_not_extra_impulse():
    level=flat()
    a=step(level,level.start,'JR');b=step(level,a,'JR')
    assert a.vy==-2 and b.vy==-1


def test_impossible_no_bridge_search():
    level=Level(tuple(['.'*25 for _ in range(10)]+['###...................###']*2))
    res=solve(level,'bfs')
    assert res.reason=='No solution' and not res.found


def test_impossible_csp_explicit():
    r=ChunkCSP(5,0.5,impossible=True).solve(False,False)
    assert r.chunks is None and r.reason=='No solution'
