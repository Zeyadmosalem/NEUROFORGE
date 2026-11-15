from game.physics import Level,State,step,replay
from search.algorithms import solve,heuristic


def flat(width=12):
    return Level(tuple(['.'*width for _ in range(10)]+['#'*width,'#'*width]))


def test_all_four_flat():
    level=flat()
    for algo in ('bfs','dfs','astar','greedy'):
        res=solve(level,algo)
        assert res.found,(algo,res.reason)
        assert res.path[0]==level.start
        path,win=replay(level,res.actions)
        assert win and path==res.path
        assert res.expanded>=1 and res.generated>=res.expanded


def test_bfs_astar_optimal():
    level=flat(); bfs=solve(level,'bfs'); astar=solve(level,'astar')
    assert bfs.cost==astar.cost==level.goal_x-level.start.x


def test_missing_bridge_is_impossible():
    lines=['.'*20 for _ in range(10)]+['####.............###','####.............###']
    level=Level(tuple(lines))
    for algo in ('bfs','dfs','astar','greedy'):
        result=solve(level,algo,max_expansions=14000)
        assert not result.found and not result.actions


def test_invalid_algorithm():
    import pytest
    with pytest.raises(ValueError):solve(flat(),'beam')


def test_jump_physics_reused():
    l=flat()
    j=step(l,l.start,'JR')
    assert j and j.y<l.start.y
    assert heuristic(l.start,l)==l.goal_x-1


def test_spike_hazard_rejected():
    a=flat(); rows=list(a.rows); rows[9]=rows[9][:3]+'^'+rows[9][4:]
    l=Level(tuple(rows))
    assert step(l,State(2,10,0),'R') is None
