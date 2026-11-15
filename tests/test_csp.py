import pytest
from csp.solver import ChunkCSP
from csp.generator import generate,compile_chunks
from search.algorithms import solve

@pytest.mark.parametrize('fc,ac',[(False,False),(True,False),(False,True),(True,True)])
def test_solver_modes(fc,ac):
    p=ChunkCSP(6,0.4,seed=17)
    r=p.solve(fc,ac)
    assert r.chunks is not None,r.reason
    assert r.stats.assignments>0
    assert len(r.chunks)==6
    assert p.budget_lo<=sum(c.difficulty for c in r.chunks)<=p.budget_hi
    for i in range(5):
        assert p.compatible(i,r.chunks[i],i+1,r.chunks[i+1])


def test_csp_impossible():
    r=ChunkCSP(5,0.5,impossible=True).solve(True,True)
    assert r.chunks is None and r.reason=='No solution'


def test_generator_is_search_verified():
    r=generate(seed=33,target=0.45)
    assert r.level is not None,r.reason
    assert r.search.found
    assert solve(r.level,'bfs').found


def test_seed_reproducible():
    a=generate(seed=123); b=generate(seed=123)
    assert a.level==b.level


def test_difficulty_changes_level():
    a=generate(seed=20,target=0.1); b=generate(seed=20,target=0.9)
    assert a.level and b.level
    assert sum(c!='meadow' for c in a.level.chunks)>=0
    from csp.solver import BY_NAME
    assert sum(BY_NAME[n].difficulty for n in b.level.chunks) > sum(BY_NAME[n].difficulty for n in a.level.chunks)
