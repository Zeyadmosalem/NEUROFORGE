"""Tests for stochastic, reactive world changes (no path memorization)."""
from random import Random
from game.physics import Level, is_dead, replay
from search.algorithms import solve
from integration.world_events import trigger_surprise
from integration.arena import AgentRunner


def flat(width=20):
    return Level(tuple(['.'*width for _ in range(10)] + ['#'*width, '#'*width]))


def test_surprise_is_seed_reproducible_and_solvable():
    level=flat()
    a=trigger_surprise(level,level.start,'astar',Random(41),kinds=('spike_bloom',))
    b=trigger_surprise(level,level.start,'astar',Random(41),kinds=('spike_bloom',))
    assert a and b
    assert a.name=='spike_bloom'
    assert a.level.rows==b.level.rows and a.plan.actions==b.plan.actions
    assert not is_dead(a.level,level.start)
    path,won=replay(a.level,a.plan.actions)
    assert won and path==a.plan.path


def test_surprise_forces_a_new_plan_not_the_old_route():
    level=flat()
    old=solve(level,'astar')
    assert set(old.actions)=={'R'}
    change=trigger_surprise(level,level.start,'astar',Random(4),kinds=('spike_bloom',))
    assert change and change.plan.found
    assert change.plan.actions!=old.actions
    assert any(action.startswith('J') for action in change.plan.actions)
    assert change.plan.path[0]==level.start


def test_every_search_plans_after_unexpected_world_change():
    level=flat(15)
    for algo in ('bfs','dfs','astar','greedy'):
        change=trigger_surprise(level,level.start,algo,Random(4),
                                kinds=('spike_bloom',),max_expansions=18000)
        assert change and change.plan.algorithm==algo,algo
        assert change.plan.path[0]==level.start
        _,won=replay(change.level,change.plan.actions)
        assert won


def test_actual_agent_observes_and_replans_mid_episode():
    agent=AgentRunner('astar',seed=5,persist=False,motor_noise=False,surprises=False)
    level=flat(20)
    agent.choice.result.level=level
    agent.state=level.start
    agent.plan=solve(level,agent.algorithm)
    agent.plan_index=0
    assert agent.inject_surprise(kinds=('spike_bloom',))
    assert agent.total_surprises==1 and agent.replans==1
    assert agent.world_events[0]['tick']==0
    assert agent.plan.path[0]==agent.state
    assert agent.plan.algorithm=='astar'
    assert agent.level.rows!=level.rows


def test_timed_gate_requires_acting_at_a_safe_phase():
    from game.physics import TimedGate
    base=flat(16)
    level=Level(base.rows,gates=(TimedGate(4,1,9,offset=0,active_ticks=6),))
    for algo in ('bfs','astar','greedy','dfs'):
        plan=solve(level,algo,max_expansions=14000)
        assert plan.found,algo
        assert all(s.phase not in range(6) for s in plan.path if s.x==4)
        assert len(plan.path)==len(plan.actions)+1


def test_moving_platform_carries_idle_actor():
    from game.physics import MovingPlatform, State, step
    level=Level(flat(18).rows,platforms=(MovingPlatform(x=3,y=8,width=2,travel=2),))
    start=State(3,8,0,0)
    advanced=step(level,start,'W')
    assert advanced is not None
    assert advanced.x==4 and advanced.y==8 and advanced.phase==1
