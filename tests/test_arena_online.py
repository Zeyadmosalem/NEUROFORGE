"""Behavioural tests for actual agent search, new-only updates and amnesia."""
from __future__ import annotations
import numpy as np
from pathlib import Path
from data.make_data import FEATURES
from nn.model import MLP, MODEL_PATH
from nn.online import OnlineAdapter
from integration.arena import make_arena, run_headless_arena
from integration.adaptive import AdaptiveController


def synthetic_features():
    return [0.6, 0.8, 0.4, 0.65, 55, 54, 4, 2, 1, 7, 3, 0.55]


def test_batch_updates_only_at_ten_and_base_frozen(tmp_path):
    model=MLP.load(MODEL_PATH)
    before={k:v.copy() for k,v in model.params.items()}
    adapter=OnlineAdapter('test_batch',forget_every=2)
    for i in range(9):
        assert adapter.observe(model,synthetic_features(),i%2==0) is False
    assert adapter.updates == 0 and len(adapter.pending)==9
    assert adapter.observe(model,synthetic_features(),True) is True
    assert adapter.updates == 1 and len(adapter.pending)==0
    assert adapter.observations==10
    assert np.linalg.norm(adapter.weights)>0
    assert all(np.array_equal(v,model.params[k]) for k,v in before.items())


def test_scheduled_forgetting_changes_only_adapter(tmp_path):
    model=MLP.load(MODEL_PATH)
    before={k:v.copy() for k,v in model.params.items()}
    adapter=OnlineAdapter('test_erasures',feature_dropout=.1,
                          forget_every=2,forget_fraction=.5)
    for i in range(20):
        adapter.observe(model,synthetic_features(),i%5 != 0)
    assert adapter.updates==2 and adapter.observations==20
    assert len(adapter.forget_events)==1
    e=adapter.last_forget_event
    assert e['reason']=='scheduled' and e['erased_count']>=1
    assert e['adapter_norm_after']<e['adapter_norm_before']
    assert e['pretrained_model_modified'] is False
    assert adapter.recovery_updates_remaining == 2
    assert all(np.array_equal(v,model.params[k]) for k,v in before.items())
    # Do NOT replay older examples during the next incremental training batch.
    for i in range(10):
        adapter.observe(model,synthetic_features(),i%5 != 0)
    assert adapter.updates==3 and len(adapter.forget_events)==1
    assert adapter.recovery_updates_remaining == 1
    assert np.linalg.norm(adapter.weights)>0
    adapter.save(tmp_path)
    restored=OnlineAdapter.load('test_erasures',tmp_path)
    assert restored.observations==30
    assert np.array_equal(restored.weights,adapter.weights)
    assert restored.forget_events==adapter.forget_events
    assert restored.prequential_losses==adapter.prequential_losses


def test_forgetting_ablation_and_manual_trigger():
    model=MLP.load(MODEL_PATH)
    adapter=OnlineAdapter('no_erasures',forgetting_enabled=False,forget_every=1)
    for _ in range(10):adapter.observe(model,synthetic_features(),True)
    previous=adapter.weights.copy()
    assert adapter.forget_now() is None
    assert not adapter.forget_events
    assert np.array_equal(previous,adapter.weights)
    adapter.forgetting_enabled=True
    event=adapter.forget_now('player-triggered')
    assert event and event['reason']=='player-triggered'
    assert not np.array_equal(previous,adapter.weights)


def test_four_independent_search_actors_and_new_levels():
    lanes=make_arena(seed=17,persist=False,motor_noise=False)
    assert {a.algorithm for a in lanes}=={'bfs','dfs','astar','greedy'}
    assert len({a.controller.owner for a in lanes})==4
    assert len({a.level.seed for a in lanes})==4
    for agent in lanes:
        assert agent.plan.algorithm==agent.algorithm
        for _ in range(140):
            if agent.history:break
            agent.tick()
        assert len(agent.history)==1
        assert agent.history[0]['algorithm']==agent.algorithm
        assert agent.level.seed!=agent.history[0]['seed']


def test_headless_arena_forgets_and_then_relearns_from_fresh_batch():
    payload=run_headless_arena(seed=42,rounds=31,forget_every=2,
                               feature_dropout=.1,forget_fraction=.4)
    for a in payload['agents']:
        assert a['online_updates']==3 and a['forget_events']==1
        assert a['new_examples_pending']==1
        assert len(a['prequential_losses'])==3
        assert a['last_forget_event']['observations']==20
        assert a['frozen_base_model'] is True
    for name,events in payload['attempts'].items():
        assert sum(bool(e['forgetting_event']) for e in events)==1
        assert events[19]['forgetting_event']['reason']=='scheduled'
        assert events[29]['online_adapter_updated'] is True


def test_physics_sweeps_midair_spikes():
    from game.physics import Level, step
    rows=['.'*12 for _ in range(10)]+['#'*12,'#'*12]
    hazardous=list(rows[8]);hazardous[1]='^';rows[8]=''.join(hazardous)
    level=Level(tuple(rows))
    assert step(level,level.start,'J') is None  # jump crosses row 8 in one tick
    plain=list(rows);plain[8]='.'*12
    assert step(Level(tuple(plain)),level.start,'J') is not None
