"""Four independent graph-search ACTORS, one frozen neural coach per actor.

BFS, DFS, A*, and Greedy supply action plans and re-plan from the true physical
state after a motor slip. NN never chooses their movement commands. Separate
controllers train small per-agent residual calibration heads every 10 attempts.
No GUI dependency: this engine can be tested headlessly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from random import Random
from pathlib import Path
from typing import Any
import json
from game.physics import State, Level, step
from search.algorithms import solve, SearchResult
from integration.world_events import trigger_surprise
from integration.adaptive import AdaptiveController, Candidate

ALGORITHMS=('bfs','dfs','astar','greedy')
STYLE={
 'bfs': {'title':'THE EXPLORER','slip':.025,'start':.29,'tone':'cyan'},
 'dfs': {'title':'THE WANDERER','slip':.065,'start':.25,'tone':'purple'},
 'astar': {'title':'THE TACTICIAN','slip':.014,'start':.41,'tone':'green'},
 'greedy': {'title':'THE SPEEDSTER','slip':.085,'start':.35,'tone':'pink'},
}

@dataclass
class AgentRunner:
    algorithm: str
    seed: int = 42
    persist: bool = False
    motor_noise: bool = True
    surprises: bool = True
    surprise_interval: tuple[int,int] = (11, 18)
    max_surprises_per_stage: int = 3
    controller: AdaptiveController = field(init=False)
    rng: Random = field(init=False)
    choice: Candidate = field(init=False)
    state: State = field(init=False)
    prev: State = field(init=False)
    plan: SearchResult = field(init=False)
    plan_index: int = 0
    ticks: int = 0
    jumps: int = 0
    landings: int = 0
    stages: int = 0
    wins: int = 0
    losses: int = 0
    replans: int = 0
    last_event: str = 'BOOTING'
    last_update: bool = False
    coins: set[tuple[int,int]] = field(default_factory=set)
    trace_index: int = 0
    history: list[dict[str,Any]] = field(default_factory=list)
    surprise_count: int = 0
    total_surprises: int = 0
    surprise_attempts: int = 0
    next_surprise_at: int = 0
    world_events: list[dict[str,Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:raise ValueError('Unknown search algorithm')
        self.rng=Random(self.seed+sum(ord(c) for c in self.algorithm))
        self.controller=AdaptiveController(
            seed=self.seed,use_nn=True,owner='arena_'+self.algorithm,
            persist=self.persist,initial_challenge=STYLE[self.algorithm]['start'])
        self.start_next_level()

    @property
    def level(self)->Level:return self.choice.result.level

    def start_next_level(self):
        self.choice=self.controller.next_level(n_candidates=7)
        self.state=self.choice.result.level.start;self.prev=self.state
        self.ticks=0;self.jumps=0;self.landings=0;self.plan_index=0
        self.replans=0;self.trace_index=0;self.coins.clear()
        self.surprise_count=0;self.world_events.clear()
        self.next_surprise_at=self.rng.randint(*self.surprise_interval)
        self.stages+=1
        self.plan=solve(self.level,self.algorithm,max_expansions=24000)
        if not self.plan.found:self.last_event='SEARCH COULD NOT FIND A ROUTE'
        else:self.last_event='NEW CHALLENGE  //  '+str(self.stages)

    def _finish(self,won:bool,reason:str):
        self.wins+=int(won);self.losses+=int(not won)
        ev=self.controller.record(self.level,won,self.ticks*0.10,self.jumps,
                                  self.landings,min(1,self.state.x/max(1,self.level.goal_x)),
                                  save=self.persist,origin='algorithm',
                                  extra={'surprise_count':self.surprise_count,
                                         'unexpected_events':list(self.world_events),
                                         'replans':self.replans})
        ev.update({'algorithm':self.algorithm,'reason':reason,
                   'predicted_win_before':round(self.choice.predicted_win,6),
                   'prediction_squared_error':round((float(won)-self.choice.predicted_win)**2,6),
                   'planned_path_length':self.plan.cost,
                   'search_expanded':self.plan.expanded,
                   'replans':self.replans,
                   'unexpected_events':list(self.world_events),
                   'surprise_count':self.surprise_count,
                   'challenge_after':round(self.controller.challenge,3),
                   'online_adapter_updated':self.controller.last_online_update,
                   'forgetting_event': (self.controller.adapter.last_forget_event
                       if self.controller.adapter.last_forget_event
                       and self.controller.adapter.last_forget_event['observations']==self.controller.adapter.observations
                       else None)})
        self.history.append(ev)
        self.last_update=self.controller.last_online_update
        self.start_next_level()
        self.last_event=('CLEARED' if won else reason.upper())+('  //  ONLINE LEARNED' if self.last_update else '')
        return ev

    def tick(self):
        """One logical time step. Returns completed attempt telemetry or None."""
        if not self.plan.found:
            return self._finish(False,'no route within search budget')
        self.ticks+=1
        if self.surprises and self.ticks>=self.next_surprise_at and (
                self.surprise_count<self.max_surprises_per_stage):
            self.inject_surprise()
            self.next_surprise_at=self.ticks+self.rng.randint(*self.surprise_interval)
        # Bounded episode: DFS is not allowed to wander forever.
        if self.ticks>min(700,max(140,int(len(self.plan.actions)*1.7)+40)):
            return self._finish(False,'time limit')
        if self.plan_index >= len(self.plan.actions):
            if self.level.complete(self.state):return self._finish(True,'goal')
            if not self._replan():return self._finish(False,'no replan')
        action=self.plan.actions[self.plan_index]
        grounded=self.level.solid(self.state.x,self.state.y) and self.state.vy==0
        if grounded and action.startswith('J'):self.jumps+=1
        # Motor noise simulates execution errors. It is explicitly distinct from
        # the planner's decisions and tagged as an algorithmic-agent simulation.
        applied=action
        if self.motor_noise and self.rng.random()<STYLE[self.algorithm]['slip']:
            applied=self.rng.choice(('W','L','R'))
        prev=self.state
        nxt=step(self.level,prev,applied)
        if nxt is None:return self._finish(False,'hazard or fall')
        self.prev=prev;self.state=nxt
        if self.level.solid(nxt.x,nxt.y) and nxt.vy==0 and not grounded:
            self.landings+=1
        for yy in (nxt.y-1,nxt.y-2):
            if self.level.tile(nxt.x,yy)=='o':self.coins.add((nxt.x,yy))
        if self.level.complete(nxt):return self._finish(True,'goal')
        self.plan_index+=1
        if self.plan_index>=len(self.plan.path) or nxt!=self.plan.path[self.plan_index]:
            if not self._replan():return self._finish(False,'search replanning failed')
        return None

    def inject_surprise(self, kinds=None) -> bool:
        """An observable mid-game world change, followed by REAL replanning.

        Also callable directly from the arena (E key) for the live demo.
        The seed schedule is never passed to the search planner.
        """
        self.surprise_attempts += 1
        event=trigger_surprise(self.level,self.state,self.algorithm,self.rng,
                               kinds=kinds,max_expansions=14000)
        if event is None:
            return False
        self.choice.result.level=event.level
        self.plan=event.plan
        self.plan_index=0
        self.replans+=1
        self.surprise_count+=1
        self.total_surprises+=1
        detail={'tick':self.ticks,'kind':event.name,'description':event.description,
                'x':event.x,'new_plan_cost':event.plan.cost,
                'search_expanded':event.plan.expanded}
        self.world_events.append(detail)
        self.last_event='SURPRISE! '+event.description+'  //  REPLANNED'
        return True

    def _replan(self):
        self.replans+=1
        # This is the lane's own algorithm, *not* an invisible A* controller.
        new_plan=solve(self.level,self.algorithm,max_expansions=24000,start=self.state)
        if not new_plan.found:return False
        self.plan=new_plan;self.plan_index=0
        return True

    def summary(self):
        return {'agent':self.algorithm,'completed':len(self.history),
                'wins':self.wins,'losses':self.losses,
                'win_rate':self.wins/max(1,self.wins+self.losses),
                'challenge':round(self.controller.challenge,3),
                'level_difficulty':round(self.level.difficulty,3),
                'level_chunks':list(self.level.chunks),
                'search_expanded':self.plan.expanded,
                'search_path_length':self.plan.cost,
                'surprises_this_stage':self.surprise_count,
                'total_surprises':self.total_surprises,
                'surprise_attempts':self.surprise_attempts,
                'surprises_enabled':self.surprises,
                'current_world_events':list(self.world_events),
                'online_updates':self.controller.adapter.updates,
                'new_examples_pending':len(self.controller.adapter.pending),
                'forget_events':len(self.controller.adapter.forget_events),
                'forget_every_updates':self.controller.adapter.forget_every,
                'feature_dropout':self.controller.adapter.feature_dropout,
                'prequential_losses':list(self.controller.adapter.prequential_losses),
                'last_forget_event':self.controller.adapter.last_forget_event,
                'recovery_updates_remaining':self.controller.adapter.recovery_updates_remaining,
                'frozen_base_model':True}


def make_arena(seed=42,persist=False,motor_noise=True,forget_every=3,
               forget_fraction=.35,feature_dropout=.12,forgetting_enabled=True,
               surprises=True):
    result=[AgentRunner(a,seed+1003*i,persist=persist,motor_noise=motor_noise,
                                surprises=surprises)
            for i,a in enumerate(ALGORITHMS)]
    for agent in result:
        adapter=agent.controller.adapter
        adapter.forget_every=forget_every
        adapter.forget_fraction=forget_fraction
        adapter.feature_dropout=feature_dropout
        adapter.forgetting_enabled=forgetting_enabled
        adapter.__post_init__()
    return result


def run_headless_arena(seed=42,rounds=10,persist=False,max_ticks=100000,
                       forget_every=3,forget_fraction=.35,feature_dropout=.12,
                       forgetting_enabled=True,surprises=False):
    lanes=make_arena(seed,persist=persist,forget_every=forget_every,
                     forget_fraction=forget_fraction,feature_dropout=feature_dropout,
                     forgetting_enabled=forgetting_enabled,surprises=surprises)
    if rounds<=0:raise ValueError('rounds must be positive')
    for _ in range(max_ticks):
        for a in lanes:
            if len(a.history)<rounds:a.tick()
        if all(len(a.history)>=rounds for a in lanes):break
    else:raise RuntimeError('Arena headless time limit exhausted')
    return {'seed':seed,'rounds':rounds,'agents':[a.summary() for a in lanes],
            'attempts':{a.algorithm:a.history for a in lanes},
            'note':'Four search-driven surrogate agents, not human results'}
