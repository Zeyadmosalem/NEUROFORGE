"""Closed loop: live gameplay -> profiles -> frozen NN + tiny online adapter
  -> CSP candidate creation -> search verification -> unique challenge selection.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from math import exp
from csp.generator import generate, GenerationResult
from data.make_data import FEATURES
from nn.model import MLP, MODEL_PATH
from nn.online import OnlineAdapter

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class PlayerProfile:
    attempts: list[dict] = field(default_factory=list)

    @classmethod
    def load_history(cls, owner='human', origin='human', limit=40):
        """Recover gameplay context without re-using old samples for gradient steps."""
        filename = 'human_gameplay.jsonl' if origin == 'human' else 'algorithm_gameplay.jsonl'
        path = ROOT / 'data' / 'generated' / filename
        if not path.exists():
            return cls()
        events = []
        for line in path.read_text(encoding='utf8').splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if event.get('owner') == owner and event.get('origin') == origin:
                events.append(event)
        return cls(events[-limit:])

    def features(self):
        events=self.attempts[-8:]
        if not events:
            return {'recent_win_rate':0.50,'jump_accuracy':0.70,
                    'deaths_per_attempt':0.9,'recent_progress':0.50,'recent_time':65.}
        n=len(events)
        jumps=sum(e['jumps'] for e in events)
        landed=sum(e['landings'] for e in events)
        return {'recent_win_rate':sum(e['won'] for e in events)/n,
                'jump_accuracy':landed/jumps if jumps else 0.7,
                'deaths_per_attempt':sum(not e['won'] for e in events)/n,
                'recent_progress':sum(e['progress'] for e in events)/n,
                'recent_time':sum(e['elapsed'] for e in events)/n}

    def record(self,level,won,elapsed,jumps,landings,progress,save=True,
               origin='human', owner='human', extra=None):
        event={'won':int(bool(won)),'elapsed':round(float(elapsed),2),
               'jumps':int(jumps),'landings':int(landings),
               'progress':round(float(progress),3),'seed':level.seed,
               'difficulty':level.difficulty,'chunks':list(level.chunks),
               'origin':origin,'owner':owner}
        if extra:
            event.update(extra)
        self.attempts.append(event)
        if save:
            dest=ROOT/'data'/'generated'/('human_gameplay.jsonl' if origin=='human'
                                        else 'algorithm_gameplay.jsonl')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('a',encoding='utf8') as f:f.write(json.dumps(event)+'\n')
        return event


def fallback_predict(profile,level):
    p=profile.features()
    skill=0.65*p['recent_win_rate']+0.35*p['jump_accuracy']
    f=level.features();challenge=0.7*level.difficulty+0.3*min(f['max_gap']/4,1)
    return 1/(1+exp(-4*(skill-challenge)))

@dataclass
class Candidate:
    result: GenerationResult
    predicted_win: float
    target_distance: float

class AdaptiveController:
    def __init__(self,seed=42,use_nn=True,target_win=0.70,owner='human',
                 persist=False,initial_challenge=0.30):
        self.seed=seed;self.use_nn=use_nn;self.target_win=target_win
        self.profile=(PlayerProfile.load_history(owner, 'human' if owner=='human' else 'algorithm')
                      if persist else PlayerProfile())
        self.model=None;self.last_candidates=[]
        self.owner=owner;self.persist=persist
        self.challenge=float(max(0.06,min(0.98,initial_challenge)))
        self.streak=0;self.recent_level_signatures=[];self.last_selection=None
        self.last_online_update=False
        self.adapter=OnlineAdapter.load(owner) if persist else OnlineAdapter(owner)
        if MODEL_PATH.exists():
            try:self.model=MLP.load(MODEL_PATH)
            except (ValueError,OSError,KeyError):self.model=None

    def feature_vector(self,level):
        fs={**self.profile.features(),**level.features()}
        return [fs[f] for f in FEATURES]

    def predict(self,level):
        if self.use_nn and self.model is not None:
            return self.adapter.predict(self.model,self.feature_vector(level))
        return float(fallback_predict(self.profile,level))

    def record(self,level,won,elapsed,jumps,landings,progress,
               save=True,origin='human',extra=None):
        """Snapshot pre-outcome features to avoid target leakage."""
        if self.last_selection and self.last_selection['level_seed']==level.seed:
            vector=self.last_selection['features']
        else:vector=self.feature_vector(level)
        event=self.profile.record(level,won,elapsed,jumps,landings,progress,
                                  save=save,origin=origin,owner=self.owner,extra=extra)
        # Win streak raises challenge quickly; repeated losses back it off.
        self.streak = self.streak+1 if won else 0
        up=0.065+0.018*min(self.streak,4)
        self.challenge = max(0.06,min(0.99,self.challenge+(up if won else -0.07)))
        self.last_online_update=False
        if self.model is not None:
            self.last_online_update=self.adapter.observe(self.model,vector,won,persist=self.persist)
            if self.persist and not self.last_online_update:
                # Keep pending evidence across a process restart.
                self.adapter.save()
        self.last_selection=None
        return event

    def next_level(self,n_candidates=7,n_chunks=None,forward_checking=True,
                   arc_consistency=True):
        # Slowly lengthen the levels, independently of how tough the chunks are.
        if n_chunks is None:n_chunks=6+min(4,len(self.profile.attempts)//4)
        options=[]
        for k in range(n_candidates):
            # Sample around the current challenge and across the whole range.
            offset=(k-(n_candidates-1)/2)*0.105
            target=max(0.04,min(0.99,self.challenge+offset))
            attempt_seed=self.seed+k*103+len(self.profile.attempts)*137
            r=generate(seed=attempt_seed,target=target,n_chunks=n_chunks,
                       forward_checking=forward_checking,arc_consistency=arc_consistency)
            if not r.level:continue
            p=self.predict(r.level)
            signature=tuple(r.level.chunks)
            repetition=0.36 if signature in self.recent_level_signatures[-7:] else 0.
            # Geometry must actually progress. Neural prediction is a guide, not
            # allowed to trap every agent on the same trivial level forever.
            score=(1.4*abs(r.level.difficulty-self.challenge)
                   +0.22*abs(self.target_win-p)+repetition)
            options.append(Candidate(r,p,score))
        self.last_candidates=options
        if not options:raise RuntimeError('No search-verified candidates generated')
        options.sort(key=lambda c:(c.target_distance,-len(set(c.result.level.chunks))))
        chosen=options[0];self.seed+=1009
        self.recent_level_signatures.append(tuple(chosen.result.level.chunks))
        self.recent_level_signatures=self.recent_level_signatures[-18:]
        self.last_selection={'level_seed':chosen.result.level.seed,
                             'features':self.feature_vector(chosen.result.level)}
        return chosen
