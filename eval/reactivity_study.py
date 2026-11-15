"""Repeatable paired-seed *descriptive* comparison of reactive world perturbations.

Each condition starts with the same independent agent seeds. Subsequent levels
may diverge as results affect adaptation; this is not a randomized human study.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from integration.arena import make_arena

RESULTS=Path(__file__).resolve().parents[1]/'results'


def run_case(seed: int, rounds: int, surprises: bool):
    lanes=make_arena(seed=seed,persist=False,motor_noise=True,surprises=surprises)
    for tick in range(20000):
        for lane in lanes:
            if len(lane.history)<rounds:
                lane.tick()
        if all(len(a.history)>=rounds for a in lanes):break
    else:
        raise RuntimeError('Reactive study exceeded logical step budget')
    return [{'seed':seed,'algorithm':a.algorithm,'surprises':int(surprises),
             'attempts':len(a.history),'wins':sum(e['won'] for e in a.history),
             'win_rate':mean(e['won'] for e in a.history),
             'world_changes':sum(e['surprise_count'] for e in a.history),
             'replans':sum(e['replans'] for e in a.history),
             'steps':sum(round(e['elapsed']*10) for e in a.history),
             'online_updates':a.controller.adapter.updates}
            for a in lanes]


def study(seeds=(11,22,33,44,55), rounds=3, out=RESULTS):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        for surprise in (False,True):
            rows.extend(run_case(seed,rounds,surprise))
        print('seed',seed,'complete',flush=True)
    fields=list(rows[0]);path=out/'reactivity_raw.csv'
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for algorithm in ('bfs','dfs','astar','greedy'):
        for surprise in (0,1):
            group=[r for r in rows if r['algorithm']==algorithm and r['surprises']==surprise]
            record={'algorithm':algorithm,'surprises':surprise,'seeds':len(group),
                    'attempts_per_seed':rounds}
            for metric in ('win_rate','world_changes','replans','steps'):
                record[f'{metric}_mean']=round(mean(r[metric] for r in group),4)
                record[f'{metric}_sd']=round(stdev(r[metric] for r in group),4) if len(group)>1 else 0
            summary.append(record)
    dest=out/'reactivity_summary.csv'
    with dest.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('Saved',path,dest)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--rounds',type=int,default=3)
    args=ap.parse_args()
    for r in study(args.seeds,args.rounds):print(r)
