"""Three reproducible experiments; ≥5 seeds; CSV, summaries and PNG plots.
Caveat: adaptation outcome simulations are NOT human validation.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean,stdev
import numpy as np
from csp.generator import generate
from csp.solver import ChunkCSP
from search.algorithms import solve
from integration.adaptive import AdaptiveController
from data.make_data import FEATURES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows,group_by,metrics):
    groups={}
    for row in rows:groups.setdefault(row[group_by],[]).append(row)
    out=[]
    for group,items in groups.items():
        record={group_by:group,'n':len(items)}
        for metric in metrics:
            vals=[float(r[metric]) for r in items if r[metric] not in ('',None)]
            record[f'{metric}_mean']=round(mean(vals),4) if vals else None
            record[f'{metric}_std']=round(stdev(vals),4) if len(vals)>1 else 0.0 if vals else None
        out.append(record)
    return out

def search_experiment(seeds):
    rows=[]
    for seed in seeds:
        level=generate(seed=seed,target=0.6,n_chunks=7).level
        if level is None:continue
        for name in ('bfs','dfs','astar','greedy'):
            r=solve(level,name,max_expansions=18000)
            rows.append({'seed':seed,'algorithm':name,'found':int(r.found),'nodes_expanded':r.expanded,
                         'nodes_generated':r.generated,'frontier_max':r.frontier_max,
                         'path_cost':r.cost if r.found else '',
                         'elapsed_ms':r.elapsed_ms})
    return rows,summarize(rows,'algorithm',('found','nodes_expanded','nodes_generated','frontier_max','path_cost','elapsed_ms'))

def csp_experiment(seeds):
    rows=[]
    for seed in seeds:
        for fc,ac in ((False,False),(True,False),(False,True),(True,True)):
            p=ChunkCSP(10,0.73,seed=seed)
            r=p.solve(fc,ac);s=r.stats
            rows.append({'seed':seed,'mode':f'FC{int(fc)}_AC3{int(ac)}',
                         'solved':int(r.chunks is not None),'assignments':s.assignments,
                         'backtracks':s.backtracks,'pruned':s.pruned,
                         'constraint_checks':s.constraint_checks,'elapsed_ms':s.elapsed_ms})
    return rows,summarize(rows,'mode',('solved','assignments','backtracks','pruned','constraint_checks','elapsed_ms'))

def summarize_seed_replicates(rows, group_by, metrics):
    """Replicate mean ± sample SD across independent SEEDS, not pooled Bernoulli rows."""
    grouped={}
    for row in rows:
        grouped.setdefault((row[group_by], row['seed']),[]).append(row)
    seeds=[]
    for (method,seed),items in grouped.items():
        record={'seed':seed,group_by:method,'attempts':len(items)}
        for field in metrics:
            record[field]=mean(float(e[field]) for e in items)
        seeds.append(record)
    return seeds, summarize(seeds,group_by,metrics)


def simulate_win(skill,level,uniform_draw,noise):
    f=level.features()
    challenge=0.57*level.difficulty+0.20*min(f['max_gap']/4,1)+0.17*min(f['hazards']/5,1)+0.06*min(f['platforms']/10,1)
    z=4*(skill-challenge)+noise
    p=1/(1+np.exp(-z))
    return bool(uniform_draw < p)

def adaptation_experiment(seeds,profiles_per_seed=8,rounds=5):
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        skills=rng.uniform(.18,.92,profiles_per_seed)
        draws=rng.random((profiles_per_seed,rounds))
        noises=rng.normal(0,.42,(profiles_per_seed,rounds))
        for pid,skill in enumerate(skills):
            for method in ('fixed','rule','nn'):
                controller=AdaptiveController(seed=seed*10000+pid*17,use_nn=(method=='nn'))
                # Only label simulator has latent skill; controller sees observed history.
                for turn in range(rounds):
                    if method=='fixed':
                        r=generate(seed=controller.seed+turn*1009,target=0.5)
                        if not r.level:continue
                        level=r.level;pred=controller.predict(level)
                    else:
                        candidate=controller.next_level(n_candidates=5)
                        level=candidate.result.level;pred=candidate.predicted_win
                    won=simulate_win(skill,level,draws[pid,turn],noises[pid,turn])
                    # Simulated telemetry is separate from real gameplay logs.
                    controller.profile.record(level,won,75-40*skill,10,
                                              round(10*(.2+.8*skill)),
                                              1 if won else .3+.6*skill,save=False)
                    rows.append({'seed':seed,'persona':pid,'round':turn,'method':method,
                                 'win':int(won),'level_difficulty':level.difficulty,
                                 'predicted_win':round(pred,5),
                                 'prediction_abs_error':round(abs(int(won)-pred),5),
                                 'distance_from_target':round(abs(.70-pred),5),
                                 'synthetic_outcome':1})
    metrics=('win','level_difficulty','prediction_abs_error','distance_from_target')
    _,summary=summarize_seed_replicates(rows,'method',metrics)
    return rows,summary

def plot_results(summaries):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    configs=[('search','algorithm','nodes_expanded_mean','Search: nodes expanded'),
             ('csp','mode','backtracks_mean','CSP: backtracks'),
             ('adaptation','method','win_mean','Adaptation: simulated win rate')]
    for kind,label,value,title in configs:
        entries=summaries[kind]
        if not entries:continue
        fig,ax=plt.subplots(figsize=(8,4.5))
        # Bars are the mean over seeds; error bars are ± one sample SD.
        spread=[e.get(value.replace('_mean','_std')) or 0 for e in entries]
        ax.bar([e[label] for e in entries],[e[value] for e in entries],
               yerr=spread,capsize=6,error_kw={'ecolor':'#333333','lw':1.2})
        ax.set_ylim(bottom=0)
        ax.set_title(f'{title} (mean ± SD, n={entries[0]["n"]} seeds)')
        ax.set_ylabel(value.replace('_',' '))
        ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        fig.savefig(OUT/f'{kind}_plot.png',dpi=160)
        plt.close(fig)

def run_all(seeds=(11,22,33,44,55),profiles_per_seed=8,rounds=5):
    OUT.mkdir(parents=True,exist_ok=True)
    stages={'search':search_experiment(seeds),
            'csp':csp_experiment(seeds),
            'adaptation':adaptation_experiment(seeds,profiles_per_seed,rounds)}
    summaries={}
    for name,(rows,summary) in stages.items():
        write_csv(OUT/f'{name}_raw.csv',rows)
        write_csv(OUT/f'{name}_summary.csv',summary)
        summaries[name]=summary
    seed_rows,_=summarize_seed_replicates(stages['adaptation'][0],'method',
        ('win','level_difficulty','prediction_abs_error','distance_from_target'))
    write_csv(OUT/'adaptation_by_seed.csv',seed_rows)
    plot_results(summaries)
    (OUT/'experiment_summary.json').write_text(json.dumps({
        'seeds':list(seeds),'adaptation_kind':'synthetic player-outcome simulation',
        'methodology_note':'no human subjects measured; latency affected by host load',
        'summary':summaries},indent=2))
    return summaries

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33,44,55])
    ap.add_argument('--profiles',type=int,default=8)
    ap.add_argument('--rounds',type=int,default=5)
    args=ap.parse_args()
    print(json.dumps(run_all(args.seeds,args.profiles,args.rounds),indent=2))
