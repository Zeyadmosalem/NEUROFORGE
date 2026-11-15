"""Exploratory ablation: fixed pretrained NN + online adapter with/without forgetting.

All training is incremental, using NEW disjoint batches of ten simulated
attempts. Run with 5 seeds for a five-seed supplemental study. Result rows
must never be presented as evidence of real human adaptation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from integration.arena import run_headless_arena

ROOT=Path(__file__).resolve().parents[1]

def run_study(seeds=(7,17,27,37,47),rounds=36,forget_every=2):
    rows=[]
    for seed in seeds:
        for forgetting in (False,True):
            trial=run_headless_arena(seed=seed,rounds=rounds,persist=False,
                                      forget_every=forget_every,
                                      forgetting_enabled=forgetting)
            for a in trial['agents']:
                events=trial['attempts'][a['agent']]
                rows.append({
                    'seed':seed,'agent':a['agent'],
                    'mode':'partial_forgetting' if forgetting else 'no_forgetting',
                    'attempts':len(events),'win_rate':mean(e['won'] for e in events),
                    'brier':mean(e['prediction_squared_error'] for e in events),
                    'mean_difficulty':mean(e['difficulty'] for e in events),
                    'after_20_brier':mean(e['prediction_squared_error'] for e in events[20:]),
                    'after_20_win_rate':mean(e['won'] for e in events[20:]),
                    'online_updates':a['online_updates'],'forget_events':a['forget_events'],
                    'batch_loss_first':a['prequential_losses'][0] if a['prequential_losses'] else None,
                    'batch_loss_last':a['prequential_losses'][-1] if a['prequential_losses'] else None,
                    'origin':'search_agent_simulation_NOT_human',
                })
    summary=[]
    for mode in ('no_forgetting','partial_forgetting'):
        for agent in ('bfs','dfs','astar','greedy'):
            subset=[r for r in rows if r['mode']==mode and r['agent']==agent]
            rec={'mode':mode,'agent':agent,'n_seeds':len(subset)}
            for field in ('win_rate','brier','after_20_brier','after_20_win_rate'):
                values=[r[field] for r in subset]
                rec[field+'_mean']=round(mean(values),5)
                rec[field+'_std']=round(stdev(values) if len(values)>1 else 0.,5)
            summary.append(rec)
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    for name,items in (('forgetting_raw.csv',rows),('forgetting_summary.csv',summary)):
        with (out/name).open('w',newline='',encoding='utf8') as f:
            wr=csv.DictWriter(f,fieldnames=items[0].keys())
            wr.writeheader();wr.writerows(items)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        x=np.arange(4);labels=['BFS','DFS','A*','Greedy']
        for ax,field,title in ((axs[0],'win_rate_mean','Completed attempts'),
                               (axs[1],'after_20_brier_mean','Prediction Brier after 20 attempts')):
            for j,mode in enumerate(('no_forgetting','partial_forgetting')):
                vals=[next(r[field] for r in summary if r['mode']==mode and r['agent']==a)
                      for a in ('bfs','dfs','astar','greedy')]
                ax.bar(x+(j-.5)*.35,vals,width=.35,
                       label='Never forget' if j==0 else 'Partial reset')
            ax.set_xticks(x,labels);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        axs[0].set_ylim(0,1.05);axs[0].legend(fontsize=9)
        axs[1].set_ylim(bottom=0)
        fig.suptitle('Exploratory synthetic-agent study; lower Brier is better')
        fig.tight_layout();fig.savefig(out/'forgetting_ablation.png',dpi=150);plt.close(fig)
    except ImportError:
        pass
    (out/'forgetting_summary.json').write_text(json.dumps({
        'seeds':list(seeds),'rounds':rounds,'forget_every':forget_every,
        'summary':summary,
        'interpretation':'No causal improvement guaranteed. Search-agent simulation, not human study.'},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27,37,47])
    p.add_argument('--rounds',type=int,default=36)
    p.add_argument('--forget-every',type=int,default=2)
    args=p.parse_args()
    print(json.dumps(run_study(args.seeds,args.rounds,args.forget_every),indent=2))
