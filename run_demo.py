"""Single entry point: GUI, headless verified AI demo, or ASCII live trace."""
import argparse
import json
from pathlib import Path
from integration.adaptive import AdaptiveController
from search.algorithms import solve
from csp.solver import ChunkCSP

def headless(seed=42):
    ctrl=AdaptiveController(seed=seed)
    chosen=ctrl.next_level(n_candidates=5)
    result=chosen.result;level=result.level
    print('=== NEUROFORGE: INTEGRATED HEADLESS DEMO ===')
    print('PLAYER PROFILE:',json.dumps(ctrl.profile.features()))
    print('CANDIDATES:',[(round(c.result.level.difficulty,2),round(c.predicted_win,3)) for c in ctrl.last_candidates])
    print('CHOSEN LEVEL: seed',level.seed,'chunks:',', '.join(level.chunks))
    for row in level.rows:print(row)
    print('CSP:',vars(result.csp.stats))
    results={}
    for algo in ('bfs','dfs','astar','greedy'):
        r=solve(level,algo)
        results[algo]={'found':r.found,'nodes_expanded':r.expanded,
                       'nodes_generated':r.generated,'path_cost':r.cost,
                       'elapsed_ms':r.elapsed_ms}
        print(algo.upper(),results[algo])
    csp_impossible=ChunkCSP(impossible=True).solve(True,True)
    print('IMPOSSIBLE CSP:',csp_impossible.reason)
    payload={'seed':seed,'level_chunks':list(level.chunks),
             'predicted_success':chosen.predicted_win,
             'csp_stats':vars(result.csp.stats),'search':results,
             'impossible_case':csp_impossible.reason}
    out=Path(__file__).resolve().parent/'results'/'demo_summary.json'
    out.parent.mkdir(exist_ok=True)
    plot=plot_cooperation(ctrl,chosen,results,out.parent/'demo_integration.png')
    if plot:payload['plot']=plot.name
    out.write_text(json.dumps(payload,indent=2))
    print('Saved:',out)
    print('Saved plot:',plot) if plot else print('Plot skipped: matplotlib is not installed')
    return payload

def plot_cooperation(ctrl,chosen,search_results,dest):
    """One figure showing how the NN, the CSP and search cooperate on this run.

    Top: the chosen level (input) with the A* route (solution). Bottom: the NN
    scoring every CSP candidate, the CSP counters under each FC/AC-3 setting for
    the chosen level's problem, and the four search algorithms on that level.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return None
    level=chosen.result.level
    fig=plt.figure(figsize=(13,8.2))
    grid=fig.add_gridspec(2,3,height_ratios=[1,1.25])
    # Input and solution: tile map, hazards, and the verified A* path.
    ax=fig.add_subplot(grid[0,:])
    for y,row in enumerate(level.rows):
        for x,tile in enumerate(row):
            if tile=='#':ax.add_patch(plt.Rectangle((x,y),1,1,color='#3b4a6b'))
            elif tile=='^':ax.add_patch(plt.Polygon([(x,y+1),(x+.5,y+.15),(x+1,y+1)],color='#e0507a'))
            elif tile=='o':ax.add_patch(plt.Circle((x+.5,y+.5),.22,color='#e8b84a'))
    for p in level.platforms:ax.add_patch(plt.Rectangle((p.x,p.y),p.width,.5,color='#4fb3c9'))
    for e in level.enemies:ax.add_patch(plt.Circle((e.x+.5,e.y+.5),.4,color='#e0507a'))
    for g in level.gates:ax.plot([g.x+.5]*2,[g.y_top,g.y_bottom+1],color='#e0507a',lw=2)
    route=solve(level,'astar')
    ax.plot([s.x+.5 for s in route.path],[s.y-.5 for s in route.path],color='#2e9e6b',lw=2,
            label=f'A* route ({route.cost} actions)')
    ax.set_xlim(0,level.width);ax.set_ylim(level.height,0);ax.set_aspect('equal')
    ax.set_xticks(range(0,level.width+1,8));ax.set_yticks([])
    ax.set_title('Input and solution: CSP-built level (seed %d, difficulty %.2f) with the search-verified route'
                 %(level.seed,level.difficulty),fontsize=11)
    ax.legend(loc='upper left',fontsize=9)
    # NN: predicted win probability for each CSP candidate; chosen one marked.
    ax=fig.add_subplot(grid[1,0])
    cands=sorted(ctrl.last_candidates,key=lambda c:c.result.level.difficulty)
    ax.plot([c.result.level.difficulty for c in cands],[c.predicted_win for c in cands],
            'o-',color='#3b6fb5',label='candidate')
    ax.plot([level.difficulty],[chosen.predicted_win],'o',ms=13,mfc='none',mec='#e0507a',mew=2,label='chosen')
    ax.axhline(ctrl.target_win,color='gray',ls='--',lw=1,label=f'target P(win) {ctrl.target_win:.2f}')
    ax.set_xlabel('candidate level difficulty (CSP)');ax.set_ylabel('NN predicted P(win)')
    ax.set_ylim(0,1);ax.set_title('NN scores each CSP candidate',fontsize=11);ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    # CSP: the same problem under the four switchable propagation settings.
    ax=fig.add_subplot(grid[1,1])
    modes=[('BT',False,False),('BT+FC',True,False),('BT+AC-3',False,True),('BT+FC+AC-3',True,True)]
    stats=[ChunkCSP(len(level.chunks),level.difficulty,level.seed).solve(fc,ac).stats for _,fc,ac in modes]
    for k,(field,colour) in enumerate((('assignments','#3b6fb5'),('backtracks','#e0507a'),('pruned','#e8b84a'))):
        ax.bar([i+(k-1)*.27 for i in range(4)],[getattr(s,field) for s in stats],.27,label=field,color=colour)
    ax.set_xticks(range(4));ax.set_xticklabels([m[0] for m in modes],fontsize=8)
    ax.set_title('CSP counters with and without FC / AC-3',fontsize=11);ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.2)
    # Search: work done by each algorithm to verify/solve the chosen level.
    ax=fig.add_subplot(grid[1,2])
    names=list(search_results)
    for k,(field,colour) in enumerate((('nodes_expanded','#3b6fb5'),('nodes_generated','#8fb0dd'))):
        ax.bar([i+(k-.5)*.4 for i in range(len(names))],[search_results[n][field] for n in names],.4,
               label=field.replace('_',' '),color=colour)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n.upper()}\ncost {search_results[n]['path_cost']}" for n in names],fontsize=8)
    ax.set_yscale('log');ax.set_title('Search on the chosen level',fontsize=11);ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.2)
    fig.suptitle('NEUROFORGE integrated demo: NN ranks CSP candidates, search verifies and solves',fontsize=13)
    fig.tight_layout()
    fig.savefig(dest,dpi=150);plt.close(fig)
    return dest

def launch_gui(seed=42,arena=False,forget_every=3,forget_fraction=.35,feature_dropout=.12):
    # Plain `python run_demo.py` opens the title screen; --arena skips it.
    mode='arena' if arena else 'title'
    while mode in ('title','manual','arena','expedition'):
        if mode=='title':
            from game.title import run_title
            mode=run_title(seed)
        elif mode=='expedition':
            from game.expedition_app import run_expedition
            mode=run_expedition(seed)
        elif mode=='arena':
            from game.arena_app import run_arena
            mode=run_arena(seed,forget_every=forget_every,
                           forget_fraction=forget_fraction,feature_dropout=feature_dropout)
        else:
            from game.app import run
            mode=run(seed)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--headless',action='store_true',help='original integrated demo')
    ap.add_argument('--arena',action='store_true',help='four simultaneous graphical agents')
    ap.add_argument('--arena-headless',action='store_true',help='four-agent simulation without GUI')
    ap.add_argument('--rounds',type=int,default=12,help='attempts per agent in --arena-headless')
    ap.add_argument('--forget-every',type=int,default=3,help='ten-example updates between partial resets')
    ap.add_argument('--forget-fraction',type=float,default=.35,help='fraction of adapter features erased')
    ap.add_argument('--feature-dropout',type=float,default=.12,help='online train-time dropout rate')
    ap.add_argument('--no-forgetting',action='store_true',help='headless ablation')
    ap.add_argument('--surprises',action='store_true',help='enable stochastic world changes in headless arena')
    ap.add_argument('--no-report',action='store_true',help='open the game without the printed demo and plot')
    args=ap.parse_args()
    if args.arena_headless:
        from integration.arena import run_headless_arena
        result=run_headless_arena(args.seed,args.rounds,forget_every=args.forget_every,
                                  forget_fraction=args.forget_fraction,
                                  feature_dropout=args.feature_dropout,
                                  forgetting_enabled=not args.no_forgetting,
                                  surprises=args.surprises)
        out=Path(__file__).resolve().parent/'results'/'arena_demo_summary.json'
        out.write_text(json.dumps(result,indent=2))
        print('FOUR-AGENT ARENA (search controls action; NN coaches levels)')
        for agent in result['agents']:
            print(agent['agent'].upper(),'wins',agent['wins'],'losses',agent['losses'],
                  'surprises',agent['total_surprises'],
                  'updates',agent['online_updates'],'forget events',agent['forget_events'],
                  'challenge',agent['challenge'])
        print('Saved:',out)
    elif args.headless:headless(args.seed)
    else:
        # The plain command first prints the integrated demo and saves its
        # cooperation plot, then opens the playable game.
        if not args.arena and not args.no_report:headless(args.seed)
        try:launch_gui(args.seed,arena=args.arena,forget_every=args.forget_every,
                       forget_fraction=args.forget_fraction,
                       feature_dropout=args.feature_dropout)
        except Exception as exc:
            if not (isinstance(exc,RuntimeError) or
                    type(exc).__module__.startswith('pygame')):
                raise
            print('GUI unavailable:',exc)
            if args.arena:
                from integration.arena import run_headless_arena
                result=run_headless_arena(args.seed,rounds=min(12,args.rounds),
                                          forget_every=args.forget_every)
                for agent in result['agents']:
                    print(agent['agent'].upper(),agent['wins'],'wins;',
                          agent['online_updates'],'online updates')
            else:
                headless(args.seed)
