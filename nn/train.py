"""Train/test NN and logistic baseline on group-disjoint synthetic personas."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from data.make_data import OUTPUT,FEATURES,generate_dataset
from nn.model import MLP,LogisticBaseline,MODEL_PATH
ROOT=Path(__file__).resolve().parents[1]

def read_rows(path):
    with open(path,encoding='utf-8') as f: return list(csv.DictReader(f))

def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p).ravel()
    eps=1e-8;pp=np.clip(p,eps,1-eps)
    # AUC via rank statistic; ties receive average ranks.
    order=np.argsort(p);ranks=np.zeros(len(p))
    sorted_p=p[order];i=0
    while i<len(p):
        j=i+1
        while j<len(p) and sorted_p[j]==sorted_p[i]:j+=1
        ranks[order[i:j]]=(i+1+j)/2
        i=j
    pos=y.sum();neg=len(y)-pos
    auc=float((ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg)) if pos>0 and neg>0 else None
    return {'accuracy':round(float(np.mean((p>=0.5)==y)),4),
            'brier':round(float(np.mean((p-y)**2)),4),
            'logloss':round(float(np.mean(-y*np.log(pp)-(1-y)*np.log(1-pp))),4),
            'roc_auc':None if auc is None else round(auc,4),
            'n':len(y)}

def train_pipeline(seed=42, dataset=None, epochs=105):
    if dataset is None:
        dataset=OUTPUT/'attempts.csv'
        if not Path(dataset).exists():generate_dataset(seed=seed)
    rows=read_rows(dataset)
    ids=np.array([int(r['persona_id']) for r in rows]);all_ids=np.sort(np.unique(ids))
    rng=np.random.default_rng(seed)
    rng.shuffle(all_ids)
    n=len(all_ids);a=max(1,int(n*0.7));b=max(a+1,int(n*0.85))
    tr_ids=set(all_ids[:a]);va_ids=set(all_ids[a:b]);te_ids=set(all_ids[b:])
    if not te_ids:raise ValueError('Need >= 3 persona groups')
    X=np.asarray([[float(r[f]) for f in FEATURES] for r in rows])
    y=np.asarray([int(r['label_win']) for r in rows])
    ix_tr=np.array([i for i,v in enumerate(ids) if v in tr_ids])
    ix_va=np.array([i for i,v in enumerate(ids) if v in va_ids])
    ix_te=np.array([i for i,v in enumerate(ids) if v in te_ids])
    nn=MLP(seed=seed);history=nn.train(X[ix_tr],y[ix_tr],epochs=epochs,seed=seed)
    nn.save()
    baseline=LogisticBaseline();baseline.train(X[ix_tr],y[ix_tr])
    result={'seed':seed,'dataset':str(dataset),'origin':'synthetic outcomes; NOT human play testing',
            'splits':{'train_personas':len(tr_ids),'val_personas':len(va_ids),
                      'test_personas':len(te_ids),'train_rows':len(ix_tr),
                      'val_rows':len(ix_va),'test_rows':len(ix_te)},
            'neural_network':{'validation':metrics(y[ix_va],nn.predict_proba(X[ix_va])),
                              'test':metrics(y[ix_te],nn.predict_proba(X[ix_te]))},
            'logistic_baseline':{'validation':metrics(y[ix_va],baseline.predict_proba(X[ix_va])),
                                 'test':metrics(y[ix_te],baseline.predict_proba(X[ix_te]))},
            'training_loss':history}
    path=ROOT/'results'/'training_metrics.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=105);args=p.parse_args()
    print(json.dumps(train_pipeline(args.seed,epochs=args.epochs),indent=2)[:1800])
