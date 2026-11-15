"""Reproducible player/level data + optional VGLC geometry download.
IMPORTANT: gameplay outcome labels here are SIMULATED Bernoulli draws, NOT real
human data. VGLC maps are real level geometry and have no player outcomes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen, Request
import numpy as np
from csp.generator import generate

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data'/'generated'
# Pinned to one upstream commit, and each file is checked against its SHA-256,
# so every run downloads and preprocesses byte-identical real level data.
VGLC_COMMIT='0e86a8f31f20ecad4eaa5741ff061af88767f7fb'
VGLC_URL=f'https://raw.githubusercontent.com/TheVGLC/TheVGLC/{VGLC_COMMIT}/Super%20Mario%20Bros/Processed/'
VGLC_SHA256={
    'mario-1-1.txt':'a04d56ae583819ebb39d2205b7322683eabaa2dd4b8b99960ee4243d43de4ac8',
    'mario-1-2.txt':'452cc0dacb4f7d694056f7eaaaa92c5c5ad9b01b257dd1d73bb8465f34d80b31',
    'mario-2-1.txt':'46b6c852051882baed2e0168e333211d7846c002373d03c3673d3301c00ec3a2',
    'mario-3-1.txt':'83375e929561ff0d82b70c50b7dba81de0ab4a550c04244c883d1226b82d68a2',
}
VGLC_NAMES=tuple(VGLC_SHA256)
PLAYER_FEATURES=('recent_win_rate','jump_accuracy','deaths_per_attempt','recent_progress','recent_time')
LEVEL_FEATURES=('length','gap_tiles','max_gap','hazards','coins','platforms','difficulty')
FEATURES=PLAYER_FEATURES+LEVEL_FEATURES
META_FIELDS=('persona_id','level_id','label_win','data_origin','latent_skill_simulator_only','p_win_simulator_only')


def generate_dataset(seed=42, personas=250, attempts=26, out=OUTPUT):
    """Generate data with *groupable* persona IDs and preverified level geometry.
    The simulated outcome model is deliberately different from the NN model.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    pool=[]
    for i in range(72):
        r=generate(seed=seed+i*41,target=float(0.05+0.9*((i*37)%71)/70),n_chunks=6)
        if r.level is not None:pool.append((i,r.level))
    if len(pool)<40:raise RuntimeError('Insufficient search-verified level pool')
    rows=[]
    for persona in range(personas):
        skill=float(rng.uniform(0.12,0.94))
        # Profile features are noisy observations of *previous* simulated attempts.
        wr=float(np.clip(skill+rng.normal(0,0.12),0,1))
        accuracy=float(np.clip(0.28+skill*0.7+rng.normal(0,0.08),0,1))
        deaths=float(np.clip(2.8*(1-skill)+rng.normal(0,0.3),0,4))
        progress=float(np.clip(0.2+0.8*skill+rng.normal(0,0.09),0,1))
        recent_time=float(np.clip(105-65*skill+rng.normal(0,13),12,150))
        for _ in range(attempts):
            level_id,level=pool[int(rng.integers(len(pool)))]
            f=level.features()
            # Latent skill is used ONLY to simulate labels; never an input to NN.
            difficulty=(float(f['difficulty'])*0.57 +
                        min(f['max_gap']/4,1)*0.20 +
                        min(f['hazards']/5,1)*0.17 +
                        min(f['platforms']/10,1)*0.06)
            # Stochastic skill vs challenge, including an interaction with jump accuracy.
            z=4.0*(skill-difficulty)+1.3*(accuracy-0.5)*min(1,f['max_gap']/3)
            z+=float(rng.normal(0,0.42))
            p=float(1/(1+math.exp(-z)))
            row={'persona_id':persona,'level_id':level_id,'data_origin':'synthetic_simulation',
                 'latent_skill_simulator_only':round(skill,6),
                 'p_win_simulator_only':round(p,6),
                 'recent_win_rate':round(wr,6),'jump_accuracy':round(accuracy,6),
                 'deaths_per_attempt':round(deaths,6),'recent_progress':round(progress,6),
                 'recent_time':round(recent_time,6),
                 'label_win':int(rng.random()<p)}
            row.update({k:round(float(f[k]),6) for k in LEVEL_FEATURES})
            rows.append(row)
    dest=out/'attempts.csv'
    with dest.open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=META_FIELDS+FEATURES)
        writer.writeheader();writer.writerows(rows)
    (out/'manifest.json').write_text(json.dumps({
        'seed':seed,'personas':personas,'attempts_per_persona':attempts,
        'rows':len(rows),'level_pool_size':len(pool),'dataset_kind':'synthetic player-outcome simulation',
        'features':FEATURES,'label':'label_win','split_policy':'persona_id grouped',
        'source_vglc_note':'independently downloadable real level geometry; NOT source of win outcomes'},indent=2))
    return dest


def process_vglc(names=VGLC_NAMES,out=OUTPUT):
    """Download real geometry as raw text, save SHA256 manifest and derived metrics.
    This online step is explicit. Network failure is reported, not silently masked.
    """
    out=Path(out); raw=out/'vglc_raw';raw.mkdir(parents=True,exist_ok=True)
    manifest=[];rows=[]
    for name in names:
        url=VGLC_URL+name
        req=Request(url,headers={'User-Agent':'NEUROFORGE-Research-Demo/1.0'})
        with urlopen(req,timeout=20) as resp:data=resp.read()
        digest=hashlib.sha256(data).hexdigest()
        if name in VGLC_SHA256 and digest!=VGLC_SHA256[name]:
            raise ValueError(f'{name}: downloaded file does not match the pinned SHA-256')
        text=data.decode('utf-8').splitlines()
        (raw/name).write_bytes(data)
        width=max(map(len,text));height=len(text)
        solid=set('XQS?Bb[]<>')
        tiles=sum(c in solid for line in text for c in line)
        enemies=sum(line.count('E') for line in text)
        ground=text[-1] if text else ''
        longest_gap=0;running=0
        for c in ground:
            if c=='-':running+=1;longest_gap=max(running,longest_gap)
            else:running=0
        rows.append({'level_name':name,'width':width,'height':height,
                     'solid_tile_fraction':round(tiles/max(width*height,1),5),
                     'enemy_count':enemies,'longest_ground_gap':longest_gap})
        manifest.append({'file':name,'url':url,'sha256':digest})
    with (out/'vglc_features.csv').open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'vglc_sources.json').write_text(json.dumps(manifest,indent=2))
    return out/'vglc_features.csv'

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--personas',type=int,default=250)
    ap.add_argument('--attempts',type=int,default=26)
    ap.add_argument('--no-download',action='store_true',
                    help='skip downloading the real VGLC level geometry (offline use)')
    ap.add_argument('--download-vglc',action='store_true',
                    help='kept for compatibility; the download is now the default')
    args=ap.parse_args()
    print('Synthetic dataset:',generate_dataset(args.seed,args.personas,args.attempts))
    if not args.no_download:
        try:
            print('External level features:',process_vglc())
        except OSError as error:
            # The synthetic file above does not depend on the download.
            print('VGLC DOWNLOAD FAILED:',error)
            print('The synthetic dataset was still written. Re-run with internet access, '
                  'or pass --no-download to skip this step.')
            raise SystemExit(1)
