from pathlib import Path
import numpy as np
from data.make_data import generate_dataset,FEATURES
from nn.model import MLP,LogisticBaseline
from integration.adaptive import AdaptiveController,PlayerProfile
from csp.generator import generate


def test_dataset_reproducible(tmp_path):
    a=generate_dataset(5,personas=4,attempts=4,out=tmp_path/'a')
    b=generate_dataset(5,personas=4,attempts=4,out=tmp_path/'b')
    assert a.read_bytes()==b.read_bytes()
    assert a.read_text().count('\n')==17


def test_nn_learns_and_saves(tmp_path):
    rng=np.random.default_rng(4)
    x=rng.normal(size=(180,len(FEATURES)))
    y=((x[:,0]+x[:,1])>0).astype(int)
    model=MLP(seed=4);hist=model.train(x,y,epochs=35,seed=4)
    assert hist[-1]<hist[0]
    model.save(tmp_path/'test.npz')
    other=MLP.load(tmp_path/'test.npz')
    assert np.allclose(other.predict_proba(x[:10]),model.predict_proba(x[:10]))
    base=LogisticBaseline();base.train(x,y)
    assert np.all((base.predict_proba(x)>0)&(base.predict_proba(x)<1))


def test_real_telemetry_affects_profile(tmp_path):
    level=generate(seed=5).level
    p=PlayerProfile()
    p.record(level,True,15,10,9,1,save=False)
    assert p.features()['recent_win_rate']==1
    assert p.features()['jump_accuracy']==0.9


def test_adaptive_candidates_search_valid():
    c=AdaptiveController(seed=9,use_nn=False)
    chosen=c.next_level(n_candidates=3)
    assert chosen.result.search.found
    assert 0<=chosen.predicted_win<=1
    assert len(c.last_candidates)==3
