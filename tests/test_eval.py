from eval.run_experiments import search_experiment,csp_experiment,adaptation_experiment
from run_demo import headless

def test_search_experiment_data():
    rows,summary=search_experiment([10,20])
    assert len(rows)==8 and len(summary)==4
    assert all(r['found']==1 for r in rows)

def test_csp_experiment_data():
    rows,summary=csp_experiment([10,20])
    assert len(rows)==8 and len(summary)==4
    assert all(r['solved']==1 for r in rows)

def test_adaptation_experiment_data():
    rows,summary=adaptation_experiment([10],2,2)
    assert len(rows)==12 and len(summary)==3
    assert all(0<=r['predicted_win']<=1 for r in rows)

def test_headless_end_to_end():
    payload=headless(5)
    assert all(r['found'] for r in payload['search'].values())
    assert payload['impossible_case']=='No solution'
