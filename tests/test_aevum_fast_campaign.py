#!/usr/bin/env python3
"""Exercise real campaign acceptance control with deterministic failing GPU stand-ins."""
import importlib.util,os,sys,tempfile,contextlib,io
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
with tempfile.TemporaryDirectory() as tmp:
    saved=sys.argv;sys.argv=['aevum_bench_runner.py','run',str(root),tmp,'2']
    spec=importlib.util.spec_from_file_location('campaign',root/'scripts/aevum_bench_runner.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    calls=[]
    def check(device,p,plan,flags,reference_flags=None):
        calls.append((p,flags.copy()))
        if flags.get('AEVUM_CARRY_WMUL')=='4' or flags.get('AEVUM_GF61_LIMB32')=='1':
            raise m.CandidateFailure('injected WORD MISMATCH p=21000029')
    m.correctness=check
    m.engine=lambda flags,device,p,plan,mode='prp',iters=256,original=False,profile=False: {
        'seconds':0.2,'resolved_plan':'1:256:4:256:101','kernels':{},'resources':{}}
    m.same=lambda a,b:None
    m.paired=lambda *a,**k:{'speedup':1.1,'baseline_median_s':1.1,'candidate_median_s':1.0}
    m.e2e=lambda *a,**k:{'test_stub':True}
    with contextlib.redirect_stdout(io.StringIO()): m.run()
    cases=m.summary[0]['cases'];assert len(cases)==2
    for case in cases:
        stages=case['candidates']
        assert len(stages)==3
        assert 'correctness_failure' in stages[0] and not stages[0]['retained']
        assert stages[1]['retained']
        assert 'correctness_failure' in stages[2] and not stages[2]['retained']
        assert case['selected']['AEVUM_CARRY_WMUL']=='1'
        assert case['selected']['AEVUM_GF61_LIMB32']=='0'
        assert case['selected']['AEVUM_FUSED_LL']=='1'
    assert any(p==21000029 for p,_ in calls)
    sys.argv=saved
print('FAST candidate failure -> revert -> next candidate: PASS')
