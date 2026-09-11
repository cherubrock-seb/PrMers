#!/usr/bin/env python3
"""Exercise campaign rollback, continued screening, confirmation and E2E failure."""
import contextlib,importlib.util,io,json,os,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp);sys.argv=['aevum_prp_pass3.py','run',str(root),str(out),'2']
    spec=importlib.util.spec_from_file_location('pass3',root/'scripts/aevum_prp_pass3.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.same=lambda *a,**k:None;m.oracle=lambda r:None
    seen=[]
    def fake(device,p,plan,flags,mode='prp',n=256,baseline=False,profile=False):
        seen.append((p,plan,flags.copy()))
        if not baseline and plan=='1:512:2:256:101':
            raise m.h.CandidateFailure('injected WORD MISMATCH')
        if flags['AEVUM_PRP_MIDDLE1']=='1':
            raise m.h.CandidateFailure('injected GPU compilation failure')
        # The next plan wins; synthetic durations stay inside the test, never benchmark evidence.
        seconds=1.0 if baseline or not plan else .75
        return {'p':p,'plan':plan,'mode':mode,'seconds':seconds,'kernels':{},'path':tmp}
    m.engine=fake
    m.limited_e2e=lambda *a:{'test_only':'pass'}
    with contextlib.redirect_stdout(io.StringIO()): m.run()
    s=m.h.summary[0]
    assert s['baseline_valid'] and s['frozen_ll_guard']['pass']
    first=s['cases'][0]
    assert 'WORD MISMATCH' in first['candidates'][0]['failure']
    assert first['selected'] and first['selected']['plan']=='1:512:2:256:202'
    assert any(a['kind']=='fused_middle1' and 'failure' in a for a in first['candidates'])
    assert first['selected']['flags']['AEVUM_FUSED_LL']=='1'
    assert first['selected']['flags']['AEVUM_FUSED_MUL3']=='0'
    assert any(p==21000029 and plan=='4:1K:4:512:202' for p,plan,_ in seen)
    # A late full-PRP failure revokes all selected profiles and exports no new env.
    for p in out.glob('*.env'):p.unlink()
    m.h.summary.clear();m.h.records.clear()
    def fail(*a):raise m.h.CandidateFailure('injected full PRP mismatch')
    m.limited_e2e=fail
    with contextlib.redirect_stdout(io.StringIO()): m.run()
    assert all(c['selected'] is None for c in m.h.summary[0]['cases'])
    assert not list(out.glob('*.env'))
print('PASS: PRP campaign continues after mismatch/compile failure and revokes failed final selections')
