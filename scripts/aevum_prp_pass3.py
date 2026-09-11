#!/usr/bin/env python3
"""PRP Pass 3: bounded plan search + structural middle-one fusion, fail and continue."""
import hashlib,json,math,os,shlex,shutil,statistics,sys
from pathlib import Path
import aevum_bench_runner as h
from aevum_bench_common import oracle,result_fields
ROOT=h.ROOT; OUT=h.OUT
BASE=h.BASE|{'AEVUM_PRP_MIDDLE1':'0','AEVUM_RADIX1K':'4'}
h.BASE=BASE
REPS=int(os.getenv('AEVUM_REPEATS','3'))


def prepare():
    for variant in ('baseline','optimized'):
        dst=OUT/'build'/variant
        def ignore(path,names):
            return h.ignore(path,names)+[n for n in names if n=='aevum_pass3_baseline']
        shutil.copytree(ROOT,dst,ignore=ignore)
        if variant=='baseline':
            shutil.copytree(ROOT/'scripts/aevum_pass3_baseline',dst,dirs_exist_ok=True)
        (dst/'third_party/aevum/src/bundle.cpp').unlink(missing_ok=True)
    files=ROOT/'scripts/aevum_pass3_baseline'
    (OUT/'source-sha256.json').write_text(json.dumps({str(p.relative_to(files)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files.rglob('*') if p.is_file()},indent=2))


def engine(device,p,plan,flags,mode='prp',n=256,baseline=False,profile=False):
    r=h.engine(flags,device,p,plan,mode,n,baseline,profile)
    text=(Path(r['path'])/'run.log').read_text(errors='replace')
    if 'may be too small' in text: raise h.CandidateFailure('FFT capacity warning: '+r['path'])
    r['authoritative_pass2']=r.pop('original_pass1')
    r['prp_middle1_active']='AEVUM_PRP_PATH middle1 fused' in text
    if flags.get('AEVUM_PRP_MIDDLE1')=='1' and not r['prp_middle1_active']:
        raise h.CandidateFailure('requested PRP fusion was not activated: '+r['path'])
    return r


def same(a,b,allow_plan_change=False):
    if not allow_plan_change:
        h.same(a,b);return
    # Reg API returns canonical compact words, so different plans may be compared.
    av=(Path(a['path'])/'residue.bin').read_bytes(); bv=(Path(b['path'])/'residue.bin').read_bytes()
    if a['p']!=b['p'] or a['mode']!=b['mode'] or av!=bv:
        raise h.CandidateFailure(f'WORD MISMATCH p={a["p"]} {a["mode"]}: {a["path"]} vs {b["path"]}')


def candidates(p):
    # At most 12 candidates/range. No old WMUL/limb/MUL3 candidates.
    if p==21000029:
        plans=['1:512:2:256:101','1:512:2:256:202','1:256:2:512:101',
               '1:512:1:512:101','1:512:1:512:202','1:1K:1:256:101']
    else:
        plans=['4:1K:4:512:202','4:1K:2:1K:202','1:512:8:512:101',
               '1:1K:2:1K:101','4:4K:1:512:202','1:4K:1:512:101']
    for plan in plans:
        yield 'plan',plan,BASE.copy()
        if plan.split(':')[2]=='1': yield 'fused_middle1',plan,BASE|{'AEVUM_PRP_MIDDLE1':'1'}


def paired(device,p,reference_plan,plan,flags,n,reference_original=True):
    times=[[],[]]
    for rep in range(REPS+1):
        pair={}
        for side in ([0,1] if rep%2==0 else [1,0]):
            pair[side]=engine(device,p,[reference_plan,plan][side],[BASE,flags][side],
                              n=n,baseline=reference_original and side==0)
            if rep: times[side].append(pair[side]['seconds'])
        same(pair[0],pair[1],reference_plan!=plan)
    med=[statistics.median(t) for t in times]
    return {'baseline_median_s':med[0],'candidate_median_s':med[1],
            'speedup':med[0]/med[1],'samples_s':times,'iterations':n}


def limited_e2e(device,selection):
    # Full small PRP only after the target-size engine gate, once per device.
    # Small equivalent shape validates integration, not large-plan performance.
    flags=selection['flags']; p=786433
    smallplan='1:256:1:256:101' if flags['AEVUM_PRP_MIDDLE1']=='1' else ''
    results=[]
    for original,conf,plan in [(True,BASE,''),(False,flags,smallplan)]:
        env=h.environment(conf,device,original)
        cmd=[OUT/'build/optimized/prmers',p,'-prp','-proof','0','-aevum','-d',device,'--noask','-f','@RUN_DIR@']
        if plan: cmd += ['-aevum-fft',plan]
        d,text,seconds=h.invoke(cmd,env,f'd{device}-e2e',timeout=3600,ok_codes=(0,1))
        if '[Backend Aevum] engine::Reg adapter active' not in text:
            raise h.CandidateFailure('end-to-end did not use AEVUM')
        if conf['AEVUM_PRP_MIDDLE1']=='1' and 'AEVUM_PRP_PATH middle1 fused' not in text:
            raise h.CandidateFailure('end-to-end fusion not active')
        text+='\n'+'\n'.join(f.read_text(errors='replace') for f in sorted(d.rglob('*.json')) if f.name!='command.json')
        results.append({'seconds':seconds,'result':result_fields(text,'prp'),'path':str(d),'plan':plan})
    if results[0]['result']!=results[1]['result']: raise h.CandidateFailure('full PRP mismatch')
    return {'p':p,'runs':results,'note':'One small completed PRP integration pair; not target-size end-to-end speedup evidence.'}


def run():
    if REPS<3: raise RuntimeError('AEVUM_REPEATS must be >=3')
    (OUT/'controls.json').write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('AEVUM_')},indent=2))
    for device in sys.argv[4].split(','):
        int(device); status={'device':device,'cases':[],'preserved':BASE,'failures':[]};h.summary.append(status)
        print(f'device {device}: PRP-only Pass 3',flush=True)
        try:
            for p in (1362763,21000029):
                a=engine(device,p,'',BASE,'check',1,True);b=engine(device,p,'',BASE,'check',1);same(a,b)
                if p==1362763: oracle(a);oracle(b)
            a=engine(device,21000029,'',BASE,'mul3',256,True)
            b=engine(device,21000029,'',BASE|{'AEVUM_FUSED_MUL3':'1'},'mul3',256);same(a,b)
            # Freeze LL: one paired guard, not a tuning campaign.
            ll=[]
            for rep in range(4):
                pair={}
                for side in ([0,1] if rep%2==0 else [1,0]):
                    pair[side]=engine(device,21000029,'',BASE,'ll',1024,side==0)
                same(pair[0],pair[1])
                if rep: ll.append([pair[0]['seconds'],pair[1]['seconds']])
            gain=statistics.median(t[0] for t in ll)/statistics.median(t[1] for t in ll)
            status['frozen_ll_guard']={'samples_s':ll,'engine_ratio':gain,'pass':gain>=.95}
            if gain<.95: raise h.CandidateFailure('frozen LL guard >5% regression; inspect load/clocks and rerun')
            status['baseline_valid']=True
        except (h.CandidateFailure,RuntimeError) as exc:
            status['baseline_failure']=str(exc);h.save();continue
        for p,reference_plan in [(21000029,''),(100000007,'4:512:8:512:202')]:
            entry={'p':p,'baseline_plan':reference_plan,'candidates':[],'selected':None};status['cases'].append(entry)
            try:
                pilot=engine(device,p,reference_plan,BASE,baseline=True)
                stock=engine(device,p,reference_plan,BASE);same(pilot,stock)
                n=int(os.getenv('AEVUM_BENCH_ITERS',str(min(4096,max(256,256*math.ceil(.20/pilot['seconds']))))))
                if n<256: raise RuntimeError('AEVUM_BENCH_ITERS must be >=256')
                reference_check=engine(device,p,reference_plan,BASE,'check',1,True)
            except (h.CandidateFailure,RuntimeError) as exc:
                entry['baseline_failure']=str(exc);h.save();continue
            for kind,plan,flags in candidates(p):
                attempt={'kind':kind,'plan':plan,'flags':flags,'retained':False};entry['candidates'].append(attempt)
                if kind=='fused_middle1' and os.getenv('AEVUM_SKIP_PRP_MIDDLE1')=='1':
                    attempt['failure']='candidate syntax gate failed';h.save();continue
                try:
                    b=engine(device,p,plan,flags);same(pilot,b,True)
                    b=engine(device,p,plan,flags,'check',1);same(reference_check,b,True)
                    # All 100M candidates also exercise the exact failing exponent at
                    # the SAME candidate plan (21M / 4M > the minimum 3 bpw).
                    if p!=21000029:
                        a=engine(device,21000029,'',BASE,baseline=True)
                        b=engine(device,21000029,plan,flags);same(a,b,True)
                    if kind=='fused_middle1':
                        small=f'{plan.split(":")[0]}:256:1:256:{plan.split(":")[-1]}'
                        a=engine(device,1362763,'',BASE,'check',1,True)
                        b=engine(device,1362763,small,flags,'check',1);same(a,b,True);oracle(b)
                        # Independent same-shape A/B isolates fusion from shape selection.
                        attempt['fusion_same_shape']=paired(device,p,plan,plan,flags,n,False)
                    timing=paired(device,p,reference_plan,plan,flags,n);attempt['engine']=timing
                    if kind=='fused_middle1':
                        diagnostic=engine(device,p,plan,flags,profile=True);same(pilot,diagnostic,True)
                        unfused=engine(device,p,plan,BASE,profile=True);same(unfused,diagnostic)
                        attempt['profile']={'same_shape_unfused':unfused,'fused':diagnostic}
                    eligible=timing['speedup']>=1.05
                    if eligible and (entry['selected'] is None or timing['speedup']>entry['selected']['engine']['speedup']):
                        if entry['selected']: entry['selected']['retained']=False
                        attempt['retained']=True;entry['selected']=attempt
                    print(f'  p={p} {kind} {plan}: {timing["speedup"]:.4f}x {"eligible" if eligible else "REVERT"}',flush=True)
                except (h.CandidateFailure,RuntimeError) as exc:
                    attempt['failure']=str(exc)
                    print(f'  p={p} {kind} {plan}: REVERT; {exc}',flush=True)
                h.save()
            # Retain baseline event evidence even if every candidate regresses.
            try: entry['baseline_profile']=engine(device,p,reference_plan,BASE,baseline=True,profile=True)
            except h.CandidateFailure as exc: entry['baseline_profile_failure']=str(exc)
            # Confirm and profile ONLY the winner. A failed confirmation removes it.
            chosen=entry['selected']
            if chosen:
                try:
                    confirm=paired(device,p,reference_plan,chosen['plan'],chosen['flags'],n)
                    entry['confirmation']=confirm
                    if confirm['speedup']<1.05: raise h.CandidateFailure('confirmation gain <5%')
                    a=engine(device,p,reference_plan,BASE,baseline=True,profile=True)
                    b=engine(device,p,chosen['plan'],chosen['flags'],profile=True);same(a,b,True)
                    entry['profile']={'baseline':a,'candidate':b,
                        'note':'Separate event diagnostic; sum of events can exceed wall time with overlapping queues.'}
                    before=sum(v['exec_ns'] for v in a['kernels'].values())
                    after=sum(v['exec_ns'] for v in b['kernels'].values())
                    if after: entry['profile']['summed_kernel_event_ratio']=before/after
                except (h.CandidateFailure,RuntimeError) as exc:
                    entry['revert_reason']=str(exc);chosen['retained']=False;entry['selected']=None
            h.save()
        winners=[e['selected'] for e in status['cases'] if e['selected']]
        if winners and os.getenv('AEVUM_E2E_VALIDATE','1')=='1':
            try: status['limited_end_to_end']=limited_e2e(device,max(winners,key=lambda a:a['engine']['speedup']))
            except (h.CandidateFailure,RuntimeError) as exc:
                status['limited_end_to_end']={'failure':str(exc)}
                for e in status['cases']:
                    if e['selected']: e['selected']['retained']=False;e['selected']=None
        else: status['limited_end_to_end']={'skipped':True,'reason':'No confirmed >=5% winner or explicitly disabled'}
        for e in status['cases']:
            if not e['selected']: continue
            c=e['selected']; env=OUT/f'prp-device{device}-p{e["p"]}.env'
            env.write_text('# Validated ONLY for this device and exponent. PRP only.\n'+
                ''.join(f'export {k}={shlex.quote(v)}\n' for k,v in c['flags'].items())+
                f'export AEVUM_VALIDATED_PRP_PLAN={shlex.quote(c["plan"])}\n'+
                f'# PrMers: ./prmers {e["p"]} -prp -aevum -d {device} -aevum-fft "$AEVUM_VALIDATED_PRP_PLAN"\n')
        h.save()
    print('Return tuning-output.zip. selected entries are authoritative; unselected candidates stay disabled.',flush=True)

if __name__=='__main__':
    try: {'prepare':prepare,'run':run,'pack':h.pack}[sys.argv[1]]()
    except Exception as exc:
        (OUT/'FAILED.txt').write_text(str(exc)+'\n');h.save();h.pack();raise
