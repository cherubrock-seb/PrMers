#!/usr/bin/env python3
"""PASS 2: FAST GPU candidate screening; numerical failures reject only that candidate."""
import hashlib,json,math,os,re,shutil,statistics,subprocess,sys,time,zipfile
from pathlib import Path
from aevum_bench_common import oracle,result_fields

ROOT=Path(sys.argv[2]).resolve(); OUT=Path(sys.argv[3]).resolve()
BASE={'AEVUM_FUSED_LL':'1','AEVUM_FUSED_MUL3':'0','AEVUM_CACHE_BUFFER_ARGS':'0',
      'AEVUM_GF61_LIMB32':'0','AEVUM_REG_LEAD_CACHE':'1','AEVUM_PFA_LEAD_BRIDGE':'0'}
REPS=int(os.getenv('AEVUM_REPEATS','3'))
TIMEOUT=int(os.getenv('AEVUM_RUN_TIMEOUT','600'))
records=[]; summary=[]; serial=0

class CandidateFailure(RuntimeError): pass

def ignore(path,names):
    return [n for n in names if
        ((Path(path)/n).is_dir() and (n.startswith(('build-','build_','benchmark-results-','.aevum'))
         or n in {'.git','__pycache__','aevum_baseline','aevum_pass2_baseline','package'}))
        or ((Path(path)/n).is_file() and (n in {'prmers','aevum','prpll'}
           or n.endswith(('.o','.d','.so','.pyc','.exe','.dylib'))))]

def prepare():
    for name in ('baseline','optimized'):
        dst=OUT/'build'/name
        shutil.copytree(ROOT,dst,ignore=ignore)
        if name=='baseline': shutil.copytree(ROOT/'scripts/aevum_pass2_baseline',dst,dirs_exist_ok=True)
        (dst/'third_party/aevum/src/bundle.cpp').unlink(missing_ok=True)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'scripts/aevum_pass2_baseline').rglob('*') if p.is_file()}
    (OUT/'source-sha256.json').write_text(json.dumps(hashes,indent=2))

def save():
    (OUT/'measurements.json').write_text(json.dumps(records,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))

def environment(flags,device,original=False,profile=False):
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('AEVUM_'): env.pop(k)
    env.update(BASE); env.update(flags)
    env['AEVUM_PROFILE_KERNELS']='1' if profile else '0'
    env['AEVUM_ENGINE_LIB']=str(OUT/'build'/('baseline' if original else 'optimized')/
                              'third_party/aevum/build-engine/libaevum_engine.so')
    return env

def invoke(cmd,env,label,cwd=None,timeout=TIMEOUT,ok_codes=(0,)):
    global serial
    serial+=1; d=OUT/'runs'/f'{serial:04}-{label}';d.mkdir(parents=True)
    if cwd is None:
        cache=OUT/('jit-'+label.split('-')[0]);cache.mkdir(exist_ok=True)
        (d/'.aevum-kernel-cache').symlink_to(cache,target_is_directory=True)
    cmd=[str(d) if str(v)=='@RUN_DIR@' else str(v) for v in cmd]
    (d/'command.json').write_text(json.dumps({'argv':cmd,'flags':{k:v for k,v in env.items() if k.startswith('AEVUM_')}},indent=2))
    t=time.perf_counter()
    try:
        with (d/'run.log').open('w') as log:
            process=subprocess.run(cmd,env=env,cwd=cwd or d,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        text=(d/'run.log').read_text(errors='replace')
        if process.returncode not in ok_codes or re.search(r'FAIL:|Memory access fault|Abandon|Gerbicz[^\n]*[Ff]ail|Aevum .* failed',text):
            raise CandidateFailure(f'process failure rc={process.returncode}: {d}')
    except subprocess.TimeoutExpired:
        raise CandidateFailure(f'timeout: {d}')
    return d,text,time.perf_counter()-t

def engine(flags,device,p,plan,mode='prp',iters=256,original=False,profile=False):
    env=environment(flags,device,original,profile)
    tmp=OUT/f'temporary-d{device}.bin';tmp.unlink(missing_ok=True)
    cwd=OUT/f'engine-cache-d{device}';cwd.mkdir(exist_ok=True)
    cmd=[OUT/'engine-bench',env['AEVUM_ENGINE_LIB'],device,p,plan,
         OUT/'build/optimized/third_party/aevum',mode,iters,tmp]
    d,text,_=invoke(cmd,env,f'd{device}-{mode}',cwd)
    tmp.rename(d/'residue.bin')
    r={'path':str(d),'device':device,'p':p,'plan':plan,'mode':mode,'flags':flags.copy(),
       'original_pass1':original,'profile':profile,'sha256':hashlib.sha256((d/'residue.bin').read_bytes()).hexdigest()}
    if mode!='check':
        match=re.search(r'AEVUM_BENCH (\{[^\n]+\})',text)
        if not match: raise CandidateFailure(f'missing engine result: {d}')
        r.update(json.loads(match[1]))
    resolved=re.findall(r'FFT: [^\n]*? (\S+:\S+) \(',text)
    r['resolved_plan']=resolved[-1] if resolved else plan
    r['kernels']={name:{'calls':int(calls),'exec_ns':int(ns)} for name,calls,ns in
                 re.findall(r'AEVUM_PROFILE name=(\S+) calls=(\d+) exec_ns=(\d+)',text)}
    r['resources']={name:{'wg':int(wg),'local_bytes':int(loc),'private_bytes':int(priv),
                            'preferred_multiple':int(pref),'query_rc':rc} for name,wg,loc,priv,pref,rc in
        re.findall(r'AEVUM_RESOURCE name=(\S+) wg=(\d+) local_bytes=(\d+) private_bytes=(\d+) preferred=(\d+) rc=([0-9,-]+)',text)}
    records.append(r);return r

def same(a,b):
    av=(Path(a['path'])/'residue.bin').read_bytes();bv=(Path(b['path'])/'residue.bin').read_bytes()
    if av!=bv:
        first=next((i for i in range(min(len(av),len(bv))) if av[i]!=bv[i]),min(len(av),len(bv)))//4
        raise CandidateFailure(f'WORD MISMATCH p={a["p"]} mode={a["mode"]} word={first}: {a["path"]} vs {b["path"]}')
    if a.get('transform')!=b.get('transform') or a['resolved_plan']!=b['resolved_plan']:
        raise CandidateFailure('effective FFT plan changed during candidate comparison')

def correctness(device,p,plan,flags,reference_flags=BASE):
    for mode,n in [('check',1),('prp',256),('ll',128),('mul3',128),('mixed',32)]:
        a=engine(reference_flags,device,p,plan,mode,n,True)
        b=engine(flags,device,p,plan,mode,n);same(a,b)
        if mode=='check' and p==1362763: oracle(a);oracle(b)

def paired(device,p,plan,old,new,iters):
    times=[[],[]]
    for rep in range(REPS+1): # discard the first pair; every process also warms its kernels
        pair={}
        for i in ([0,1] if rep%2==0 else [1,0]):
            pair[i]=engine([old,new][i],device,p,plan,iters=iters)
            if rep: times[i].append(pair[i]['seconds'])
        same(pair[0],pair[1])
    med=[statistics.median(t) for t in times]
    return {'baseline_median_s':med[0],'candidate_median_s':med[1],
            'speedup':med[0]/med[1],'samples_s':times,'iterations':iters}

def e2e(device,flags):
    p=786433;results=[]
    # One pair only, after >=5% engine improvement. This is validation, not a median claim.
    for original,conf in [(True,BASE),(False,flags)]:
        env=environment(conf,device,original)
        d,text,seconds=invoke([OUT/'build/optimized/prmers',p,'-prp','-proof','0','-aevum',
                              '-d',device,'--noask','-f','@RUN_DIR@'],env,f'd{device}-e2e',timeout=3600,ok_codes=(0,1))
        if '[Backend Aevum] engine::Reg adapter active' not in text:
            raise CandidateFailure('end-to-end did not use AEVUM')
        text+='\n'+'\n'.join(f.read_text(errors='replace') for f in sorted(d.rglob('*.json')) if f.name!='command.json')
        try: signature=result_fields(text,'prp')
        except RuntimeError as exc: raise CandidateFailure(str(exc))
        results.append({'seconds':seconds,'result':signature,'path':str(d)})
    if results[0]['result']!=results[1]['result']: raise CandidateFailure('end-to-end result mismatch')
    return {'exponent':p,'runs':results,'note':'One completed PRP pair; no statistically validated end-to-end speedup.'}

def run():
    if REPS<3: raise RuntimeError('AEVUM_REPEATS must be >=3')
    (OUT/'controls.json').write_text(json.dumps({k:v for k,v in os.environ.items() if k.startswith('AEVUM_')},indent=2))
    for device in sys.argv[4].split(','):
        int(device)
        status={'device':device,'pass1_retained':BASE,'cases':[],'failures':[],'baseline_valid':False}
        summary.append(status); save()
        print(f'device {device}: baseline/oracle + p=21000029 safety regression',flush=True)
        try:
            correctness(device,1362763,'',BASE)
            correctness(device,21000029,'',BASE)
            # Even a stale environment requesting the invalid fusion must stay on generic MUL3.
            a=engine(BASE,device,21000029,'','mul3',256,True)
            b=engine(BASE|{'AEVUM_FUSED_MUL3':'1'},device,21000029,'','mul3',256);same(a,b)
            status['mul3_forced_flag_regression']='PASS: unsafe fusion cannot be enabled'
            status['baseline_valid']=True
        except (CandidateFailure,RuntimeError) as exc:
            status['failures'].append({'baseline_failure':str(exc)});save();continue
        for p,plan in [(21000029,''),(100000007,'4:512:8:512:202')]:
            selected=BASE.copy(); entry={'p':p,'plan':plan,'candidates':[]};status['cases'].append(entry)
            try: pilot=engine(BASE,device,p,plan,iters=256)
            except CandidateFailure as exc:
                entry['selected']=BASE.copy();entry['baseline_failure']=str(exc);save();continue
            # Aim for >=150ms/sample while bounding the screening cost on the large plan.
            iters=int(os.getenv('AEVUM_BENCH_ITERS',str(min(4096,max(256,256*math.ceil(.15/pilot['seconds']))))))
            if iters<256: raise RuntimeError('AEVUM_BENCH_ITERS must be >=256')
            entry['resolved_plan']=pilot['resolved_plan']
            # Carry batching is independently selected on each device/plan, then GF61 arithmetic.
            for name,delta in [('carry_batch4',{'AEVUM_CARRY_WMUL':'4'}),
                               ('carry_batch1',{'AEVUM_CARRY_WMUL':'1'}),
                               ('gf61_limb32',{'AEVUM_GF61_LIMB32':'1'})]:
                candidate=selected|delta;attempt={'candidate':name,'flags':candidate.copy(),'retained':False}
                entry['candidates'].append(attempt)
                if name=='gf61_limb32' and os.getenv('AEVUM_SKIP_GF61_LIMB32')=='1':
                    attempt['build_failure']='Enabled-kernel syntax gate failed; see host-tests.log';save();continue
                try:
                    correctness(device,p,plan,candidate)
                    # Every candidate must include the known failing exponent, including type-4 screening.
                    if p!=21000029: correctness(device,21000029,'',candidate)
                    if delta.get('AEVUM_GF61_LIMB32')=='1': correctness(device,1362763,'',candidate)
                    timing=paired(device,p,plan,selected,candidate,iters)
                    attempt['engine']=timing;attempt['retained']=timing['speedup']>1.05
                    if attempt['retained']: selected=candidate
                    print(f'  p={p} {name}: {"KEEP" if attempt["retained"] else "REVERT"} {timing["speedup"]:.4f}x',flush=True)
                except (CandidateFailure,RuntimeError) as exc:
                    attempt['correctness_failure']=str(exc)
                    print(f'  p={p} {name}: REVERT; {exc}',flush=True)
                save()
            # Only the resulting combination is checked on supplemental PFA plans.
            if selected!=BASE:
                try:
                    for extra in ('pfa:3','pfa:9'): correctness(device,21000029,extra,selected)
                    final=paired(device,p,plan,BASE,selected,iters);entry['final_engine_vs_pass1']=final
                    if final['speedup']<=1.05:
                        entry['revert_reason']='combined gain <=5%';selected=BASE.copy()
                except (CandidateFailure,RuntimeError) as exc:
                    entry['revert_reason']=str(exc);selected=BASE.copy()
            entry['selected']=selected
            try:
                a=engine(BASE,device,p,plan,iters=256,profile=True)
                b=engine(selected,device,p,plan,iters=256,profile=True);same(a,b)
                entry['kernel_profile']={'baseline':a['kernels'],'candidate':b['kernels'],
                    'note':'Separate 256-square diagnostic pair, not throughput samples.',
                    'baseline_resources':a['resources'],'candidate_resources':b['resources']}
                entry['kernel_mean_speedup']={k:(v['exec_ns']/v['calls'])/(b['kernels'][k]['exec_ns']/b['kernels'][k]['calls'])
                    for k,v in a['kernels'].items() if k in b['kernels'] and v['calls'] and b['kernels'][k]['exec_ns']}
            except CandidateFailure as exc:
                entry['profiling_failure']=str(exc)
                entry['selected']=BASE.copy()
            save()
        winners=[c for c in status['cases'] if c['selected']!=BASE]
        if winners and os.getenv('AEVUM_E2E_VALIDATE','1')=='1':
            best=max(winners,key=lambda c:c['final_engine_vs_pass1']['speedup'])
            try:
                correctness(device,786433,'',best['selected'])
                status['limited_end_to_end']=e2e(device,best['selected'])
            except (CandidateFailure,RuntimeError) as exc:
                status['limited_end_to_end']={'failure':str(exc),'reverted_all_pass2':True}
                for c in status['cases']: c['selected']=BASE.copy()
        else: status['limited_end_to_end']={'skipped':True,'reason':'No >=5% winner, or explicitly disabled'}
        # Per exponent/plan, never silently extrapolate one GPU's result to another.
        for c in status['cases']:
            envfile=OUT/f'retained-device-{device}-p{c["p"]}.env'
            envfile.write_text('# Tested on this device/exponent/plan only.\n'+
                ''.join(f'export {k}={v}\n' for k,v in c['selected'].items())+
                ('unset AEVUM_CARRY_WMUL\n' if 'AEVUM_CARRY_WMUL' not in c['selected'] else ''))
        save()
    print('FAST screening complete. See summary.json.',flush=True)

def pack():
    with zipfile.ZipFile(OUT/'tuning-output.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file() and p.suffix in {'.json','.txt','.env','.log'}: z.write(p,p.name)
        # These logs contain exact compiler options, resource failures and device identity.
        for p in (OUT/'runs').rglob('run.log'): z.write(p,str(p.relative_to(OUT)))
        for p in (OUT/'runs').rglob('command.json'): z.write(p,str(p.relative_to(OUT)))

if __name__=='__main__':
    try: {'prepare':prepare,'run':run,'pack':pack}[sys.argv[1]]()
    except Exception as exc:
        (OUT/'FAILED.txt').write_text(str(exc)+'\n');save();pack();raise
