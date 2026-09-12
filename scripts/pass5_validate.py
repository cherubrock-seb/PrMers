#!/usr/bin/env python3
"""Pass-5/R5 PRP validation: same-version -use gain plus Pass4 non-regression."""
import os,sys,subprocess,time,json,statistics,shutil,zipfile,re,hashlib,signal
from pathlib import Path
ROOT=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[3]).resolve() if len(sys.argv)>3 else ROOT/'pass5-results'
DEVICE=sys.argv[4] if len(sys.argv)>4 else '0'
EXPS=[21000029,70000001,100000007,147800003,150000007,180000007,196999969,197000003,210000017]
REPORTER='INPLACE=1,LOADS=10040,STORES=22,TABMUL_CHAIN32=1,MODM31=2,ZEROHACK_W=0'
count=0;records=[]
def prepare():
 shutil.copytree(ROOT/'third_party/aevum',OUT/'baseline',ignore=shutil.ignore_patterns('build-*','__pycache__','*.o','*.d'),dirs_exist_ok=True)
 for f in (ROOT/'scripts/pass5-baseline/third_party/aevum').rglob('*'):
  if f.is_file():shutil.copy2(f,OUT/'baseline'/f.relative_to(ROOT/'scripts/pass5-baseline/third_party/aevum'))
 (OUT/'cases').mkdir(exist_ok=True)
def environment():
 # No inherited manual plan/kernel override may contaminate runtime tuning.
 env={k:v for k,v in os.environ.items() if not k.startswith(('AEVUM_','PRMERS_AEVUM_'))}
 env.update(AEVUM_AUTOTUNE_CACHE=str(OUT/'cache-shape.tsv'),AEVUM_AUTOTUNE='off',AEVUM_PRP_USE_TUNE='off')
 return env
def engine(p,plan='',profile=None,variant='new',runtime=None,epoch=False,mode='prp',n=256,profiling=False,use_runtime=None):
 global count
 count+=1;case=OUT/'cases'/f'{count:04d}-{p}-{mode}';case.mkdir(parents=True)
 env=environment()
 if runtime:env.update(AEVUM_AUTOTUNE=runtime,AEVUM_PRP_USE_TUNE=runtime)
 if use_runtime:env['AEVUM_PRP_USE_TUNE']=use_runtime
 if profile is not None:env['AEVUM_PRP_USE']=profile
 if epoch:env['AEVUM_PRP_CARRY_EPOCH']='1'
 if profiling:env['AEVUM_PROFILE_KERNELS']='1'
 lib=OUT/'baseline/build-engine/libaevum_engine.so' if variant=='old' else ROOT/'third_party/aevum/build-engine/libaevum_engine.so'
 args=[OUT/'bench',lib,DEVICE,str(p),plan,ROOT/'third_party/aevum',mode,str(n),case/'residue.bin','1']
 start=time.monotonic();rc=127
 with (case/'run.log').open('w') as log:
  try:
   proc=subprocess.Popen(list(map(str,args)),cwd=OUT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=proc.wait(timeout=120)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
  except OSError as e:log.write(str(e))
 txt=(case/'run.log').read_text(errors='replace')
 row={'id':count,'p':p,'shape':plan,'profile':profile,'variant':variant,'runtime':runtime,'epoch':epoch,'rc':rc,'wall_s':time.monotonic()-start,'path':str(case)}
 for line in txt.splitlines():
  if line.startswith('AEVUM_BENCH '):row.update(json.loads(line.split(' ',1)[1]))
 shapes=re.findall(r'FFT:\s+\S+\s+(\S+)',txt)
 if shapes:row['shape']=shapes[-1]
 uses=re.findall(r'AEVUM_PRP_USE source=(\S+) shape=(\S+) profile=(\S+) gain=(\S+)',txt)
 if uses:
  source,shape,use,gain=uses[-1];row.update(source=source,shape=shape,profile='' if use=='defaults' else use,cached_gain=float(gain))
 row['kernels']={k:{'calls':int(c),'exec_ns':int(ns)} for k,c,ns in re.findall(r'AEVUM_PROFILE name=(\S+) calls=(\d+) exec_ns=(\d+)',txt)}
 row['searches']=len(re.findall(r'AEVUM_PRP_USE search|Aevum autotune: candidate=',txt))
 row['implementation_searches']=len(re.findall(r'AEVUM_PRP_USE search',txt))
 row['shape_cache_hit']='Aevum autotune: cache hit' in txt
 decision=re.findall(r'AEVUM_USE_DECISION state=(\S+)',txt)
 if decision:row['decision']=decision[-1]
 resume=re.findall(r'AEVUM_USE_DECISION state=\S+ screened=(\d+)/(\d+) full_candidates=(\d+) finalized=(\d+)/(\d+).*?next_screen=(\d+) next_finalist=(\d+)',txt)
 if resume:
  screened,planned,full_candidates,finalized,finalists,next_screen,next_finalist=map(int,resume[-1])
  row.update(use_screened=screened,use_planned=planned,use_full_candidates=full_candidates,
             use_finalized=finalized,use_finalists=finalists,use_next_screen=next_screen,use_next_finalist=next_finalist)
 budget=re.findall(r'AEVUM_PRP_USE search .*?budget=(\d+)ms',txt)
 if budget:row['use_budget_ms']=int(budget[-1])
 row['use_cache_records']=[]
 for cache in Path(str(OUT/'cache-shape.tsv')+'.prp-use-v4').glob('*.tsv'):
  for line in cache.read_text(errors='replace').splitlines():
   fields=line.split('\t')
   if len(fields)==8 and f'|p={p}|' in fields[1] and f'|shape={row["shape"]}|' in fields[1]:
    row['use_cache_records'].append({'file':cache.name,'profile':fields[2]})
 f=case/'residue.bin'
 if f.exists():row['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
 records.append(row);(OUT/'measurements.json').write_text(json.dumps(records,indent=2))
 if rc or (mode!='check' and 'seconds' not in row):raise RuntimeError(f'case {count}: rc={rc}')
 return row
def same(a,b,keep_left=False):
 left=Path(a['path'])/'residue.bin';right=Path(b['path'])/'residue.bin'
 if left.read_bytes()!=right.read_bytes():raise RuntimeError(f'WORD MISMATCH p={a["p"]}, cases {a["id"]}/{b["id"]}')
 if not keep_left:left.unlink()
 right.unlink()
def reject_cached_use(p):
 # Evict only this exponent; preserve every other validated band/shape record.
 for f in Path(str(OUT/'cache-shape.tsv')+'.prp-use-v4').glob('*.tsv'):
  if f'|p={p}|' in f.read_text(errors='replace'):f.unlink()
def _pair_core(p,shape,profile,baseline_variant='new',epoch=False):
 aa=[];bb=[];ids=[]
 for i in range(3):
  # -use validation must compare the SAME R5 library/shape with and without
  # the profile.  Pass4 comparison is a separate non-regression gate.
  if epoch:
   f=lambda:engine(p,shape,profile,variant='new')
   g=lambda:engine(p,shape,profile,epoch=True,variant='new')
  else:
   f=lambda:engine(p,shape,'',variant=baseline_variant)
   g=lambda:engine(p,shape,profile,variant='new')
  if i%2:b=g();a=f()
  else:a=f();b=g()
  same(a,b);aa.append(a['seconds']);bb.append(b['seconds']);ids.append([a['id'],b['id']])
 ratios=[x/y for x,y in zip(aa,bb)];gain=statistics.median(aa)/statistics.median(bb)
 return {'p':p,'shape':shape,'profile':profile,'epoch':epoch,'baseline_variant':baseline_variant,
         'word_exact':True,'baseline_s':aa,'candidate_s':bb,'engine_gain':gain,
         'repeatable':sum(x>=1.02 for x in ratios)>=2,'no_regression':min(ratios)>=0.985,
         'accepted':gain>=1.03 and sum(x>=1.02 for x in ratios)>=2 and min(ratios)>=0.985,'cases':ids}
def pair(p,shape,profile,epoch=False):
 return _pair_core(p,shape,profile,'new',epoch)
def pair_pass4(p,shape,profile=''):
 # Overall release non-regression: Pass4 same shape/defaults versus R6 final.
 # Radeon DVFS occasionally produced one isolated 3-5% outlier in an otherwise
 # flat/positive triplet. Five alternating pairs make this a release gate rather
 # than an outlier detector, while still rejecting persistent >=1.5% regressions.
 aa=[];bb=[];ids=[]
 for i in range(5):
  f=lambda:engine(p,shape,None,variant='old')
  g=lambda:engine(p,shape,profile,variant='new')
  if i%2:b=g();a=f()
  else:a=f();b=g()
  same(a,b);aa.append(a['seconds']);bb.append(b['seconds']);ids.append([a['id'],b['id']])
 ratios=[x/y for x,y in zip(aa,bb)];gain=statistics.median(aa)/statistics.median(bb)
 return {'p':p,'shape':shape,'profile':profile,'word_exact':True,'baseline_s':aa,'candidate_s':bb,
         'engine_gain':gain,'pair_ratios':ratios,
         'no_regression':gain>=0.985 and sum(x>=0.97 for x in ratios)>=4,'cases':ids}

def cold_auto(p,shape='',warm=False):
 # A use-only warm retune freezes the shape, preserving the shape-cache decision.
 a=engine(p,shape,runtime='auto' if warm else 'retune',use_runtime='retune' if warm else None)
 b=engine(p,shape,runtime='auto');same(a,b)
 if a.get('use_budget_ms',100)<100 and a.get('source')!='deferred':
  raise RuntimeError('zero budget was finalized instead of DEFERRED')
 if a.get('source')=='deferred':
  prior_positive=warm and a.get('use_cache_records') and all(
      r['profile'] not in ('defaults','defaults-complete') for r in a['use_cache_records'])
  if prior_positive and b.get('source')=='cache-hit':return a,b
  if a.get('use_cache_records'):raise RuntimeError('DEFERRED wrote a final use-cache record')
  if not b.get('implementation_searches'):raise RuntimeError('AUTO did not retry deferred use tuning')
  if not shape and not b.get('shape_cache_hit'):raise RuntimeError('AUTO lost the shape-cache hit')
 elif b['searches'] or b.get('source')!='cache-hit':
  raise RuntimeError('completed implementation decision did not cache-hit')
 return a,b
def validate_final(summary,p,shape,choice,stage):
 profile=choice.get('profile','');result=None;error=None
 try:result=pair(p,shape,profile)
 except Exception as exc:error=str(exc)
 # A completed-negative/default implementation decision has no candidate
 # to promote or evict.  Keep its old-vs-new timing as a diagnostic, but do
 # not turn normal GPU run-to-run jitter into a false implementation failure.
 # Exact word equality remains mandatory; optimized profiles still use every
 # >=3% / repeatability / no-regression acceptance gate below.
 valid=bool(result and result['word_exact'] and (result['accepted'] if profile else True))
 if result:
  result['stage']=stage;summary.setdefault('engine',[]).append(result)
 final={'p':p,'shape':shape,'profile':profile,'stage':stage,
        'tuner_internal_gain':choice.get('cached_gain'),
        'independent_engine_gain':result['engine_gain'] if result else None,
        'independently_accepted':bool(profile and valid),
        'status':('accepted' if profile else 'defaults') if valid else 'rejected',
        'independent_ab':result,'error':error}
 summary.setdefault('final_implementation',{})[str(p)]=final
 if not valid:
  reject_cached_use(p)
  summary.setdefault('failures',[]).append({'case':stage,'error':error or 'independent acceptance failed',
                                          'action':'REVERT cached use; CONTINUE'})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 return result if valid else None
def run():
 reporter=len(sys.argv)>5 and sys.argv[5]=='--reporter'
 exps=[147800003,180000007,196999969] if reporter else EXPS
 summary={'engine':[],'startup':[],'structural':[],'failures':[],'profiles':[],
          'end_to_end':'not measured; engine throughput and startup are separate', 'exponents':exps}
 def save(): (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 def attempt(label,fn):
  try:return fn()
  except Exception as e:summary['failures'].append({'case':label,'error':str(e),'action':'REVERT candidate; CONTINUE'});save();return None
 from aevum_bench_common import oracle
 # Check p21000029 even in the short external reporter run.
 for p in ([21000029] if reporter else [1362763,21000029]):
  for profile in ['',REPORTER]:
   def check(p=p,profile=profile):
    a=engine(p,mode='check',variant='old',n=1);b=engine(p,a['shape'],profile,mode='check',n=1)
    if p==1362763:oracle(b)
    same(a,b)
   attempt(f'regression-{p}-{profile}',check)
 for p in exps:
  def cold_hit(warm=False,shape=''):
   a,b=cold_auto(p,shape,warm)
   if b.get('source')=='deferred':raise RuntimeError('implementation still DEFERRED; no final cache written')
   if reporter:
    expected='1:512:8:512:202' if p==147800003 else '4:512:8:512:202'
    if b['shape']!=expected:raise RuntimeError(f'5090 shape non-regression gate: expected {expected}, got {b["shape"]}')
   summary['startup'].append({'p':p,'cold_wall_s':a['wall_s'],'hit_wall_s':b['wall_s'],
       'cold_create_s':a.get('create_seconds'),'hit_create_s':b.get('create_seconds'),
       'cold_case':a['id'],'hit_case':b['id'],'shape':b['shape'],'profile':b.get('profile','')})
   return b
  choice=attempt(f'runtime-{p}',cold_hit)
  if choice is None:reject_cached_use(p);continue
  shape=choice['shape'];profile=choice.get('profile','')
  result=validate_final(summary,p,shape,choice,f'implementation-{p}')
  if result is None:save();continue
  # If a cold compiler exhausted the budget, independently test the historical
  # full vector once, then retry cache population with its binaries warm.
  if profile=='' and ('rtx' in OUT.name.lower()):
   full=attempt(f'reporter-vector-{p}',lambda:pair(p,shape,REPORTER))
   if full:
    full['candidate']='reporter-vector';summary['engine'].append(full)
    if full['accepted']:
     warmed=attempt(f'warm-retune-{p}',lambda:cold_hit(True,shape))
     if warmed and warmed.get('profile','')!=profile:
      if validate_final(summary,p,shape,warmed,f'final-warm-profile-{p}') is None:save();continue
      profile=warmed.get('profile','');choice=warmed
  def profile_run(epoch=False):
   a=engine(p,shape,profile if epoch else None,variant='new' if epoch else 'old',profiling=True)
   b=engine(p,shape,profile,epoch=epoch,profiling=True)
   same(a,b);summary['profiles'].append({'p':p,'epoch':epoch,'baseline_case':a['id'],'candidate_case':b['id'],
       'before':a['kernels'],'after':b['kernels'],
       'kernel_speedups':{k:v['exec_ns']/b['kernels'][k]['exec_ns'] for k,v in a['kernels'].items()
           if k in b['kernels'] and v['calls']==b['kernels'][k]['calls'] and b['kernels'][k]['exec_ns']}})
  attempt(f'profile-{p}',profile_run)
  if not reporter and os.environ.get('AEVUM_PASS5_TEST_EPOCH')=='1' and not os.environ.get('AEVUM_PASS5_SKIP_EPOCH'):
   # Word differential before any timing acceptance; LL/sub/aliases included.
   def structural_check():
    a=engine(p,shape,profile,mode='check',n=1);b=engine(p,shape,profile,epoch=True,mode='check',n=1);same(a,b);return True
   if attempt(f'epoch-exact-{p}',structural_check):
    trial=attempt(f'epoch-{p}',lambda:pair(p,shape,profile,True))
    if trial:
     summary['structural'].append(trial)
     if trial['accepted']:attempt(f'epoch-profile-{p}',lambda:profile_run(True))
  save()
 gains=[x['independent_engine_gain'] for x in summary.get('final_implementation',{}).values() if x['status']!='rejected']
 if gains:summary['geomean_engine_gain']=statistics.geometric_mean(gains)
 save()
def focused():
 target=sys.argv[5].lower() if len(sys.argv)>5 else 'rtx3080'
 if target not in ('rtx3080','radeonvii'):raise ValueError('focused target must be rtx3080 or radeonVII')
 p=21000029 if target=='rtx3080' else 180000007
 reduced='INPLACE=1,LOADS=10040,STORES=22'
 summary={'target':target,'p':p,'engine':[],'failures':[],'runtime':[],
          'end_to_end':'not measured','structural':'OFF; rejected experiments not repeated'}
 def save(): (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 def attempt(label,fn):
  try:return fn()
  except Exception as exc:
   summary['failures'].append({'case':label,'error':str(exc),'action':'REJECT; continue bounded checks'});save();return None
 def runtime(warm=False,shape=''):
  try:a,b=cold_auto(p,shape,warm)
  except Exception:
   reject_cached_use(p);raise
  summary['runtime'].append({'stage':'warm-use-retune' if warm else 'cold-then-auto',
      'first':a,'auto':b,'deferred_observed':a.get('source')=='deferred',
      'no_final_cache_when_deferred':not a.get('use_cache_records') if a.get('source')=='deferred' else None})
  save();return b
 def trace(shape,profile):
  a=engine(p,shape,variant='old',mode='check',n=1)
  b=engine(p,shape,profile,mode='check',n=1);same(a,b);return True
 choice=attempt('cold-then-auto',runtime)
 if choice is None:save();return
 shape=choice['shape'];warmed=False
 # Deferred implementation tuning is resumable. Bound the focused harness
 # to a finite number of AUTO continuations so slow Radeon OpenCL compilation
 # can advance candidate-by-candidate without any infinite retune loop.
 if choice.get('source')=='deferred':
  for resume_round in range(1,11):
   again=attempt(f'deferred-auto-{resume_round}',lambda:engine(p,runtime='auto'))
   if not again:break
   choice=again;summary['runtime'].append({'stage':f'deferred-auto-{resume_round}','auto':again})
   save()
   if choice.get('source')!='deferred':break
 if target=='rtx3080' and not choice.get('profile'):
  retry=attempt('warm-use-retune',lambda:runtime(True,shape));warmed=True
  if retry:choice=retry
 def selected(candidate,stage):
  if candidate.get('source')=='deferred':
   summary.setdefault('final_implementation',{})[str(p)]={'status':'deferred','profile':'',
       'tuner_internal_gain':None,'independent_engine_gain':None,'independently_accepted':False}
   save();return None
  profile=candidate.get('profile','')
  if p==21000029 and not attempt(stage+'-word-trace',lambda:trace(shape,profile)):
   reject_cached_use(p)
   summary.setdefault('final_implementation',{})[str(p)]={'status':'rejected','profile':profile,
       'tuner_internal_gain':candidate.get('cached_gain'),'independent_engine_gain':None,
       'independently_accepted':False,'error':'word-trace failed'}
   save();return None
  return validate_final(summary,p,shape,candidate,stage)
 result=selected(choice,'final-selected-profile')
 if target=='rtx3080':
  if choice.get('profile','')==reduced:
   probe=result
  elif attempt('reduced-memory-word-trace',lambda:trace(shape,reduced)):
   probe=attempt('reduced-memory-independent',lambda:pair(p,shape,reduced))
  else:probe=None
  summary['reduced_memory_profile']={'profile':reduced,'independently_accepted':bool(probe and probe['accepted']),
      'independent_engine_gain':probe['engine_gain'] if probe else None,'independent_ab':probe}
  # A newly discovered warm winner always receives a NEW independent A/B x3.
  if probe and probe['accepted'] and choice.get('profile','')!=reduced and not warmed:
   retry=attempt('warm-use-retune',lambda:runtime(True,shape))
   if retry and retry.get('profile','')!=choice.get('profile',''):
    choice=retry;result=selected(choice,'final-warm-profile')
    if choice.get('profile','')==reduced:
     summary['reduced_memory_profile'].update(independently_accepted=bool(result and result['accepted']),
         independent_engine_gain=result['engine_gain'] if result else None,independent_ab=result)
 final=summary.get('final_implementation',{}).get(str(p),{})
 if final.get('status') in ('accepted','defaults'):
  def hit():
   a=engine(p,shape,choice.get('profile',''))
   b=engine(p,runtime='auto');same(a,b)
   if b.get('source')!='cache-hit' or b['searches'] or b.get('profile','')!=choice.get('profile','') or b['shape']!=shape:
    raise RuntimeError('final accepted/default profile did not cache-hit unchanged')
   return b
  summary['cache_hit_after_independent_ab']=attempt('final-cache-hit',hit)
 save()
 print('FOCUSED_RESULT '+json.dumps({'p':p,'final':summary.get('final_implementation',{}).get(str(p)),
       'reduced_memory':summary.get('reduced_memory_profile'),'failures':summary['failures']}))

def ranges():
 target=sys.argv[5].lower() if len(sys.argv)>5 else 'rtx3080'
 if target not in ('rtx3080','radeonvii','rtx5090'):raise ValueError('ranges target must be rtx3080, radeonVII or rtx5090')
 requested=os.environ.get('AEVUM_RANGE_EXPS','').strip()
 exps=[int(x) for x in requested.split(',') if x.strip()] if requested else EXPS
 summary={'mode':'r5-use-range-validation','target':target,'exponents':exps,'ranges':{},'failures':[],
          'policy':'same-version R5 defaults vs R5 selected -use; Pass4 is a separate non-regression gate'}
 def save():(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 for p in exps:
  item={'p':p,'resume':[]};summary['ranges'][str(p)]=item;save()
  try:
   first=engine(p,runtime='retune');item['resume'].append(first)
   choice=first
   # R5 has at most 12 bounded screen candidates + 4 finalists.  Twenty AUTO
   # continuations is therefore a hard harness bound even on slow OpenCL JITs.
   for round_no in range(1,21):
    if choice.get('source')!='deferred':break
    choice=engine(p,runtime='auto');item['resume'].append(choice);save()
   if choice.get('source')=='deferred':raise RuntimeError('R5 -use search did not complete within 20 bounded resumes')
   shape=choice['shape'];profile=choice.get('profile','')
   item.update(shape=shape,profile=profile,decision=choice.get('decision'),tuner_gain=choice.get('cached_gain'))

   # Exact same-version implementation gain.  This is the authoritative -use
   # acceptance measurement and intentionally contains NO Pass4 library.
   if profile:
    use_ab=pair(p,shape,profile)
    item['same_version_use_ab']=use_ab
    if not use_ab['accepted']:
     reject_cached_use(p)
     raise RuntimeError(f'autotuner promoted -use that failed same-version A/B: {use_ab["engine_gain"]:.5f}x')
   else:
    # A completed negative is a valid result: no profile beat the strict gate.
    item['same_version_use_ab']={'profile':'','status':'defaults-complete','accepted':True}

   # Release-level gate: same explicit shape on Pass4 versus R5 final profile.
   release_ab=pair_pass4(p,shape,profile)
   item['pass4_nonregression_ab']=release_ab
   if not release_ab['no_regression']:
    raise RuntimeError(f'Pass4 non-regression failed: median {release_ab["engine_gain"]:.5f}x')

   # Final cache hit must be benchmark-free and reproduce both shape and -use.
   hit=engine(p,runtime='auto');item['final_cache_hit']=hit
   if hit.get('source')!='cache-hit' or hit.get('searches') or hit.get('shape')!=shape or hit.get('profile','')!=profile:
    raise RuntimeError('final AUTO did not reproduce shape/use cache hit exactly')
   item['status']='PASS'
  except Exception as exc:
   item['status']='FAIL';item['error']=str(exc)
   summary['failures'].append({'p':p,'error':str(exc)})
  save()
 passed=[x for x in summary['ranges'].values() if x.get('status')=='PASS']
 positive=[x['same_version_use_ab']['engine_gain'] for x in passed if x.get('profile') and x.get('same_version_use_ab',{}).get('accepted')]
 release=[x['pass4_nonregression_ab']['engine_gain'] for x in passed if 'pass4_nonregression_ab' in x]
 if positive:summary['geomean_selected_use_gain']=statistics.geometric_mean(positive)
 if release:summary['geomean_r5_vs_pass4_same_shape']=statistics.geometric_mean(release)
 summary['passed']=len(passed);summary['total']=len(exps);save()
 print('R5_RANGE_RESULT '+json.dumps({'target':target,'passed':summary['passed'],'total':summary['total'],
      'geomean_selected_use_gain':summary.get('geomean_selected_use_gain'),
      'geomean_r5_vs_pass4_same_shape':summary.get('geomean_r5_vs_pass4_same_shape'),
      'failures':summary['failures']}))

def pack():
 with zipfile.ZipFile(OUT/'tuning-output.zip','w',zipfile.ZIP_DEFLATED,9) as z:
  for f in OUT.rglob('*'):
   if not f.is_file() or 'baseline' in f.relative_to(OUT).parts or f.suffix not in ('.log','.json','.tsv','.txt'):continue
   data=f.read_bytes()
   if len(data)>300000:data=data[:50000]+b'\n... clipped ...\n'+data[-250000:]
   z.writestr(str(f.relative_to(OUT)),data)
if __name__=='__main__':globals()[sys.argv[1]]()
