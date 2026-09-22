import importlib.util,json,os,sys,tempfile,unittest
from unittest import mock
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('pass5_campaign',ROOT/'scripts/pass5_validate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Campaign(unittest.TestCase):
 def test_inherited_tuning_is_not_a_runtime_override(self):
  os.environ['AEVUM_TUNE_DIR']='/empty/manual/directory'
  try:self.assertNotIn('AEVUM_TUNE_DIR',m.environment())
  finally:del os.environ['AEVUM_TUNE_DIR']
 def test_cache_rejection_preserves_other_exponents(self):
  with tempfile.TemporaryDirectory() as d:
   oldout=m.OUT;m.OUT=Path(d)
   try:
    cache=Path(str(m.OUT/'cache-shape.tsv')+'.prp-use-v4');cache.mkdir()
    rejected=cache/'rejected.tsv';rejected.write_text('prp-use-v4|p=21000029|use-flags=')
    kept=cache/'kept.tsv';kept.write_text('prp-use-v4|p=196999969|use-flags=')
    m.reject_cached_use(21000029)
    self.assertFalse(rejected.exists());self.assertTrue(kept.exists())
   finally:m.OUT=oldout
 def test_mismatch_does_not_abort_later_exponents(self):
  with tempfile.TemporaryDirectory() as d:
   m.OUT=Path(d);m.ROOT=ROOT;m.EXPS=[21000029,196999969]
   os.environ['AEVUM_PASS5_SKIP_EPOCH']='1'
   cache=Path(str(m.OUT/'cache-shape.tsv')+'.prp-use-v4');cache.mkdir()
   rejected=cache/'rejected.tsv';rejected.write_text('prp-use-v4|p=21000029|use-flags=')
   profiled=[];serial=0
   def fake(p,plan='',profile=None,variant='new',runtime=None,**kw):
    nonlocal serial
    if kw.get('profiling'):profiled.append((variant,profile))
    serial+=1;path=m.OUT/str(serial);path.mkdir();(path/'residue.bin').write_bytes(b'exact')
    return dict(id=serial,p=p,path=str(path),shape=plan or '1:512:8:512:202',profile=profile or '',source='cache-hit',searches=0,wall_s=1.,seconds=1.,kernels={})
   m.engine=fake;realpair=m.pair
   def injected(p,*args,**kw):
    if p==21000029:raise RuntimeError('WORD MISMATCH injected')
    return realpair(p,*args,**kw)
   m.pair=injected
   m.run()
   summary=json.loads((m.OUT/'summary.json').read_text())
   self.assertTrue(any('WORD MISMATCH' in f.get('error','') for f in summary['failures']))
   self.assertTrue(any(x['p']==196999969 for x in summary['engine']))
   self.assertFalse(rejected.exists())
   self.assertEqual(profiled,[('old',None),('new','')])
   m.pack()
   self.assertTrue((m.OUT/'tuning-output.zip').exists())
   del os.environ['AEVUM_PASS5_SKIP_EPOCH']
 def test_reporter_resume_completes_long_deferred_search(self):
  with tempfile.TemporaryDirectory() as d:
   oldout=m.OUT;m.OUT=Path(d);calls=[];serial=0
   try:
    def fake(p,plan='',profile=None,variant='new',runtime=None,**kw):
     nonlocal serial
     serial+=1;calls.append(runtime)
     path=m.OUT/str(serial);path.mkdir();(path/'residue.bin').write_bytes(b'exact')
     # Model the RTX5090 observation: 12 bounded screening continuations,
     # then a completed winner; the following AUTO must be a pure cache hit.
     if serial<=12:source='deferred'
     elif serial==13:source='selected'
     else:source='cache-hit'
     return dict(id=serial,p=p,path=str(path),shape='1:512:8:512:202',
       profile='INPLACE=1,LOADS=10040,MODM31=2,STORES=22' if serial>=13 else '',
       source=source,searches=0 if source=='cache-hit' else 1,
       implementation_searches=1,shape_cache_hit=True,wall_s=.01,seconds=.01)
    old=m.engine;m.engine=fake
    first=fake(147800003,runtime='auto')
    choice,hit=m.complete_deferred_auto(147800003,first,max_resumes=20)
    self.assertEqual(choice['source'],'selected')
    self.assertEqual(hit['source'],'cache-hit')
    self.assertTrue(hit['shape_cache_hit'])
    self.assertEqual(serial,14)
    self.assertEqual(choice['profile'],'INPLACE=1,LOADS=10040,MODM31=2,STORES=22')
   finally:
    m.engine=old;m.OUT=oldout
 def test_explicit_persistent_cache_root(self):
  with tempfile.TemporaryDirectory() as d:
   oldout=m.OUT;m.OUT=Path(d)/'campaign';m.OUT.mkdir()
   cache_root=Path(d)/'persistent'
   try:
    with mock.patch.dict(os.environ,{'AEVUM_PASS5_CACHE_ROOT':str(cache_root)}):
     env=m.environment()
     self.assertEqual(Path(env['AEVUM_AUTOTUNE_CACHE']),cache_root.resolve()/'cache-shape.tsv')
     self.assertTrue(cache_root.exists())
   finally:m.OUT=oldout
if __name__=='__main__':unittest.main()
