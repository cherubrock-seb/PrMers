import importlib.util,json,os,sys,tempfile,unittest
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
if __name__=='__main__':unittest.main()
