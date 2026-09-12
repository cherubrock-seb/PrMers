"""Host-only control-flow tests. Synthetic timings are NOT hardware measurements."""
import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
MEM='INPLACE=1,LOADS=10040,STORES=22'
SHAPE='1:256:4:256:101'
class HwFix(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='rtx-hwfix-test-');self.addCleanup(self.temp.cleanup)
  with patch.object(sys,'argv',['test']):
   spec=importlib.util.spec_from_file_location('campaign_hwfix',ROOT/'scripts/pass5_validate.py')
   self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  self.m.ROOT=ROOT;self.m.OUT=Path(self.temp.name);self.calls=[];self.serial=0
 def row(self,p=21000029,profile='',source='cache-hit',seconds=1.,**extra):
  self.serial+=1;d=self.m.OUT/str(self.serial);d.mkdir();(d/'residue.bin').write_bytes(b'exact words')
  return dict(id=self.serial,p=p,path=str(d),shape=SHAPE,profile=profile,source=source,
      searches=0,wall_s=1.,seconds=seconds,kernels={},cached_gain=1.1515,**extra)
 def cached(self,p=21000029):
  d=Path(str(self.m.OUT/'cache-shape.tsv')+'.prp-use-v4');d.mkdir(exist_ok=True)
  f=d/f'{p}.tsv';f.write_text(f'prp-use-v4|p={p}|use-flags=');return f
 def test_deferred_then_auto_retries(self):
  rows=[self.row(source='deferred',use_budget_ms=0,use_cache_records=[]),
        self.row(source='measured-defaults',implementation_searches=1,shape_cache_hit=True)]
  with patch.object(self.m,'engine',side_effect=rows) as run:
   a,b=self.m.cold_auto(21000029)
   self.assertEqual(b['source'],'measured-defaults');self.assertEqual(run.call_count,2)
 def test_zero_budget_negative_is_failure(self):
  rows=[self.row(source='measured-defaults',use_budget_ms=0),self.row()]
  with patch.object(self.m,'engine',side_effect=rows):
   with self.assertRaisesRegex(RuntimeError,'zero budget'):self.m.cold_auto(21000029)
 def test_completed_negative_cache_hit(self):
  rows=[self.row(source='measured-defaults',use_budget_ms=8000),self.row()]
  with patch.object(self.m,'engine',side_effect=rows):self.m.cold_auto(21000029)
 def test_deferred_must_not_write_defaults(self):
  rows=[self.row(source='deferred',use_cache_records=[{'profile':'defaults'}]),
        self.row(implementation_searches=1,shape_cache_hit=True)]
  with patch.object(self.m,'engine',side_effect=rows):
   with self.assertRaisesRegex(RuntimeError,'final use-cache'):self.m.cold_auto(21000029)
 def test_prior_positive_survives_incomplete_forced_retune(self):
  rows=[self.row(source='deferred',use_cache_records=[{'profile':MEM}]),self.row(profile=MEM)]
  with patch.object(self.m,'engine',side_effect=rows):
   self.assertEqual(self.m.cold_auto(21000029,SHAPE,True)[1]['profile'],MEM)
 def fake_pair_engine(self,p,plan='',profile=None,variant='new',**kw):
  self.calls.append((variant,profile))
  return self.row(p,profile or '',seconds=.9 if profile else 1.)
 def test_independent_result_separate_from_internal_and_alternates(self):
  self.m.engine=self.fake_pair_engine;summary={};choice={'profile':MEM,'cached_gain':1.1515}
  self.assertIsNotNone(self.m.validate_final(summary,21000029,SHAPE,choice,'warm-final'))
  r=summary['final_implementation']['21000029']
  self.assertEqual(r['tuner_internal_gain'],1.1515)
  self.assertAlmostEqual(r['independent_engine_gain'],1/.9)
  self.assertTrue(r['independently_accepted'])
  self.assertEqual([v for v,p in self.calls],['new','new','new','new','new','new'])
 def test_release_nonregression_is_separate_pass4_gate(self):
  self.m.engine=self.fake_pair_engine
  r=self.m.pair_pass4(21000029,SHAPE,MEM)
  self.assertTrue(r['word_exact'])
  self.assertEqual(len(self.calls),10)
  self.assertEqual(sum(v=='old' for v,p in self.calls),5);self.assertEqual(sum(v=='new' for v,p in self.calls),5)
 def test_failed_final_profile_evicts_only_its_cache(self):
  bad=self.cached();other=self.cached(196999969)
  def slow(p,plan='',profile=None,variant='new',**kw):return self.row(p,profile or '',seconds=1.1 if profile else 1.)
  self.m.engine=slow;s={}
  self.assertIsNone(self.m.validate_final(s,21000029,SHAPE,{'profile':MEM},'warm-final'))
  self.assertFalse(bad.exists());self.assertTrue(other.exists())
  self.assertEqual(s['final_implementation']['21000029']['status'],'rejected')

 def test_completed_defaults_not_rejected_by_timing_jitter(self):
  def jitter(p,plan='',profile=None,variant='new',**kw):
   # A defaults-vs-defaults diagnostic is not an implementation candidate.
   # Preserve exactness, but simulate enough noise to fail the optimized
   # no-regression ratio gate.
   return self.row(p,'',seconds=1.03 if variant=='new' else 1.0)
  self.m.engine=jitter;s={}
  r=self.m.validate_final(s,180000007,'4:512:8:512:202',{'profile':'','cached_gain':1.0},'defaults')
  self.assertIsNotNone(r)
  self.assertEqual(s['final_implementation']['180000007']['status'],'defaults')
  self.assertFalse(s.get('failures'))
 def test_word_mismatch_rejects_final_profile(self):
  cached=self.cached()
  def mismatch(p,plan='',profile=None,variant='new',**kw):
   row=self.row(p,profile or '')
   if profile:(Path(row['path'])/'residue.bin').write_bytes(b'wrong words')
   return row
  self.m.engine=mismatch;s={}
  self.assertIsNone(self.m.validate_final(s,21000029,SHAPE,{'profile':MEM},'warm-final'))
  self.assertFalse(cached.exists());self.assertIn('WORD MISMATCH',s['failures'][0]['error'])
 def test_full_campaign_validates_changed_warm_profile(self):
  self.m.EXPS=[21000029];profiles=[]
  def cold(p,shape='',warm=False):return self.row(p),self.row(p,MEM if warm else '')
  def pair(p,shape,profile,epoch=False):
   profiles.append(profile);gain=1.12 if profile==MEM else 1.08 if profile else 1.
   return dict(p=p,profile=profile,engine_gain=gain,word_exact=True,accepted=bool(profile),no_regression=True)
  self.m.cold_auto=cold;self.m.pair=pair;self.m.engine=self.fake_pair_engine
  with patch.object(sys,'argv',['test']),patch('aevum_bench_common.oracle'),patch.dict(os.environ,{'AEVUM_PASS5_SKIP_EPOCH':'1'}):self.m.run()
  self.assertEqual(profiles,['',self.m.REPORTER,MEM])
  s=json.loads((self.m.OUT/'summary.json').read_text());final=s['final_implementation']['21000029']
  self.assertEqual(final['profile'],MEM);self.assertEqual(final['independent_engine_gain'],1.12)
  self.assertEqual(final['stage'],'final-warm-profile-21000029')
 def test_focused_radeon_resumes_multiple_deferred_searches_then_cache_hits(self):
  autos=0
  def engine(p,plan='',profile=None,variant='new',runtime=None,**kw):
   nonlocal autos
   if runtime=='retune':r=self.row(p,source='deferred',use_budget_ms=0,use_cache_records=[])
   elif runtime=='auto':
    autos+=1
    if autos<=2:r=self.row(p,source='deferred',implementation_searches=1,shape_cache_hit=True,use_next_screen=autos)
    elif autos==3:r=self.row(p,source='measured-defaults',implementation_searches=1,shape_cache_hit=True,use_next_screen=7)
    else:r=self.row(p,source='cache-hit',implementation_searches=0,shape_cache_hit=True)
   else:r=self.row(p,profile or '')
   r['shape']='4:512:8:512:202';return r
  self.m.engine=engine
  with patch.object(sys,'argv',['test','focused',str(ROOT),str(self.m.OUT),'0','radeonVII']),patch('builtins.print'):self.m.focused()
  s=json.loads((self.m.OUT/'summary.json').read_text())
  self.assertFalse(s['failures']);self.assertTrue(s['runtime'][0]['deferred_observed'])
  self.assertEqual([x['auto'].get('use_next_screen') for x in s['runtime'][1:3]],[2,7])
  self.assertEqual(s['cache_hit_after_independent_ab']['source'],'cache-hit')
  self.assertEqual(s['final_implementation']['180000007']['status'],'defaults')
  self.assertEqual(autos,4)
 def test_focused_rtx_reports_memory_profile_external_gain(self):
  def engine(p,plan='',profile=None,variant='new',runtime=None,**kw):
   profile=MEM if runtime else profile or ''
   return self.row(p,profile,source='measured' if runtime=='retune' else 'cache-hit',seconds=.9 if profile else 1.)
  self.m.engine=engine
  with patch.object(sys,'argv',['test','focused',str(ROOT),str(self.m.OUT),'2','rtx3080']),patch('builtins.print'):self.m.focused()
  s=json.loads((self.m.OUT/'summary.json').read_text());r=s['reduced_memory_profile']
  self.assertFalse(s['failures']);self.assertTrue(r['independently_accepted'])
  self.assertAlmostEqual(r['independent_engine_gain'],1/.9)
  self.assertEqual(s['final_implementation']['21000029']['tuner_internal_gain'],1.1515)
  self.assertEqual(self.serial,12)
 def test_focused_runtime_mismatch_evicts(self):
  cached=self.cached()
  with patch.object(self.m,'cold_auto',side_effect=RuntimeError('WORD MISMATCH')):
   with patch.object(sys,'argv',['test','focused',str(ROOT),str(self.m.OUT),'2','rtx3080']):self.m.focused()
  self.assertFalse(cached.exists())
if __name__=='__main__':unittest.main()
