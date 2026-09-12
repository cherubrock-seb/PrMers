from pathlib import Path
import re,unittest
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'third_party/aevum/src'
class Guards(unittest.TestCase):
 def test_shape_cache_unchanged(self):
  import hashlib,json
  for f,h in json.loads((ROOT/'scripts/pass5-preserved.json').read_text()).items():
   self.assertEqual(hashlib.sha256((ROOT/f).read_bytes()).hexdigest(),h,f)
 def test_separation_and_exact_gate(self):
  t=(S/'EngineApi.cpp').read_text();u=t[t.index('std::vector<KeyVal> selectPrpUse'):t.index('class Runtime {')]
  self.assertLess(u.index('std::getenv("AEVUM_PRP_USE")'),u.index('TuneEntry::readTuneFile'))
  self.assertLess(u.index('TuneEntry::readTuneFile'),u.index('if(auto r=load(cache_key))'))
  self.assertIn('gain>=1.03',u);self.assertIn('worst>=0.985',u)
  self.assertIn('if(reference!=value)throw std::runtime_error("WORD MISMATCH")',u)
  # Slow OpenCL compilation may exceed the start budget, but a candidate that
  # already started must finish and checkpoint instead of being paid twice.
  self.assertIn('state.next_screen=index+1',u);self.assertIn('storeProgress(cache_key,state)',u)
  self.assertIn('if(planned<full_planned)progress.interrupted=true;',u)
  self.assertIn('auto prp_cost=[&](const std::string& profile,int quick)',u)
  self.assertIn('prepared_ = gpu_->makeTransformBufVector',t)
  self.assertIn('gpu_->regSquareStep(regs_[0], pending_lead_, true, false)',t)
  self.assertIn('if(elapsed()>=static_cast<int64_t>(budget)){progress.interrupted=true;break;}',u)
  confirm=u[u.index('for(unsigned fi=state.next_finalist;'):]
  self.assertIn('state.top.size()<=4',u)
  self.assertIn('if(top.size()>4)top.resize(4)',u)
  self.assertIn('const int confirm_quick=exponent<=30000000u?8',u)
  self.assertIn('(void)prp_cost("",10);(void)prp_cost(screened.profile,10);',confirm)
  self.assertIn('prp_cost(screened.profile,confirm_quick)',confirm)
  self.assertIn('Gpu::timePRP itself for tune.cpp',u)
  self.assertIn('for(unsigned repeat=0;repeat<3;++repeat)',confirm)
  self.assertIn('AEVUM_USE_ENGINE_GATE',confirm)
  self.assertIn('probe.sequence(64)',confirm)
  self.assertIn('probe.sequence(256)',confirm)
  self.assertIn('engine_keep=engine_gain>=1.03',confirm)
  self.assertIn('state.next_finalist=fi+1',confirm)
  self.assertIn('persistDecision(cache_key,decision',u)
  self.assertNotIn('hasManualPlanOverrideEnvironment',u)
  self.assertIn('!explicit_fft_spec && !gb202_profile && !manual_plan_env',t)
  self.assertIn('args_.clean = true;',t)
 def test_rejected_epoch_experiment_removed(self):
  t=(S/'cl/carryfused.cl').read_text();g=(S/'Gpu.cpp').read_text();h=(S/'Gpu.h').read_text();e=(S/'EngineApi.cpp').read_text()
  # Carry-epoch never passed the hardware gate.  Keep the production carry kernel
  # byte-for-byte identical to the Pass-4 snapshot so an OFF experiment cannot
  # perturb Radeon code generation.
  before=(ROOT/'scripts/pass5-baseline/third_party/aevum/src/cl/carryfused.cl').read_text()
  self.assertEqual(before,t)
  for token in ['PRP_CARRY_EPOCH','carryEpochEnabled','carryEpochDirty','carryEpoch']:
   self.assertNotIn(token,t+g+h+e)
if __name__=='__main__':unittest.main()
