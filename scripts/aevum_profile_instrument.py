#!/usr/bin/env python3
"""Apply the SAME additive, disabled-by-default profiling hook to either build."""
from pathlib import Path
import sys
r = Path(sys.argv[1]) / 'third_party/aevum/src'
p = r / 'EngineApi.h'
s = p.read_text()
if 'aevum_engine_profile_report' not in s:
    p.write_text(s.replace('AEVUM_ENGINE_API int aevum_engine_sync(aevum_engine_handle handle);', 'AEVUM_ENGINE_API int aevum_engine_sync(aevum_engine_handle handle);\n// Diagnostic: synchronize, optionally emit kernel event counters, then reset them.\nAEVUM_ENGINE_API int aevum_engine_profile_report(aevum_engine_handle handle, int emit);'))
p = r / 'EngineApi.cpp'
s = p.read_text()
if 'AEVUM_PROFILE_KERNELS' not in s:
    s = s.replace('    args_.setDefaults();', '    args_.setDefaults();\n    args_.profile = std::getenv("AEVUM_PROFILE_KERNELS") && std::strcmp(std::getenv("AEVUM_PROFILE_KERNELS"), "1") == 0;')
    s = s.replace('  void set_u32(size_t dst, uint32_t value) {', '  void profile_report(bool emit) {\n    sync();\n    gpu_->regProfileReport(emit);\n  }\n\n  void set_u32(size_t dst, uint32_t value) {')
    s += '\nextern "C" AEVUM_ENGINE_API int aevum_engine_profile_report(aevum_engine_handle handle, int emit) {\n  return invoke([&] { runtime(handle).profile_report(emit != 0); });\n}\n'
    p.write_text(s)
p = r / 'Gpu.h'; s = p.read_text()
if 'regProfileReport' not in s:
    p.write_text(s.replace('  void regSync();', '  void regSync();\n  void regProfileReport(bool emit);'))
p = r / 'Queue.h'; s = p.read_text()
if 'collectProfileEvents' not in s:
    p.write_text(s.replace('  void finish();', '  void finish();\n  void collectProfileEvents() { events.synced(); }'))
p = r / 'Gpu.cpp'; s = p.read_text()
if 'void Gpu::regProfileReport' not in s:
    s += '''
void Gpu::regProfileReport(bool emit) {
  // Profile-only drain: auxiliary events otherwise remain uncollected until destruction.
  if (args.profile) for (auto& q : auxQueues) {
    ::finish(q.get());
    q.collectProfileEvents();
  }
  if (emit) for (const TimeInfo* p : profile.get()) {
    log("AEVUM_PROFILE name=%s calls=%u exec_ns=%lld\\n", p->name.c_str(), p->n,
        static_cast<long long>(p->times[2]));
  }
  profile.reset();
}
'''
    p.write_text(s)
