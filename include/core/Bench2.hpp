#pragma once

#include "io/CliParser.hpp"

namespace core::bench2 {

// Production-faithful PRP selector benchmark.
//
// This is deliberately separate from the historical -bench path.
// It creates the same generic engine::Reg backend ordinary PRP uses:
//   auto Marin/Aevum policy + 8 registers + PRP workload identity.
//
// Timing is steady-state arithmetic only. Setup/JIT/backend probing and
// profiler/resource collection are intentionally outside the timed samples.
int run(const io::CliOptions& options);

} // namespace core::bench2
