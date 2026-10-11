#pragma once

namespace io { struct CliOptions; }

namespace core::bench3 {

// Public Bench3: compare the production AUTO selection against a bounded set
// of legal explicit Aevum FFT/PFA alternatives using exact residues and
// production-representative PRP timing.
int run(const io::CliOptions& options);

} // namespace core::bench3
