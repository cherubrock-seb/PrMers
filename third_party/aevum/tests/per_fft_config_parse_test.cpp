#include "Args.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

int main() {
  Args args(true);
  const std::vector<KeyVal> none;

  // In current Aevum an unprefixed NTT shape defaults to FFT3161, whose
  // explicit numeric type prefix is 1.  Type 0 is FFT64 and is not equivalent.
  const std::string canonical = FFTConfig{"256:2:256"}.shape.spec();
  const std::string canonical_prefixed = FFTConfig{"1:256:2:256"}.shape.spec();
  const std::string canonical_variant = FFTConfig{"256:2:256:202"}.shape.spec();

  require(canonical == canonical_prefixed,
          "explicit FFT3161 type-one spelling did not canonicalize to unprefixed shape");
  require(canonical == canonical_variant,
          "variant spelling did not canonicalize to same shape");

  args.parse("! 1:256:2:256 NO_ASM=1 UNROLL_W=0 # first");
  require(args.valueFor("NO_ASM", -1, canonical, none) == 1,
          "first whitespace-separated flag was lost");
  require(args.valueFor("UNROLL_W", -1, canonical, none) == 0,
          "second whitespace-separated flag was lost");

  args.parse("! 256:2:256:202 NO_ASM=0, TAIL_TRIGS61=2 # variant spelling");
  require(args.valueFor("NO_ASM", -1, canonical, none) == 0,
          "later equivalent FFT spelling did not replace matching key");
  require(args.valueFor("UNROLL_W", -1, canonical, none) == 0,
          "repeated line dropped unrelated prior key");
  require(args.valueFor("TAIL_TRIGS61", -1, canonical, none) == 2,
          "mixed comma and whitespace flag syntax was not retained");

  args.parse("! 256:2:256 TAIL_TRIGS31=1,TAIL_TRIGS32=2");
  require(args.valueFor("TAIL_TRIGS31", -1, canonical, none) == 1,
          "comma-separated first flag was not retained");
  require(args.valueFor("TAIL_TRIGS32", -1, canonical, none) == 2,
          "comma-separated second flag was not retained");
  require(args.perFftConfig.size() == 1,
          "equivalent FFT spellings created multiple configuration entries");

  const auto size_before_invalid = args.perFftConfig.size();
  args.parse("! foo TAIL_TRIGS61=3");
  require(args.perFftConfig.size() == size_before_invalid,
          "invalid FFT spelling created configuration state");

  args.parse("! 256:2:256 NO_ASM=1");
  require(args.valueFor("NO_ASM", -1, canonical, none) == 1,
          "later value did not replace matching key");
  require(args.valueFor("UNROLL_W", -1, canonical, none) == 0,
          "later replacement dropped unrelated key");
  require(args.valueFor("TAIL_TRIGS61", -1, canonical, none) == 2,
          "later replacement dropped another unrelated key");

  std::cout << "ok\n";
  return 0;
}
