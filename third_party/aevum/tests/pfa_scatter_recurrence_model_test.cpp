// Host-only scalar model for NVIDIA PFA GF31/GF61 fftW scatter recurrence.
// Check the algebraic transformation, complete permutation and source guards.
// This does not substitute for native GPU full-output correctness tests.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

struct Shape {
  u32 width;
  u32 height;
  u32 radix;
  u32 nw;
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void checkSource() {
  const auto load = [](const char* filename) {
    std::ifstream file(filename);
    require(static_cast<bool>(file), std::string("cannot read ") + filename);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>{});
  };

  const std::string fftw = load("src/cl/fftw.cl");
  const std::string gpu = load("src/Gpu.cpp");
  const auto has = [](const std::string& text, const char* needle) {
    return text.find(needle) != std::string::npos;
  };
  const std::string step =
      "const u32 outStep = PFA_LOGICAL_STEP / (2u * BIG_HEIGHT);";

  require(has(fftw, "#if NVIDIAGPU\n"),
          "NVIDIA GF31 recurrence guard missing");
  require(has(fftw, "#if NVIDIAGPU && (PFA_RADIX == 7 || PFA_RADIX == 9)"),
          "GF61 recurrence radix guard missing");
  require(has(fftw, "static inline u32 pfaWLogicalIndex("),
          "canonical Good-Thomas map missing");
  require(has(fftw, "static inline u32 pfaWCanonicalPairIndex("),
          "canonical pair map missing");
  require(has(fftw, "const u32 baseEven = (pairEven - xEven * BIG_HEIGHT) * WIDTH;"),
          "recurrence base formula missing");
  require(has(fftw, "const u32 baseOdd  = (pairOdd  - xOdd  * BIG_HEIGHT) * WIDTH;"),
          "recurrence odd-base formula missing");
  require(has(fftw, "outEven += outStep; if (outEven >= endEven) outEven -= WIDTH;"),
          "even recurrence step/wrap missing");
  require(has(fftw, "outOdd  += outStep; if (outOdd  >= endOdd)  outOdd  -= WIDTH;"),
          "odd recurrence step/wrap missing");
  require(has(fftw, "outScalar31[outEven * 2u + 1u] = u[i].y;"),
          "GF31 even destination missing");
  require(has(fftw, "outScalar61[outOdd  * 2u]      = u[i].x;"),
          "GF61 odd destination missing");

  const auto first = fftw.find(step);
  require(first != std::string::npos && fftw.find(step, first + 1) != std::string::npos,
          "both GF31/GF61 recurrence steps must exist");
  require(fftw.find(step, fftw.find(step, first + 1) + 1) == std::string::npos,
          "unexpected additional recurrence step");

  require(has(gpu, "const u32 binary_length = 2u * fft.shape.width * fft.shape.height;"),
          "host binary length formula changed");
  require(has(gpu, "const u32 binary_step = binary_length / fft.shape.nW();"),
          "host binary step formula changed");
  require(has(gpu, "if (candidate % fft.pfa_radix == 0)"),
          "host logical-step modular gate changed");
  std::cout << "PFA_SCATTER_PRODUCTION_SOURCE_CONTRACT=PASS\n";
}

void checkShape(const Shape& s) {
  const u32 binaryLength = 2u * s.width * s.height;
  const u32 bigHeight = s.height * s.radix;
  const u32 nwords = 2u * s.width * bigHeight;

  require(s.width % s.nw == 0, "width/nw mismatch");

  u32 linv = 0;
  for (u32 x = 1; x < s.radix; ++x) {
    if ((binaryLength % s.radix) * x % s.radix == 1u) {
      linv = x;
      break;
    }
  }
  require(linv != 0, "PFA modular inverse missing");

  const u32 binaryStep = binaryLength / s.nw;
  u32 logicalStep = 0;

  for (u32 k = 0; k < s.radix; ++k) {
    const u32 candidate = binaryStep + binaryLength * k;
    if (candidate % s.radix == 0) {
      logicalStep = candidate;
      break;
    }
  }

  require(logicalStep != 0 && logicalStep < nwords,
          "PFA logical step invalid");
  require(logicalStep % (2u * bigHeight) == 0,
          "PFA scatter step not line-preserving");

  const u32 outStep = logicalStep / (2u * bigHeight);
  require(outStep > 0 && outStep < s.width,
          "PFA scatter step exceeds width");

  const auto logicalIndex = [&](u32 row, u32 binaryIndex) {
    const u32 delta =
        (row + s.radix - binaryIndex % s.radix) % s.radix;
    const u32 t = (delta * linv) % s.radix;
    return binaryIndex + binaryLength * t;
  };

  const auto canonicalPair = [&](u32 logical) {
    const u32 pair = logical >> 1;
    const u32 x = pair / bigHeight;
    return (pair - x * bigHeight) * s.width + x;
  };

  std::vector<std::uint8_t> hits(nwords, 0);
  u64 comparisons = 0;

  for (u32 row = 0; row < s.radix; ++row) {
    for (u32 y = 0; y < s.height; ++y) {
      for (u32 me = 0; me < s.width / s.nw; ++me) {
        const u32 firstBinaryPair = me * s.height + y;

        u32 nEven = logicalIndex(row, firstBinaryPair * 2u);
        u32 nOdd = logicalIndex(row, firstBinaryPair * 2u + 1u);

        const u32 pairEven = nEven >> 1;
        const u32 pairOdd = nOdd >> 1;

        const u32 xEven = pairEven / bigHeight;
        const u32 xOdd = pairOdd / bigHeight;

        const u32 baseEven =
            (pairEven - xEven * bigHeight) * s.width;
        const u32 baseOdd =
            (pairOdd - xOdd * bigHeight) * s.width;

        u32 outEven = baseEven + xEven;
        u32 outOdd = baseOdd + xOdd;

        const u32 endEven = baseEven + s.width;
        const u32 endOdd = baseOdd + s.width;

        for (u32 i = 0; i < s.nw; ++i) {
          require(outEven == canonicalPair(nEven),
                  "even scalar recurrence differs from canonical");
          require(outOdd == canonicalPair(nOdd),
                  "odd scalar recurrence differs from canonical");

          const u32 evenScalar = outEven * 2u + 1u;
          const u32 oddScalar = outOdd * 2u;

          require(evenScalar < nwords && oddScalar < nwords,
                  "scatter output index out of range");
          require(hits[evenScalar] == 0,
                  "duplicate even scalar destination");
          require(hits[oddScalar] == 0,
                  "duplicate odd scalar destination");

          hits[evenScalar] = 1;
          hits[oddScalar] = 1;
          ++comparisons;

          outEven += outStep;
          if (outEven >= endEven) outEven -= s.width;

          outOdd += outStep;
          if (outOdd >= endOdd) outOdd -= s.width;

          nEven += logicalStep;
          if (nEven >= nwords) nEven -= nwords;

          nOdd += logicalStep;
          if (nOdd >= nwords) nOdd -= nwords;
        }
      }
    }
  }

  const u64 covered =
      std::accumulate(hits.begin(), hits.end(), u64{0});
  require(covered == nwords,
          "scatter permutation fails full destination coverage");
  require(comparisons * 2u == nwords,
          "scatter source/destination cardinality mismatch");

  std::cout
      << "SCATTER_MODEL_CASE=PASS"
      << " width=" << s.width
      << " height=" << s.height
      << " radix=" << s.radix
      << " nw=" << s.nw
      << " logical_step=" << logicalStep
      << " output_step=" << outStep
      << " scalar_destinations=" << covered
      << '\n';
}

int main() {
  checkSource();

  const std::vector<Shape> cases = {
      {512, 512, 3, 8},
      {512, 512, 7, 8},
      {512, 512, 9, 8},
      {512, 1024, 3, 8},
      {512, 1024, 7, 8},
      {512, 1024, 9, 8},
      {1024, 512, 7, 8},
      {1024, 512, 9, 8},
  };

  for (const Shape& s : cases) checkShape(s);

  std::cout << "PFA_SCATTER_HOST_MODEL=PASS\n";
  std::cout << "EXECUTED_GEOMETRIES=" << cases.size() << '\n';
  std::cout << "CANONICAL_RECURRENCE_EQUIVALENCE=PASS\n";
  std::cout << "FULL_SCALAR_DESTINATION_PERMUTATION=PASS\n";
  std::cout << "GF31_RADICES=3,7,9\n";
  std::cout << "GF61_OPTIMIZED_RADICES=7,9\n";
  std::cout << "DEVICE_EXECUTION=NONE\n";
}
