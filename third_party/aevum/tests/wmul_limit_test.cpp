// Host test: carryFused's WMUL must fit the device's maximum workgroup size and local memory.
//
// carryFused runs G_W * WMUL threads per workgroup.  The default WMUL=2 at a 1K width with NW=4 asks for 2 * 256 = 512 threads,
// more than an Intel UHD's maximum workgroup size of 256, so the first launch failed with INVALID_WORK_GROUP_SIZE.

#include "WmulLimit.h"
#include "StrictInt.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using namespace wmul_limit;

static int failures = 0;

static void expect(const char* what, uint32_t got, uint32_t want) {
  if (got != want) {
    std::printf("FAIL %s: got %u, want %u\n", what, got, want);
    ++failures;
  }
}

static void expectBool(const char* what, bool got, bool want) {
  if (got != want) {
    std::printf("FAIL %s: got %d, want %d\n", what, int(got), int(want));
    ++failures;
  }
}

// Match Gpu.cpp's explicit -use WMUL policy without involving any OpenCL
// device: syntax must be a complete decimal integer and the requested value
// must be at least one.  clampWmul() remains defensive for internal callers.
static bool parseExplicitWmul(const std::string& text, uint32_t& value) {
  int parsed = 0;
  if (!parseStrictInt(text, parsed) || parsed < 1) return false;
  value = static_cast<uint32_t>(parsed);
  return true;
}

static void expectGpuParserWiring() {
  std::ifstream input("src/Gpu.cpp");
  if (!input) {
    std::printf("FAIL cannot read src/Gpu.cpp for WMUL parser source contract\n");
    ++failures;
    return;
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string source = buffer.str();

  const char* required[] = {
      "if (k == \"WMUL\") {",
      "parseStrictInt(v, parsedWmul)",
      "parsedWmul < 1",
      "Invalid WMUL='%s', must be an integer >= 1",
      "wmul = static_cast<u32>(parsedWmul);",
  };

  for (const char* token : required) {
    if (source.find(token) == std::string::npos) {
      std::printf("FAIL Gpu.cpp WMUL parser missing source contract token: %s\n", token);
      ++failures;
    }
  }

  if (source.find("if (k == \"WMUL\") wmul = atoi(v.c_str());") != std::string::npos) {
    std::printf("FAIL Gpu.cpp still uses atoi for WMUL\n");
    ++failures;
  }
}

// Effective WMUL for a plan on a device, as clDefines computes it.
static uint32_t effective(uint32_t requested, uint32_t width, uint32_t nW, uint32_t bigHeight, uint32_t shufl, uint64_t localMem, uint32_t maxWg) {
  return clampWmul(requested, maxWmul(width, nW, shufl, localMem, maxWg), bigHeight);
}

int main() {
  // Explicit -use WMUL is configuration input, not the automatic/default
  // policy.  Refuse values that atoi() used to silently reinterpret.
  {
    uint32_t parsed = 99;
    expectBool("explicit wmul 0 rejected", parseExplicitWmul("0", parsed), false);
    expectBool("explicit wmul negative rejected", parseExplicitWmul("-1", parsed), false);
    expectBool("explicit wmul garbage rejected", parseExplicitWmul("garbage", parsed), false);
    expectBool("explicit wmul partial rejected", parseExplicitWmul("3x", parsed), false);
    expectBool("explicit wmul empty rejected", parseExplicitWmul("", parsed), false);
    expectBool("explicit wmul 1 accepted", parseExplicitWmul("1", parsed), true);
    expect("explicit wmul 1 value", parsed, 1);
    expectBool("explicit wmul 4 accepted", parseExplicitWmul("4", parsed), true);
    expect("explicit wmul 4 value", parsed, 4);
  }

  expectGpuParserWiring();

  // The default is unchanged where it fits.  AMD: 1024 threads, 64KB local memory.  NVIDIA: 1024 threads, 48KB.
  expect("amd 512 wmul2",   effective(2, 512,  4, 4096, 8, 65536, 1024), 2);
  expect("amd 1K wmul2",    effective(2, 1024, 8, 2048, 8, 65536, 1024), 2);
  expect("nv 1K wmul2",     effective(2, 1024, 8, 2048, 8, 49152, 1024), 2);
  expect("nv 256 wmul4",    effective(4, 256,  4, 2048, 8, 49152, 1024), 4);
  expect("amd 4K wmul2",    effective(2, 4096, 8, 2048, 8, 65536, 1024), 1);   // 4K width is capped at 1
  expect("amd 1K wmul4",    effective(4, 1024, 8, 2048, 8, 65536, 1024), 2);   // 1K width is capped at 2

  // Intel UHD: maximum workgroup 256, 64KB local memory.  A 1K width with NW=8 has G_W=128, so WMUL=2 is 256 threads and fits;
  // NW=4 (the 1K default without radix-8) has G_W=256, so WMUL=2 would be 512 threads and must fall back to 1.
  expect("uhd 1K nW8 wmul2", effective(2, 1024, 8, 2048, 8, 65536, 256), 2);
  expect("uhd 1K nW4 wmul2", effective(2, 1024, 4, 2048, 8, 65536, 256), 1);
  expect("uhd 512 nW4 wmul2", effective(2, 512, 4, 2048, 8, 65536, 256), 2);
  expect("uhd 256 nW4 wmul4", effective(4, 256, 4, 2048, 8, 65536, 256), 4);
  expect("uhd 256 nW4 wmul8", effective(8, 256, 4, 2048, 8, 65536, 256), 4);    // 8 * 64 = 512 threads > 256

  // A smaller device limit than one line still yields WMUL=1, never 0.
  expect("tiny wg",         effective(2, 1024, 4, 2048, 8, 65536, 128), 1);
  expect("wmul 0 request",  effective(0, 512, 4, 2048, 8, 65536, 1024), 1);

  // The local memory limit: 16-byte shuffles at 1K width need 16KB per line, so only two lines fit in 32KB.
  expect("shufl16 1K",      effective(2, 1024, 8, 2048, 16, 65536, 1024), 2);
  expect("shufl16 4K",      effective(2, 4096, 8, 2048, 16, 65536, 1024), 1);   // used to compute a maximum of 0
  expect("small lds",       effective(2, 1024, 8, 2048, 8, 8192, 1024), 1);     // 8KB: one 8KB line

  // WMUL must divide BIG_HEIGHT.
  expect("odd height",      effective(2, 512, 4, 2049, 8, 65536, 1024), 1);
  expect("height 12 wmul4", effective(4, 256, 4, 12,   8, 65536, 1024), 4);
  expect("height 9 wmul4",  effective(4, 256, 4, 9,    8, 65536, 1024), 3);

  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("wmul_limit_test OK\n");
  return 0;
}
