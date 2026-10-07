// Host test: carryFused's WMUL must fit the device's maximum workgroup size and local memory.
//
// carryFused runs G_W * WMUL threads per workgroup.  The default WMUL=2 at a 1K width with NW=4 asks for 2 * 256 = 512 threads,
// more than an Intel UHD's maximum workgroup size of 256, so the first launch failed with INVALID_WORK_GROUP_SIZE.

#include "WmulLimit.h"

#include <cstdint>
#include <cstdio>

using namespace wmul_limit;

static int failures = 0;

static void expect(const char* what, uint32_t got, uint32_t want) {
  if (got != want) {
    std::printf("FAIL %s: got %u, want %u\n", what, got, want);
    ++failures;
  }
}

// Effective WMUL for a plan on a device, as clDefines computes it.
static uint32_t effective(uint32_t requested, uint32_t width, uint32_t nW, uint32_t bigHeight, uint32_t shufl, uint64_t localMem, uint32_t maxWg) {
  return clampWmul(requested, maxWmul(width, nW, shufl, localMem, maxWg), bigHeight);
}

int main() {
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
