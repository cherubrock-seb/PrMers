#include "TrigBufCache.h"
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}

template <class F> static bool throws(F&& f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}

// Every distinct (number type in use, TAIL_TRIGS) configuration must map to its own trig cache key; otherwise two
// Gpus sharing a TrigBufCache can be handed each other's twiddle tables.
int main() {
  // All supported configurations: each of the four number types is unused, or in use with TAIL_TRIGS 0..MAX_TAIL_TRIGS.
  // A type that is not in use has no table, so its setting does not take part in the key (it must not matter, and
  // an unsupported value there must not matter either).
  const int values = MAX_TAIL_TRIGS + 1;
  const u32 perType = 1 + values;
  std::map<u32, std::vector<u32>> byKey;
  u32 configs = 0;
  for (u32 c = 0; c < perType * perType * perType * perType; ++c) {
    u32 sel[4], rest = c;
    for (u32& s : sel) { s = rest % perType; rest /= perType; }
    for (u32 tk = 0; tk < 2; ++tk) {
      ++configs;
      u32 key = trigKeyPart(sel[0] > 0, sel[0] ? sel[0] - 1 : 0, sel[1] > 0, sel[1] ? sel[1] - 1 : 0,
                            sel[2] > 0, sel[2] ? sel[2] - 1 : 0, sel[3] > 0, sel[3] ? sel[3] - 1 : 0, tk);
      byKey[key].push_back(c);
      // The setting of an unused type must not change the key.
      require(key == trigKeyPart(sel[0] > 0, sel[0] ? sel[0] - 1 : 2, sel[1] > 0, sel[1] ? sel[1] - 1 : 1,
                                 sel[2] > 0, sel[2] ? sel[2] - 1 : 2, sel[3] > 0, sel[3] ? sel[3] - 1 : 1, tk),
              "the setting of an unused number type changed the key");
    }
  }
  std::cout << configs << " configurations, " << byKey.size() << " distinct keys\n";
  require(byKey.size() == configs, "trig cache key aliases distinct configurations");

  // The case the old additive key got wrong: FP64 in use with TAIL_TRIGS=1 vs FP64 unused with TAIL_TRIGS=2.
  require(trigKeyPart(true, 1, false, 0, false, 0, false, 0, false) != trigKeyPart(false, 2, false, 0, false, 0, false, 0, false),
          "in-use flag and TAIL_TRIGS are not separate");

  // A value outside the supported range must be refused, never truncated into the field: with a 3-bit field, 8 used to
  // give the same key as 0 (the generators and kernels treat those two very differently), and 3..7 are unsupported.
  for (u32 bad : {3u, 4u, 7u, 8u, 9u, 16u, 255u, 0x80000000u}) {
    require(throws([&] { trigKeyField(true, bad); }), "an out-of-range TAIL_TRIGS value must not produce a key");
    require(throws([&] { trigKeyPart(true, bad, false, 0, false, 0, false, 0, false); }), "FP64 out-of-range value was accepted");
    require(throws([&] { trigKeyPart(false, 0, true, bad, false, 0, false, 0, false); }), "GF31 out-of-range value was accepted");
    require(throws([&] { trigKeyPart(false, 0, false, 0, true, bad, false, 0, false); }), "FP32 out-of-range value was accepted");
    require(throws([&] { trigKeyPart(false, 0, false, 0, false, 0, true, bad, false); }), "GF61 out-of-range value was accepted");
  }
  for (u32 ok = 0; ok <= u32(MAX_TAIL_TRIGS); ++ok) {
    require(!throws([&] { trigKeyField(true, ok); }), "a supported TAIL_TRIGS value was refused");
  }
  require(trigKeyField(true, 0) != trigKeyField(false, 0), "TAIL_TRIGS=0 in use must differ from unused");

  // The settings as they arrive from -use / config.txt: strict integers in range only.
  for (const char* key : {"TAIL_TRIGS", "TAIL_TRIGS31", "TAIL_TRIGS32", "TAIL_TRIGS61"}) {
    for (int ok = 0; ok <= MAX_TAIL_TRIGS; ++ok) {
      require(parseTailTrigs(key, std::to_string(ok)) == u32(ok), "valid TAIL_TRIGS text was misparsed");
    }
    for (const char* bad : {"", " ", "garbage", "1x", "x1", "1 ", " 1", "1.0", "-1", "3", "8", "09", "99999999999999999999", "0x1"}) {
      require(throws([&] { parseTailTrigs(key, bad); }), (std::string("accepted TAIL_TRIGS text '") + bad + "'").c_str());
    }
  }
  require(throws([] { checkTailTrigs("TAIL_TRIGS", -1); }), "negative TAIL_TRIGS accepted");
  require(throws([] { checkTailTrigs("TAIL_TRIGS", 8); }), "TAIL_TRIGS=8 accepted");
  require(!throws([] { checkTailTrigs("TAIL_TRIGS", 2); }), "TAIL_TRIGS=2 refused");

  std::cout << "ok\n";
  return 0;
}
