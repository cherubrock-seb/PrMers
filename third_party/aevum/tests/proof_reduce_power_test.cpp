// Host-only test: ProofSet::reducePower() must leave the set exactly as a ProofSet built at the lower power.
// The checkpoint table is private, so it is compared through the public next() walk.

#include "Proof.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

static int failures = 0;

static void check(bool ok, const char* what, u64 E, u32 power) {
  if (!ok) {
    std::fprintf(stderr, "FAIL E=%llu power=%u: %s\n", (unsigned long long) E, power, what);
    ++failures;
  }
}

// Every checkpoint in (0, E], in order, as reported by next().
static std::vector<u64> walk(const ProofSet& ps, u64 E) {
  std::vector<u64> out;
  for (u64 k = 0; k < E;) {
    k = ps.next(k);
    out.push_back(k);
    if (out.size() > (1u << 13)) { break; } // a broken table must not hang the test
  }
  return out;
}

int main() {
  char dirTemplate[] = "proof_reduce_power_XXXXXX";
  if (!mkdtemp(dirTemplate)) { std::perror("mkdtemp"); return 2; }
  std::string dir = dirTemplate;
  if (chdir(dir.c_str())) { std::perror("chdir"); return 2; }

  for (u64 E : {u64(9689), u64(216091), u64(110503)}) {
    for (u32 start : {2u, 5u, 9u}) {
      ProofSet ps{E, start};
      // Leave the iterator cache pointing near the end, as a long run would.
      ps.next(E - 1);
      for (u32 power = start; power > 1;) {
        ps.reducePower();
        --power;
        check(ps.power == power, "power not decremented", E, power);
        ProofSet fresh{E, power};
        std::vector<u64> got = walk(ps, E), want = walk(fresh, E);
        check(want.size() == (1u << power), "fresh table has the wrong size", E, power);
        check(got == want, "reduced table differs from a fresh one", E, power);
        check(ps.next(E) == fresh.next(E), "guard element differs", E, power);
        for (u64 k : {u64(0), E / 3, E / 2, E - 1}) {
          check(ps.next(k) == fresh.next(k), "next(k) differs", E, power);
        }
      }
    }
  }

  if (chdir("..")) { std::perror("chdir"); }
  std::string cmd = "rm -rf '" + dir + "'";
  if (std::system(cmd.c_str())) { std::perror("cleanup"); }

  if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
  std::puts("proof_reduce_power_test: ok");
  return 0;
}
