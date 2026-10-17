#include "StrictInt.h"
#include <climits>
#include <iostream>
#include <stdexcept>
#include <string>

static void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

// -use values such as TAIL_KERNELS are range checked on the host and forwarded verbatim to the kernels, so the
// host must refuse everything that is not a plain integer.  atoi() turned "garbage" into 0 and "3x" into 3.
int main() {
  struct Ok { const char* text; int value; };
  for (const Ok& c : {Ok{"0", 0}, Ok{"1", 1}, Ok{"3", 3}, Ok{"+2", 2}, Ok{"-1", -1}, Ok{"07", 7}, Ok{"2147483647", INT_MAX}, Ok{"-2147483648", INT_MIN}}) {
    int v = -12345;
    require(parseStrictInt(c.text, v), std::string("rejected valid integer '") + c.text + "'");
    require(v == c.value, std::string("wrong value for '") + c.text + "'");
  }

  // Nonnumeric text, trailing/leading junk, other bases, empty, and values that do not fit an int.
  for (const char* bad : {"", " ", "garbage", "TAIL", "3x", "x3", "3 ", " 3", "\t3", "3\n", "1.0", "1e1", "0x3", "0b1", "--1", "+-1", "+", "-",
                          "1,2", "2147483648", "-2147483649", "99999999999999999999", "٣"}) {
    int v = 42;
    require(!parseStrictInt(bad, v), std::string("accepted '") + bad + "'");
    require(v == 42, std::string("modified the output for rejected '") + bad + "'");
  }
  std::cout << "ok\n";
  return 0;
}
