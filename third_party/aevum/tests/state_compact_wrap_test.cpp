#include "state.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static std::vector<Word> modulus_digits(u32 N, u64 E) {
  std::vector<Word> digits(N);
  for (u32 i = 0; i < N; ++i) {
    const u32 bits = bitlen(N, E, i);
    digits[i] = static_cast<Word>((u64(1) << bits) - 1u);
  }
  return digits;
}

static void expect_word(const std::vector<u32>& actual, const std::vector<u32>& expected, const char* label) {
  if (actual != expected) throw std::runtime_error(std::string(label) + " failed");
}

int main() {
  constexpr u64 E = 127;
  constexpr u32 N = 8;
  const u32 lastBits = bitlen(N, E, N - 1);
  const Word topBase = static_cast<Word>(u64(1) << lastBits);

  auto plusOne = modulus_digits(N, E);
  plusOne.back() += topBase;
  expect_word(compactBits(plusOne, E), {1u, 0u, 0u, 0u}, "positive carry wrap");

  auto plusTwo = modulus_digits(N, E);
  plusTwo.back() += 2 * topBase;
  expect_word(compactBits(plusTwo, E), {2u, 0u, 0u, 0u}, "positive carry two");

  std::vector<Word> minusOne(N, 0);
  minusOne.back() -= topBase;
  expect_word(compactBits(minusOne, E), {0xfffffffeu, 0xffffffffu, 0xffffffffu, 0x7fffffffu}, "negative carry wrap");

  std::vector<Word> minusTwo(N, 0);
  minusTwo.back() -= 2 * topBase;
  expect_word(compactBits(minusTwo, E), {0xfffffffdu, 0xffffffffu, 0xffffffffu, 0x7fffffffu}, "negative carry two");

  std::cout << "Aevum compact Mersenne carry tests passed\n";
  return 0;
}
