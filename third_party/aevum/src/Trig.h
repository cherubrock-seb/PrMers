// Copyright Mihai Preda

#pragma once

#include "common.h"
#include <array>

struct TrigCoefs {
  u32 scale;
  std::array<double, 8> sinCoefs;
  std::array<double, 8> cosCoefs;
};

TrigCoefs trigCoefs(u32 N);

struct TrigCoefsFP32 {
  double scale;
  std::array<double, 8> sinCoefs;
  std::array<double, 8> cosCoefs;
};

TrigCoefsFP32 trigCoefsFP32(u32 N);
