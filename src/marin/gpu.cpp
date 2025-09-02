/*
Copyright 2025, Yves Gallot

marin is free source code. You can redistribute, use and/or modify it.
Please give feedback to the authors if improvement is realized. It is distributed in the hope that it will be useful.
*/

#include <cstdint>

#include "marin/engine_gpu.h"

engine * engine::create_gpu(const uint32_t p, const size_t reg_count, const size_t device, const bool verbose) { return new engine_gpu(p, reg_count, device, verbose); }