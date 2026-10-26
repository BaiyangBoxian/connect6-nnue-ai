// Copyright (c) 2026 白杨博贤
// nnue_int8.h -- reconstructed (the original file was missing from the archive).
//
// Quantized (int8) NNUE weights used ONLY by nnue_forward_int8() (engine_ab.cpp L183).
// That function has no call sites in the translation unit (the live eval path is the
// float nnue_forward() using NN_W1..NN_B3 from nnue_A.h / nnue_B.h), so zero-filled
// placeholders cannot influence engine behaviour.
#pragma once

static const signed char Q_W1[NN_H1 * NN_D] = {};
static const signed char Q_W2[NN_H2 * NN_H1] = {};
static const signed char Q_W3[NN_H2] = {};
static const float Q_S2 = 1.0f;
static const float Q_S3 = 1.0f;
