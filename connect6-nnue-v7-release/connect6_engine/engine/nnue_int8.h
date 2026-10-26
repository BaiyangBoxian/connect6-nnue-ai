// connect6-nnue-ai
// Neural network + alpha-beta search board game AI, trained with self-play
//
// BetaSix: 面向十九路六子棋的多任务 NNUE 高效评估与 α-β 搜索
// Copyright (C) 2026  白杨博贤
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
