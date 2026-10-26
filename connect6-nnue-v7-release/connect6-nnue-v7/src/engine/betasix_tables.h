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
// betasix_tables.h -- reconstructed (the original file was missing from the archive).
//
// Symbols required by engine_ab.cpp:
//   ColorHash[2]   : CharBoard stone codes.  Call sites: SELF = ColorHash[1] = 7,
//                    OPP  = ColorHash[0] = 1  (engine_ab.cpp L94-95, L498).
//   ScoreHash1/2   : assigned to the global `ScoreHash` (L505, L520) but NEVER read
//                    anywhere in the translation unit -- vestigial legacy tables from
//                    the pre-NNUE hand-crafted eval.  Zero-filled placeholders.
#pragma once

static const unsigned char ColorHash[2] = { 1, 7 };   // [0]=opp=1, [1]=self=7

static const int ScoreHash1[1][2][2][256] = {};
static const int ScoreHash2[1][2][2][256] = {};
