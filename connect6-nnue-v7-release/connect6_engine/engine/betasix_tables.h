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
