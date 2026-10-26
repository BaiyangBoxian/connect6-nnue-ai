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
// tss.inc -- reconstructed (the original file was missing from the archive).
//
// TSS (threat-space search) helpers, reconstructed from their call sites in
// engine_ab.cpp:
//   TSS_AT(b, r, c)                     L237+ : board access in 0..18 coords
//                                        (CharBoard has a 5-cell sentinel border,
//                                        data lives at [r+5][c+5])
//   tss_win_now(b, code, ocode, out)    L446  : can `code` complete six-in-a-row
//                                        with at most 2 stones (= win this move)?
//   tss_pick2nd(b, r1,c1, r0,c0, &r2,&c2) L414 : choose the second stone of a move
//                                        (only used by the disabled forced_move path)
//   is_gameover_char(b, r,c, r2,c2, code) L544 : does `code` have six-in-a-row
//                                        through (r,c) / (r2,c2)?
// All functions take CharBoard coordinates (already +5 offset) except TSS_AT,
// which mirrors the engine's 0..18 convention.
#pragma once

#define TSS_AT(b, r, c) ((b)[(r) + 5][(c) + 5])

// does `code` make 6 consecutive through CharBoard point (r5,c5) ?
static int tss_six_through(unsigned char b[29][29], int r5, int c5, unsigned char code) {
    static const int DR[4] = { 0, 1, 1, 1 }, DC[4] = { 1, 0, 1, -1 };
    for (int d = 0; d < 4; ++d) {
        int n = 1;
        for (int i = 1; i < 6; ++i) {
            int rr = r5 + i * DR[d], cc = c5 + i * DC[d];
            if (rr < 5 || rr > 23 || cc < 5 || cc > 23 || b[rr][cc] != code) break;
            ++n;
        }
        for (int i = 1; i < 6; ++i) {
            int rr = r5 - i * DR[d], cc = c5 - i * DC[d];
            if (rr < 5 || rr > 23 || cc < 5 || cc > 23 || b[rr][cc] != code) break;
            ++n;
        }
        if (n >= 6) return 1;
    }
    return 0;
}

// game over check for the last move(s): six-in-a-row through (r5,c5) or (r5b,c5b)
static int is_gameover_char(unsigned char b[29][29], char r5, char c5,
                            char r5b, char c5b, unsigned char code) {
    if (tss_six_through(b, (int)r5, (int)c5, code)) return 1;
    if ((r5 != r5b || c5 != c5b) && tss_six_through(b, (int)r5b, (int)c5b, code)) return 1;
    return 0;
}

// can `code` complete six-in-a-row with <= 2 stones (i.e. win on this move)?
// Fills out[0..3] with up to 2 empty cells (r1,c1,r2,c2); -1 when unused.
static int tss_win_now(unsigned char b[29][29], unsigned char code, unsigned char ocode, int* out) {
    static const int DR[4] = { 0, 1, 1, 1 }, DC[4] = { 1, 0, 1, -1 };
    int found = 0, bestEc = 9;
    for (int d = 0; d < 4; ++d) {
        for (int r = 0; r < 19; ++r) {
            for (int c = 0; c < 19; ++c) {
                int lr = r + 5 * DR[d], lc = c + 5 * DC[d];
                if (lr < 0 || lr > 18 || lc < 0 || lc > 18) continue;   // window leaves board
                int ec = 0, blocked = 0, e[6][2];
                for (int i = 0; i < 6; ++i) {
                    unsigned char v = TSS_AT(b, r + i * DR[d], c + i * DC[d]);
                    if (v == code) continue;
                    if (v == ocode) { blocked = 1; break; }
                    e[ec][0] = r + i * DR[d];
                    e[ec][1] = c + i * DC[d];
                    ++ec;
                }
                if (blocked || ec > 2) continue;
                // window has (6-ec) code stones, ec <= 2 empties, no opponent:
                // filling the empties completes six consecutive for `code`
                if (!found || ec < bestEc) {
                    found = 1; bestEc = ec;
                    out[0] = ec > 0 ? e[0][0] : -1;
                    out[1] = ec > 0 ? e[0][1] : -1;
                    out[2] = ec > 1 ? e[1][0] : -1;
                    out[3] = ec > 1 ? e[1][1] : -1;
                }
            }
        }
    }
    return found;
}

// pick the second stone of a move whose first stone is (r1,c1); (r0,c0) is the
// anchor point of the threat being answered.  Only reachable from forced_move(),
// which the search never calls -- kept for compilation completeness.
static void tss_pick2nd(unsigned char b[29][29], int r1, int c1, int r0, int c0,
                        int* r2, int* c2) {
    static const int DR[4] = { 0, 1, 1, 1 }, DC[4] = { 1, 0, 1, -1 };
    for (int d = 0; d < 4; ++d) {
        for (int s = -5; s <= 5; ++s) {
            if (s == 0) continue;
            int rr = r1 + s * DR[d], cc = c1 + s * DC[d];
            if (rr < 0 || rr > 18 || cc < 0 || cc > 18) continue;
            if (TSS_AT(b, rr, cc) == 0) { *r2 = rr; *c2 = cc; return; }
        }
    }
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            int rr = r0 + dr, cc = c0 + dc;
            if (rr < 0 || rr > 18 || cc < 0 || cc > 18) continue;
            if (TSS_AT(b, rr, cc) == 0) { *r2 = rr; *c2 = cc; return; }
        }
    }
    *r2 = 9; *c2 = 9;
}
