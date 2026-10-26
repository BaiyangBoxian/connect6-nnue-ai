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
// engine_ab.cpp -- Connect6 engine, Version A: alpha-beta (negamax) + threat-space
// search + Zobrist transposition table + iterative deepening.  Shares the eval
// (eval.inc) and TSS (tss.inc) with the greedy version.  Same C ABI.
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <utility>
using namespace std;

#define _EXTEND_ 5
#define _PART_
#define BOOK_KEY ***  // board cells + side
#define BOOK_REC 366                // key + move
// eval.inc / tables rely on these globals (declared below).
#include "betasix_tables.h"
#include "nnue_A.h"   // NEW net (Run 8)  -> NN_*_A
#include "nnue_B.h"   // OLD net (Run 1)  -> NN_*_B
static int g_net = 0;   // 0 = new (A), 1 = old (B)
static const float* NW1[2] = {NN_W1_A, NN_W1_B};
static const float* NB1[2] = {NN_B1_A, NN_B1_B};
static const float* NW2[2] = {NN_W2_A, NN_W2_B};
static const float* NB2[2] = {NN_B2_A, NN_B2_B};
static const float* NW3[2] = {NN_W3_A, NN_W3_B};
static const float* NB3[2] = {NN_B3_A, NN_B3_B};
#define NN_D 736   // 722 board + 14 features
#define NN_H1 256   // narrow (v7)
#define NN_H2 64
#define NN_VSCALE 200000.0f
#define NN_W1 NW1[g_net]
#define NN_B1 NB1[g_net]
#define NN_W2 NW2[g_net]
#define NN_B2 NB2[g_net]
#define NN_W3 NW3[g_net]
#define NN_B3 NB3[g_net]
#include "nnue_int8.h"
#define NNUE_ON 1   // multi-task NNUE (value head) -- RESTORED 2026-10-16 06:26 (was left at 0 after the E1/zero-sum experiment on 10-15 18:43; that edit's feat12-convention fix + recompute bugfix are kept)

unsigned short int is_white;
std::pair<std::pair<short int, short int>, std::pair<short int, short int> > other;
short int board[19][19];
unsigned char CharBoard[29][29];
const int (*ScoreHash)[2][2][256];
double _FACTOR_;

#include "eval.inc"
#include "tss.inc"

// ---- Zobrist transposition table ----
static uint64_t Z[19][19][3];          // [row][col][0=opp,1=self,2=empty]
static uint64_t g_hash;
static void zinit() {
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    for (int r = 0; r < 19; ++r) for (int c = 0; c < 19; ++c) for (int k = 0; k < 3; ++k) {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        Z[r][c][k] = z ^ (z >> 31);
    }
}
struct TTEntry { uint64_t key; int score; short depth; short flag; };   // flag: 0 exact,1 lower,2 upper
static TTEntry* tt = 0;
static int ttMask = 0;
static const int TT_BITS = 18;                    // 2^18 entries (~7MB)
static void tt_init() {
    if (tt) return;
    ttMask = (1 << TT_BITS) - 1;
    tt = (TTEntry*)malloc(sizeof(TTEntry) * (size_t)(ttMask + 1));
    if (tt) memset(tt, 0, sizeof(TTEntry) * (size_t)(ttMask + 1));
}
static inline void tt_store(uint64_t key, int depth, int score, int flag) {
    if (!tt) return;
    TTEntry& e = tt[key & (uint64_t)ttMask];
    if (e.key == key && e.depth > depth) return;
    e.key = key; e.depth = (short)depth; e.score = score; e.flag = (short)flag;
}
static inline int tt_probe(uint64_t key, int depth, int alpha, int beta, int* hit) {
    if (!tt) { *hit = 0; return 0; }
    TTEntry& e = tt[key & (uint64_t)ttMask];
    if (e.key != key) { *hit = 0; return 0; }
    *hit = 1;
    if (e.depth < depth) return 0;
    if (e.flag == 0) return e.score;
    if (e.flag == 1 && e.score >= beta) return e.score;
    if (e.flag == 2 && e.score <= alpha) return e.score;
    return 0;
}

// ---- board helpers (self=ColorHash[1]=7, opp=ColorHash[0]=1) ----
static const unsigned char SELF = ColorHash[1], OPP = ColorHash[0];
// ---- incremental NNUE (Rapfi-style: update only affected first-layer column) ----
static float g_nnA1[NN_H1];   // tracks x @ NN_W1 incrementally (board 722 dims)
static float g_feat12[14];    // 14 features (recomputed on each forward)
static inline unsigned char get(int r, int c);   // fwd decl
// compute the 12 tactical features from the relative board (SELF/OPP)
static void nnue_feat12() {   // computes 14 features
    int dr[4]={0,1,1,1}, dc[4]={1,0,1,-1};
    int ts[5]={0,0,0,0,0}, to[5]={0,0,0,0,0};
    for (int pass=0; pass<2; ++pass) {
        unsigned char col = pass ? (unsigned char)OPP : (unsigned char)SELF;
        int* o = pass ? to : ts;
        for (int r=0;r<19;r++) for (int c=0;c<19;c++) {
            if ((unsigned char)get(r,c)!=col) continue;
            for (int d=0;d<4;d++) {
                int pr=r-dr[d], pc=c-dc[d];
                if (pr>=0&&pr<19&&pc>=0&&pc<19&&(unsigned char)get(pr,pc)==col) continue;
                int n=0, rr=r, cc=c;
                while (rr>=0&&rr<19&&cc>=0&&cc<19&&(unsigned char)get(rr,cc)==col) { n++; rr+=dr[d]; cc+=dc[d]; }
                int r0=r-dr[d], c0=c-dc[d];
                bool openB=(r0>=0&&r0<19&&c0>=0&&c0<19&&get(r0,c0)==0);
                bool openA=(rr>=0&&rr<19&&cc>=0&&cc<19&&get(rr,cc)==0);
                int opens=openB+openA;
                // SUB-RUN counting (matches gen countThreats): n-run -> 1 openN + (n-1) openN-1 + ... + (n-1) open2
                for (int k=n; k>=2; --k) {
                    int sub = n - k + 1;
                    if (opens==2) { if (k==2) o[0]+=sub; else if (k==3) o[1]+=sub; else if (k>=4) o[3]+=sub; }
                    else if (opens==1) { if (k==3) o[2]+=sub; else if (k==2) o[0]+=sub; else if (k>=4) o[3]+=sub; }
                }
            }
        }
    }
    // E1 EXPERIMENT: emit features in the training-DATA convention (computeFeat14
    // ternary is selfCol=mover?1:2, i.e. swapped: data f[0..4]=OPP threats,
    // f[5..9]=SELF threats, f[12/13] SELF-side). Engine's ts=SELF, to=OPP.
    for (int i=0;i<5;i++){ g_feat12[i]=(float)to[i]; g_feat12[5+i]=(float)ts[i]; }
    int stones=0; for (int r=0;r<19;r++) for (int c=0;c<19;c++) if (get(r,c)!=0) stones++;
    g_feat12[10]=is_white?1.0f:0.0f;   // 1 = black to move (matches gen computeFeat14: mover?0:1; is_white==!g_ai_is_white -> SELF is black; was hardcoded 1.0f -> wrong for white to move)
    g_feat12[11]=(float)stones/361.0f;
    // FIX (encoding consistency): the training data (gen computeFeat14) puts the OPPONENT's
    // threat flags in f[12]/f[13] (to[]).  Match it here (was ts[] -> swapped defense signal).
    g_feat12[12] = (to[3] > 0 || to[1] >= 2) ? 1.0f : 0.0f;   // opp_threatens_win
    g_feat12[13] = (to[3] > 0) ? 1.0f : 0.0f;                  // opp_has_open4
}
static inline void nnue_touch(int cell, int plane, float delta) {
    // plane 0 = self, 1 = opp; feature index = plane*361 + cell
    const float* col = &NN_W1[(plane*361 + cell)*NN_H1];
    for (int j = 0; j < NN_H1; ++j) g_nnA1[j] += delta * col[j];
}

static inline int zidx(unsigned char v) { return v == SELF ? 1 : (v == OPP ? 0 : 2); }
extern "C" void b6_set_net(int which);
extern "C" void b6_nnue_recompute();
void b6_nnue_recompute() {
    for (int j = 0; j < NN_H1; ++j) g_nnA1[j] = 0.0f;
    for (int r = 0; r < 19; ++r) for (int c = 0; c < 19; ++c) {
        // BUGFIX: use CharBoard codes (SELF=7/OPP=1) like put() does. The old
        // code read the short board[][] (holds 0/1/-1 color bits) and compared
        // against SELF=7/OPP=1 -> never matched SELF, black stones landed in the
        // OPP plane, white stones dropped entirely. Every set_net/set_perspective
        // rebuilt a corrupted accumulator (put()'s incremental path was fine).
        unsigned char v = (unsigned char)get(r, c);
        if (v == SELF) nnue_touch(r*19+c, 0, +1.0f);
        else if (v == OPP) nnue_touch(r*19+c, 1, +1.0f);
    }
}
void b6_set_net(int which) { g_net = which; b6_nnue_recompute(); }
extern "C" void b6_get_feat12(float* out) { nnue_feat12(); for (int i=0;i<14;i++) out[i]=g_feat12[i]; }
long g_place=0,g_unplace=0;
static inline void put(int r, int c, unsigned char v) {
    if (v) g_place++; else g_unplace++;
    unsigned char old = CharBoard[r + 5][c + 5];
    CharBoard[r + 5][c + 5] = v;
    if (NNUE_ON) { // incremental NNUE: remove old, add new
      int cell = r * 19 + c;
      if (old == SELF) nnue_touch(cell, 0, -1.0f);
      else if (old == OPP) nnue_touch(cell, 1, -1.0f);
      if (v == SELF) nnue_touch(cell, 0, +1.0f);
      else if (v == OPP) nnue_touch(cell, 1, +1.0f); }
    g_hash ^= Z[r][c][zidx(old)] ^ Z[r][c][zidx(v)];   // correct symmetric update
    board[r][c] = (short)(v == SELF ? is_white : (v == OPP ? !is_white : -1));
}
static inline unsigned char get(int r, int c) { return CharBoard[r + 5][c + 5]; }

// static eval from self's perspective (higher = better for self).
// Threat-aware window scan: strongly rewards building sixes / fives / fours and
// punishes the opponent's same threats, so the search blocks & builds properly.
// int8 QUANTIZED forward (QAT): uses int8 weights (dequantized) -- self-play inference path
static int nnue_forward_int8() {
    float a1[NN_H1];
    for (int j = 0; j < NN_H1; ++j) { float z = NN_B1[j]; const signed char* col = &Q_W1[j*NN_D]; for (int i = 0; i < NN_D; ++i) z += 0; (void)col; a1[j]=z; }  // (first layer via incremental g_nnA1 below)
    for (int j = 0; j < NN_H1; ++j) {
        float z = g_nnA1[j] + NN_B1[j];
        for (int k = 0; k < 14; ++k) z += g_feat12[k] * NW1[g_net][(722+k)*NN_H1 + j];   // 12 features
        a1[j] = z > 0 ? z : 0;
    }
    float a2[NN_H2];
    for (int j = 0; j < NN_H2; ++j) { float z = NN_B2[j]; const signed char* row = &Q_W2[j*NN_H1]; for (int i = 0; i < NN_H1; ++i) z += a1[i]*(row[i]*Q_S2); a2[j] = z > 0 ? z : 0; }
    float out = NN_B3[0];
    for (int i = 0; i < NN_H2; ++i) out += a2[i]*(Q_W3[i]*Q_S3);
    return (int)(out * NN_VSCALE);
}
// forward from incremental g_nnA1 -> value (self's advantage, engine scale)
static int nnue_forward() {
    nnue_feat12();   // compute the 12 tactical features for this position
    float a1[NN_H1];
    for (int j = 0; j < NN_H1; ++j) {
        float z = g_nnA1[j] + NN_B1[j];
        for (int k = 0; k < 14; ++k) z += g_feat12[k] * NW1[g_net][(722+k)*NN_H1 + j];   // 12 features
        a1[j] = z > 0 ? z : 0;
    }
    float a2[NN_H2];
    for (int j = 0; j < NN_H2; ++j) { float z = NN_B2[j]; const float* row = &NN_W2[j*NN_H1]; for (int i = 0; i < NN_H1; ++i) z += a1[i]*row[i]; a2[j] = z > 0 ? z : 0; }
    float out = NN_B3[0];
    for (int i = 0; i < NN_H2; ++i) out += a2[i]*NN_W3[i];
    return (int)(out * NN_VSCALE);
}

// ---- connect6 evaluation: run-based with OPENNESS + fork (NCTU6-style shapes) ----
// Score maximal runs of stones by (length, open-ends).  Open-5 (活五) >> closed-5,
// open-4 (活四, win-in-2 since we place 2) >> closed-4.  Opp runs weigh slightly
// more (defensive).  Plus a fork/double-threat bonus (decisive in connect6).
static int runScore(int len, int open) {
    static const int RS[7][3] = {
        // open=0     1        2
        {0,0,0},        // L=0
        {0,4,14},       // L=1
        {0,18,70},      // L=2
        {0,150,650},    // L=3
        {0,3500,26000}, // L=4  open-4 = win in 2
        {0,80000,420000}, // L=5 open-5 = win in 1
        {2000000000,2000000000,2000000000} // L=6 win
    };
    if (len >= 6) return RS[6][2];
    return RS[len][open];
}
static int g_cand_scoring = 0;   // 1 while gen_candidates ranks cells (NNUE blend suppressed there)
static int evaluate() {
    static const int DR[4] = { 0, 1, 1, 1 }, DC[4] = { 1, 0, 1, -1 };
    int score = 0, selfThreats = 0, oppThreats = 0;
    for (int d = 0; d < 4; ++d) {
        for (int r = 0; r < 19; ++r) for (int c = 0; c < 19; ++c) {
            unsigned char v = TSS_AT(CharBoard, r, c);
            if (v != SELF && v != OPP) continue;
            int pr = r - DR[d], pc = c - DC[d];
            if (pr >= 0 && pr < 19 && pc >= 0 && pc < 19 && TSS_AT(CharBoard, pr, pc) == v) continue; // not run start
            int L = 0, rr = r, cc = c;
            while (rr >= 0 && rr < 19 && cc >= 0 && cc < 19 && TSS_AT(CharBoard, rr, cc) == v) { ++L; rr += DR[d]; cc += DC[d]; }
            int open = 0;
            if (pr < 0 || pr >= 19 || pc < 0 || pc >= 19 || TSS_AT(CharBoard, pr, pc) == 0) ++open;
            if (rr < 0 || rr >= 19 || cc < 0 || cc >= 19 || TSS_AT(CharBoard, rr, cc) == 0) ++open;
            int sc = runScore(L, open);
            if (v == SELF) { score += sc; if (L >= 4 && open >= 1) ++selfThreats; }
            else           { score -= sc; if (L >= 4 && open >= 1) ++oppThreats; }   // EQUAL weight (zero-sum)
        }
    }
    // fork / double-threat: two+ open fours/fives for one side = decisive
    if (selfThreats >= 2) score += 7000000 * (selfThreats - 1);
    if (oppThreats  >= 2) score -= 7000000 * (oppThreats  - 1);   // EQUAL (zero-sum)
    // GAP-AWARE window threats: a 6-window with 4+ stones of one colour (no opp)
    // is a threat EVEN WITH GAPS (e.g. XX_XX = win-in-2).  Run-based scoring misses
    // these; the window scan makes the SEARCH "think" to block (not just rely on TSS).
    { static const int GDR[4]={0,1,1,1}, GDC[4]={1,0,1,-1};
      int selfGap=0, oppGap=0;
      for (int d=0; d<4; ++d) for (int r=0; r<19; ++r) for (int c=0; c<19; ++c) {
          int lr=r+5*GDR[d], lc=c+5*GDC[d]; if (lr<0||lr>18||lc<0||lc>18) continue;
          int sc=0, oc=0;
          for (int i=0;i<6;i++){ unsigned char v=TSS_AT(CharBoard, r+i*GDR[d], c+i*GDC[d]); if(v==SELF)++sc; else if(v==OPP)++oc; }
          if (oc==0 && sc>=4) ++selfGap;
          if (sc==0 && oc>=4) ++oppGap;
      }
      // weight by DANGER: a window with 5 opp = win-in-1 (near-loss); 4 opp = win-in-2
      int selfBig=0, oppBig=0;
      for (int d=0; d<4; ++d) for (int r=0; r<19; ++r) for (int c=0; c<19; ++c) {
          int lr=r+5*GDR[d], lc=c+5*GDC[d]; if (lr<0||lr>18||lc<0||lc>18) continue;
          int sc=0, oc=0;
          for (int i=0;i<6;i++){ unsigned char v=TSS_AT(CharBoard, r+i*GDR[d], c+i*GDC[d]); if(v==SELF)++sc; else if(v==OPP)++oc; }
          if (oc==0 && sc==5) ++selfBig;
          if (sc==0 && oc==5) ++oppBig;
      }
      if (selfGap) score += 26000 * selfGap;
      if (oppGap)  score -= 26000 * oppGap;   // EQUAL (zero-sum)
      if (selfBig) score += 3000000 * selfBig;
      if (oppBig)  score -= 3000000 * oppBig;  // EQUAL (zero-sum)
    }
    // blend in the NNUE (Rapfi-style classical + NNUE).  NNUE_ALPHA tuned as the
    // network trains; keep hand-crafted dominant until the net is well-trained.
    if (NNUE_ON && !g_cand_scoring) { const float A = 0.55f;   // NNUE as PRIMARY evaluator (leaf eval only; candidate scoring is heuristic-only, see gen_candidates)
        return (int)((1.0f-A)*(float)score + A*(float)nnue_forward()); }
    return score;
}

// ---- move generation: candidate cells near stones, then pairs ----
static int cand[256][2], candScore[256], candN;
// FIX 2026-10-16: candidate ranking must NOT use the NNUE leaf blend.
// With the net in the loop the noisy value delta pushed real block/fork cells out of
// the top-K candidate window -> the pair generator never produced the defensive
// move -> the search saw phantom forced wins (WIN-ply) everywhere. Rank candidates
// with the hand-crafted heuristic only; the NNUE still contributes at leaf evals.
static void gen_candidates(int to_move) {
    candN = 0;
    // if board is empty-ish, use centre neighbourhood
    int any = 0;
    for (int r = 0; r < 19 && !any; ++r) for (int c = 0; c < 19; ++c) if (get(r, c)) { any = 1; break; }
    for (int r = 0; r < 19; ++r) for (int c = 0; c < 19; ++c) {
        if (get(r, c) != 0) continue;
        int near = any ? 0 : 1;
        if (!any) {
            // empty board: only central points
            if (abs(r - 9) <= 3 && abs(c - 9) <= 3) near = 1;
        } else {
            for (int dr = -3; dr <= 3 && !near; ++dr) for (int dc = -3; dc <= 3; ++dc) {
                int rr = r + dr, cc = c + dc;
                if (rr >= 0 && rr < 19 && cc >= 0 && cc < 19 && get(rr, cc)) { near = 1; break; }
            }
        }
        if (!near) continue;
        // quick score: gain of placing `to_move` here (local eval delta)
        unsigned char code = to_move ? SELF : OPP;
        put(r, c, code);
        g_cand_scoring = 1;
        int s = evaluate();
        g_cand_scoring = 0;
        put(r, c, 0);
        // score from the MOVER's perspective (evaluate() is self's advantage:
        // self maximises it, opponent minimises it) so top-K picks each side's best.
        cand[candN][0] = r; cand[candN][1] = c; candScore[candN] = to_move ? s : -s;
        ++candN;
        if (candN >= 250) { r = 19; break; }
    }
}
static int K = 12;
struct Move { int r1, c1, r2, c2, score; };
static Move moves[40000];
static int movesN;
static void gen_moves(int to_move) {
    gen_candidates(to_move);
    // pick top-K candidates
    int idx[256]; for (int i = 0; i < candN; ++i) idx[i] = i;
    int kk = candN < K ? candN : K;
    // partial selection sort for top-kk
    for (int a = 0; a < kk; ++a) {
        int best = a;
        for (int b = a + 1; b < candN; ++b) if (candScore[idx[b]] > candScore[idx[best]]) best = b;
        swap(idx[a], idx[best]);
    }
    movesN = 0;
    for (int a = 0; a < kk; ++a) {
        for (int b = a + 1; b < kk; ++b) {
            int i = idx[a], j = idx[b];
            moves[movesN].r1 = cand[i][0]; moves[movesN].c1 = cand[i][1];
            moves[movesN].r2 = cand[j][0]; moves[movesN].c2 = cand[j][1];
            // SYNERGY: prefer the 2 stones working TOGETHER (a line/threat), not
            // two scattered "individually-best" cells.  Connect6 places 2/turn, so
            // a pair must build a combined shape.
            { int dr = cand[i][0]-cand[j][0]; if (dr<0) dr=-dr;
              int dc = cand[i][1]-cand[j][1]; if (dc<0) dc=-dc;
              int dist = dr>dc?dr:dc;
              int syn = 0;
              if (dist <= 2) syn += 1200;                    // adjacent: build together
              if (dr==0 || dc==0 || dr==dc) syn += 900;       // aligned: forms a line
              moves[movesN].score = candScore[i] + candScore[j] + syn; }
            ++movesN;
            if (movesN >= 39998) { a = kk; break; }
        }
    }
    // sort moves by score desc
    for (int a = 0; a < movesN; ++a) {
        int best = a;
        for (int b = a + 1; b < movesN; ++b) if (moves[b].score > moves[best].score) best = b;
        if (best != a) { Move t = moves[a]; moves[a] = moves[best]; moves[best] = t; }
        if (a > 64) break;   // only need decent ordering; cap sort work
    }
}

static const int WIN = 2000000000;

static long long g_deadline = 0;   // wall-clock ms (robust under emscripten asm.js)
static inline long long nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}
static int g_stop = 0;

// negamax with alpha-beta; returns score for side to move.

// terminal win check: does `code` have 6-in-a-row on the board right now?
static int made_six(unsigned char code) {
    static const int DR[4]={0,1,1,1}, DC[4]={1,0,1,-1};
    int calls=0;
    for (int r=0;r<19;r++) for (int c=0;c<19;c++) {
        if (TSS_AT(CharBoard,r,c)!=code) continue;
        for (int d=0;d<4;d++) {
            int n=0;
            for (int i=0;i<6;i++){ int rr=r+i*DR[d], cc=c+i*DC[d]; if(rr<0||rr>18||cc<0||cc>18||TSS_AT(CharBoard,rr,cc)!=code) break; n++; }
            if (n>=6) return 1;   // FIX 2026-10-16: was counted-but-never-tested, always returned 0
        }
    }
    return 0;
}

// ---- forced-move detection (win-in-1 / loss-in-1), done CORRECTLY ----
// If we can make 6 now -> take it.  If the opponent can make 6 next (a live-5 /
// open-4 that completes in 1), block BOTH ends so they can't.  This is a forced
// move (must do it or lose) -- placed here so the search never misses it.
static int forced_move(unsigned char self, unsigned char opp, int* out) {
    static const int DR[4]={0,1,1,1}, DC[4]={1,0,1,-1};
    // 1) can SELF win now?  (find 2 empties that complete a 6)
    // scan self windows with >=4 self + empties
    int bestCov=-1, bestInd=-1, b1r=-1,b1c=-1,b2r=-1,b2c=-1;
    int winC[32][2][2], winCN[32], nw=0;   // opp's winning windows
    for (int d=0; d<4; d++) for (int r=0;r<19;r++) for (int c=0;c<19;c++) {
        int lr=r+5*DR[d], lc=c+5*DC[d]; if (lr<0||lr>18||lc<0||lc>18) continue;
        int sc=0,oc=0,ec=0,e[2][2],en=0;
        for (int i=0;i<6;i++){ unsigned char v=TSS_AT(CharBoard,r+i*DR[d],c+i*DC[d]);
            if(v==self)++sc; else if(v==opp)++oc; else { if(en<2){e[en][0]=r+i*DR[d];e[en][1]=c+i*DC[d];} ++en; ++ec; } }
        // self win-in-1: 5 self + 1 empty -> place it -> 6
        if (sc==5 && ec==1) {   // VERIFY: placing e[0] must REALLY make 6 consecutive
            put(e[0][0], e[0][1], self); int real = made_six(self); put(e[0][0], e[0][1], 0);
            if (real) { out[0]=e[0][0]; out[1]=e[0][1]; tss_pick2nd(CharBoard,out[0],out[1],out[0],out[1],&out[2],&out[3]); return 1; }
        }
        // opp win window: opp fills empties to 6 (ec<=2, no self) -> we must block
        if (sc==0 && ec<=2 && oc+ec==6 && nw<32) {
            // VERIFY: filling the empties must really make 6 consecutive for opp
            for (int k=0;k<en;k++) put(e[k][0], e[k][1], opp);
            int real = made_six(opp);
            for (int k=0;k<en;k++) put(e[k][0], e[k][1], 0);
            if (real) { winCN[nw]=en<2?en:2; for(int k=0;k<winCN[nw];k++){winC[nw][k][0]=e[k][0];winC[nw][k][1]=e[k][1];} ++nw; }
        }
    }
    if (nw>0) {
        // pick 2 cells covering the most opp win windows (block both ends)
        for (int i=0;i<nw;i++) for (int k=0;k<winCN[i];k++) for (int j2=0;j2<nw;j2++) for (int l=0;l<winCN[j2];l++) {
            int r1=winC[i][k][0],c1=winC[i][k][1],r2=winC[j2][l][0],c2=winC[j2][l][1];
            if (r1==r2&&c1==c2) continue;
            int cov=0; for (int w=0;w<nw;w++){ int hit=0; for(int m=0;m<winCN[w];m++){ if((winC[w][m][0]==r1&&winC[w][m][1]==c1)||(winC[w][m][0]==r2&&winC[w][m][1]==c2)){hit=1;break;} } cov+=hit; }
            int c1n=0,c2n=0; for(int w=0;w<nw;w++){ for(int m=0;m<winCN[w];m++){ if(winC[w][m][0]==r1&&winC[w][m][1]==c1)c1n++; if(winC[w][m][0]==r2&&winC[w][m][1]==c2)c2n++; } }
            int ind=c1n+c2n;
            if (cov>bestCov || (cov==bestCov && ind>bestInd)){ bestCov=cov; bestInd=ind; b1r=r1;b1c=c1;b2r=r2;b2c=c2; }
        }
        if (b1r>=0){ out[0]=b1r;out[1]=b1c;out[2]=b2r;out[3]=b2c; return 1; }
    }
    return 0;
}

static int ab_search(int depth, int alpha, int beta, int self_to_move, int ply) {
    if (g_deadline && nowMs() > g_deadline) { g_stop = 1; return 0; }
    // win/block via TSS at every node (exact tactics)
    int tmp[4];
    unsigned char code = self_to_move ? SELF : OPP;
    unsigned char ocode = self_to_move ? OPP : SELF;
    if (tss_win_now(CharBoard, code, ocode, tmp)) return WIN - ply;         // side to move wins now

    int hit = 0;
    int tval = tt_probe(g_hash, depth, alpha, beta, &hit);
    if (hit && (tval <= alpha || tval >= beta || tval == 0)) {
        if (tval != 0 || depth <= 0) return tval;
    }
    if (depth <= 0) {
        int e = evaluate();
        return self_to_move ? e : -e;
    }

    gen_moves(self_to_move);
    if (movesN == 0) { if(ply<=2)fprintf(stderr,"  ab(ply%d d%d) NOMOVES\n",ply,depth); int e = evaluate(); return self_to_move ? e : -e; }

    int best = -WIN + ply;
    int flag = 2;   // upper
    // NULL-MOVE PRUNING: if passing (opp gets 2 extra stones) we're still >= beta, prune.
    if (depth >= 3 && beta < WIN && beta > -WIN && !g_stop) {
        int R = 2 + (depth >= 6 ? 2 : depth / 3);
        int nv = -ab_search(depth - 1 - R, -beta, -beta + 1, (unsigned char)(self_to_move ^ 1), ply + 1);
        if (nv >= beta) return beta;
    }
    int myMovesN = movesN; if (myMovesN > 256) myMovesN = 256;   // cap
    Move local[256]; for (int z = 0; z < myMovesN; ++z) local[z] = moves[z];   // STABLE copy (moves[] is clobbered by recursion)
    for (int m = 0; m < myMovesN && !g_stop; ++m) {
        if (g_deadline && nowMs() > g_deadline) { g_stop = 1; break; }
        Move mv = local[m];
        // LATE MOVE REDUCTIONS: later (likely bad) moves searched shallower; re-search if good
        int doLMR = (depth >= 3 && m >= 4 && !g_stop);
        int red = doLMR ? (1 + (m >= 10 ? 1 : 0)) : 0;
        put(mv.r1, mv.c1, code);
        put(mv.r2, mv.c2, code);
        if (made_six(code)) {          // terminal: the mover just won
            put(mv.r1, mv.c1, 0); put(mv.r2, mv.c2, 0);
            return WIN - ply;
        }
        int v = -ab_search(depth - 1 - red, -beta, -alpha, !self_to_move, ply + 1);
        if (red && v > alpha) v = -ab_search(depth - 1, -beta, -alpha, !self_to_move, ply + 1);   // re-search full
        put(mv.r1, mv.c1, 0);
        put(mv.r2, mv.c2, 0);
        if (g_stop) break;
        if (v > best) best = v;
        if (v > alpha) { alpha = v; flag = 0; }
        if (alpha >= beta) { flag = 1; break; }
    }
    tt_store(g_hash, depth, best, flag);
    return best;
}

// ---- C ABI (identical to the greedy version) ----
static int g_ai_is_white = 1;
static unsigned char g_self = ColorHash[1], g_opp = ColorHash[0];

extern "C" {
void b6_new_game(int ai_is_white) {
    (void)ai_is_white;
    g_ai_is_white = ai_is_white ? 1 : 0;
    is_white = (unsigned short int)(ai_is_white ? 0 : 1);
    ScoreHash = is_white ? ScoreHash2 : ScoreHash1;
    zinit(); tt_init();
    memset(CharBoard, 46, sizeof(CharBoard));
    for (int j = 0; j < 19; ++j) for (int i = 0; i < 19; ++i) CharBoard[j + 5][i + 5] = 0;
    memset(board, -1, sizeof(board));
    g_hash = 0;
}
// FIX: make the search play for `colour` (the side to move), not always for g_ai_is_white.
extern "C" void b6_set_perspective(int colour);
void b6_set_perspective(int colour) {
    int want = colour ? 1 : 0;
    if (want == g_ai_is_white) return;
    int old = is_white;
    g_ai_is_white = want;
    is_white = (unsigned short int)(want ? 0 : 1);
    ScoreHash = is_white ? ScoreHash2 : ScoreHash1;
    for (int r=0;r<29;r++) for (int c=0;c<29;c++) {
        if (CharBoard[r][c]==SELF) CharBoard[r][c]=OPP;
        else if (CharBoard[r][c]==OPP) CharBoard[r][c]=SELF;
    }
    for (int r=0;r<19;r++) for (int c=0;c<19;c++) if (board[r][c]>=0) board[r][c] = (short)(1-board[r][c]);
    b6_nnue_recompute();
}
void b6_play(int r, int c, int colour) {
    if (r < 0 || r > 18 || c < 0 || c > 18) return;
    put(r, c, (colour == g_ai_is_white) ? SELF : OPP);
}
void b6_clear(int r, int c) {
    if (r < 0 || r > 18 || c < 0 || c > 18) return;
    put(r, c, 0);
}
int b6_cell(int r, int c) {
    unsigned char code = get(r, c);
    if (code == 0) return 0;
    if (code == SELF) return g_ai_is_white ? 2 : 1;
    return g_ai_is_white ? 1 : 2;
}
int b6_win_at(int r, int c, int colour) {
    unsigned char code = (colour == g_ai_is_white) ? SELF : OPP;
    return is_gameover_char(CharBoard, (char)(r + 5), (char)(c + 5), (char)(r + 5), (char)(c + 5), code) ? 1 : 0;
}
int b6_opening(int* out) { out[0] = 9; out[1] = 9; return 1; }

// ---- opening book (real engine data from the preloaded /book.bin) ----
static unsigned char* g_book = 0;
static int g_book_n = 0;
static void book_load() {
    if (g_book) return;
    FILE* f = fopen("/book.bin", "rb");
    if (!f) return;
    unsigned char hdr[8]; if (fread(hdr, 1, 8, f) != 8) { fclose(f); return; }
    g_book_n = hdr[4] | (hdr[5] << 8) | (hdr[6] << 16) | (hdr[7] << 24);
    if (g_book_n <= 0) { fclose(f); return; }
    g_book = (unsigned char*)malloc((size_t)g_book_n * 366);
    if (!g_book) { g_book_n = 0; fclose(f); return; }
    if (fread(g_book, 1, (size_t)g_book_n * 366, f) != (size_t)g_book_n * 366) { free(g_book); g_book = 0; g_book_n = 0; }
    fclose(f);
}
static int book_lookup(int self_is_white, int* out) {
    if (!g_book) return 0;
    unsigned char bkey[362];
    for (int r = 0; r < 19; ++r) for (int c = 0; c < 19; ++c) bkey[r * 19 + c] = (unsigned char)b6_cell(r, c);
    bkey[361] = (unsigned char)(self_is_white ? 1 : 0);
    for (int e = 0; e < g_book_n; ++e) {
        unsigned char* p = g_book + (size_t)e * 366;
        int ok = 1; for (int k = 0; k < 362; ++k) if (p[k] != bkey[k]) { ok = 0; break; }
        if (!ok) continue;
        int r1 = p[362], c1 = p[363], r2 = p[364], c2 = p[365];
        if (b6_cell(r1, c1) != 0 || b6_cell(r2, c2) != 0) return 0;
        out[0] = r1; out[1] = c1; out[2] = r2; out[3] = c2; return 1;
    }
    return 0;
}

// Return the SEARCH VALUE of the CURRENT position from `colour`'s perspective.
// Used by the NNUE data generator (search value = a better training target than
// the raw hand-crafted eval, and less biased than win/loss in first-player-win connect6).
extern "C" int b6_eval_value(int colour, int depth) {
    int self_to_move = (colour == g_ai_is_white) ? 1 : 0;
    g_stop = 0; g_deadline = 0;   // no time cap; fixed depth
    return ab_search(depth, -WIN, WIN, self_to_move, 0);   // perspective-relative (zero-sum after equal weights)
}

int b6_search(int colour, int maxDepth, int timeLimitMs, int* out) {
    // 1) TSS: exact win / forced block first.
    if (out) {
        // TSS REMOVED per user request: let the SEARCH decide all wins/blocks
        // (tss_win_now / tss_block disabled -- the search "thinks" for everything).
    }
    // 2) Opening book.
    book_load();
    // DIAG: book off
    // RULE FIX: connect6 first move = 1 stone (only when the board is empty).
    if (out) {
        int stoneCnt = 0; for (int r=0;r<19;r++) for (int c=0;c<19;c++) if (get(r,c)) { stoneCnt=1; break; }
        if (stoneCnt == 0) {
            gen_moves(1);
            if (movesN > 0) {
                int bestV = -WIN, bestR = moves[0].r1, bestC = moves[0].c1;
                for (int m = 0; m < movesN; ++m) {
                    put(moves[m].r1, moves[m].c1, SELF);
                    int v = evaluate();
                    put(moves[m].r1, moves[m].c1, 0);
                    if (v > bestV) { bestV = v; bestR = moves[m].r1; bestC = moves[m].c1; }
                }
                out[0] = bestR; out[1] = bestC; out[2] = -1; out[3] = -1;
                return 1;
            }
        }
    }
    // 2.5) FORCED MOVES (win-in-1 / loss-in-1) done correctly -- never miss a live-5/open-4 block.
    // DIAG: forced off
    // 3) alpha-beta with iterative deepening (depth- and/or time-capped).
    g_stop = 0;
    g_deadline = (timeLimitMs > 0) ? nowMs() + timeLimitMs : 0;
    Move best; best.r1 = best.c1 = best.r2 = best.c2 = -1;
    int topDepth = (maxDepth > 0) ? maxDepth : 40;   // maxDepth 0 => time-limited deep search
    int dReached = 0;
    for (int d = 1; d <= topDepth && !g_stop; ++d) {
        dReached = d;
        gen_moves(1);
        if (movesN == 0) break;
        Move iterBest = moves[0];   // (valid: before any recursion)
        int alpha = -WIN, beta = WIN;
        int myMovesN = movesN; if (myMovesN > 256) myMovesN = 256;
        Move local[256]; for (int z = 0; z < myMovesN; ++z) local[z] = moves[z];   // STABLE copy
        for (int m = 0; m < myMovesN && !g_stop; ++m) {
            if (g_deadline && nowMs() > g_deadline) { g_stop = 1; break; }
            Move mv = local[m];
            put(mv.r1, mv.c1, SELF);
            put(mv.r2, mv.c2, SELF);
            if (made_six(SELF)) {       // terminal: I win now
                put(mv.r1, mv.c1, 0); put(mv.r2, mv.c2, 0);
                best = mv; g_stop = 1; break;
            }
            int v = -ab_search(d - 1, -beta, -alpha, 0, 1);
            put(mv.r1, mv.c1, 0);
            put(mv.r2, mv.c2, 0);
            if (g_stop) break;
            if (v > alpha) { alpha = v; iterBest = mv; }   // ROOT FIX: track the best move
        }
        if (!g_stop) best = iterBest;
        if (maxDepth > 0 && d >= maxDepth) break;   // fixed depth reached
    }
    if (best.r1 < 0) {
        // fallback: first candidate pair
        gen_moves(1);
        if (movesN > 0) best = moves[0];
        else {
            if (out) { out[0] = 9; out[1] = 9; out[2] = 9; out[3] = 10; }
            return 0;
        }
    }
    
    if (getenv("DBGD")) fprintf(stderr,"[depth reached=%d]\n", dReached);
    if (out) { out[0] = best.r1; out[1] = best.c1; out[2] = best.r2; out[3] = best.c2; }
    return 1;
}
}  // extern "C"
