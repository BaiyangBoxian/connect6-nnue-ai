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
// verify_fork.cpp -- NETWORK-DRIVEN fork/block tactics gate (v2 draft, 2026-10-16)
//
// WHY THE REWRITE (v1 was a no-op guard):
//   v1 placed a FIXED stone sequence and judged it with b6_get_feat12().
//   feat12 is a pure board pattern counter -- it never touches the NNUE
//   weights -- so the verdict was a constant function of the position:
//   every old/new network got "FAIL", the gate always took the keep branch
//   and could never detect a tactical regression. Two more v1 bugs:
//   (a) it read f[0..4] believing they were SELF threat counts; in the
//       engine's feat12 layout f[0..4]=OPP, f[5..9]=SELF (see nnue_feat12:
//       g_feat12[i]=to[i]=OPP, g_feat12[5+i]=ts[i]=SELF). So v1 was
//       measuring the OPPONENT's threes after black's "fork".
//   (b) the label claimed moves [7,8]+[8,10] while it actually played
//       (10,9)+(9,6). v2 prints exactly the coordinates it plays.
//
// v2 DESIGN: the ENGINE ITSELF (b6_search with the loaded net) picks the move
// pair; the verdict is then a function of the network weights -> old/new nets
// can disagree, which is what a gate needs. Four scenarios:
//   F1/F2  fork-make  : engine to move with a double-three opportunity;
//                       it must actually CREATE the fork (>=2 open threes,
//                       or a four+three, or two fours).
//   B1/B2  fork-block : engine to move while the opponent threatens a fork;
//                       after the engine's move NO opponent stone pair may
//                       create a fork anymore (or the engine wins at once).
//
// feat12 slots used (engine layout): [1]=OPP open3, [3]=OPP open4,
// [6]=SELF open3, [8]=SELF open4. The double3 slots [4]/[9] are never filled
// by nnue_feat12 (stays 0), so the fork criterion is built from open3/open4:
//   fork := o3>=2 || o4>=2 || (o4>=1 && o3>=1)
// (sub-run counting: one open four also yields o3+=2, so an open four alone
//  passes; a lone sleep-four yields o4=1,o3=0 and correctly does NOT pass).
//
// usage: ./verify_fork [net]   net=0 -> A(new, default)  net=1 -> B(old)
// output stays grep-compatible with sN_round.sh: last PASS/FAIL is the verdict.

#include <cstdio>
#include <cstring>
#include <cstdlib>
extern "C" {
  void b6_new_game(int); void b6_play(int,int,int); void b6_clear(int,int);
  int  b6_cell(int,int); int b6_win_at(int,int,int);
  int  b6_search(int,int,int,int*);
  void b6_set_net(int); void b6_set_perspective(int);
  void b6_get_feat12(float*);
}

struct Stone { int r, c, col; };   // col: 0=black 1=white

static bool isFork(float o3, float o4) {
  return (o3 >= 2) || (o4 >= 2) || (o4 >= 1 && o3 >= 1);
}

// ---- F1/F2: engine (black, SELF) must create the double threat itself ----
static int test_fork_make(int id, const Stone* st, int n) {
  b6_new_game(0);
  b6_set_perspective(0);              // black = SELF
  for (int i = 0; i < n; i++) b6_play(st[i].r, st[i].c, st[i].col);
  int o[4] = {-1,-1,-1,-1};
  b6_search(0, 5, 500, o);            // the NETWORK chooses the move pair
  if (o[0] < 0 || b6_cell(o[0],o[1]) || b6_cell(o[2],o[3]) ||
      (o[0]==o[2] && o[1]==o[3])) {
    printf("F%d: engine move=[%d,%d]+[%d,%d] INVALID -> FAIL\n", id, o[0],o[1],o[2],o[3]);
    return 0;
  }
  b6_play(o[0], o[1], 0);
  b6_play(o[2], o[3], 0);
  float f[14]; b6_get_feat12(f);
  bool win = b6_win_at(o[0],o[1],0) || b6_win_at(o[2],o[3],0);
  bool ok = win || isFork(f[6], f[8]);          // SELF = black = mover
  printf("F%d: engine move=[%d,%d]+[%d,%d] self open3=%.0f open4=%.0f -> %s\n",
         id, o[0],o[1],o[2],o[3], f[6], f[8],
         ok ? (win ? "PASS(六连即胜)" : "PASS(双活三/双威胁)") : "FAIL(未制造双威胁)");
  return ok ? 1 : 0;
}

// ---- B1/B2: opponent (white) threatens a fork; engine must remove it ----
static int test_fork_block(int id, const Stone* st, int n) {
  b6_new_game(0);
  b6_set_perspective(0);              // black = SELF (blocks), white = OPP
  for (int i = 0; i < n; i++) b6_play(st[i].r, st[i].c, st[i].col);

  // sanity: white must be able to fork BEFORE black moves (scenario check)
  int cand[48][2], cn = 0;
  for (int r = 0; r < 19; r++) for (int c = 0; c < 19; c++) {
    if (b6_cell(r,c)) continue;
    bool near = false;
    for (int i = 0; i < n; i++) if (st[i].col == 1 &&
        abs(st[i].r - r) <= 3 && abs(st[i].c - c) <= 3) { near = true; break; }
    if (near && cn < 48) { cand[cn][0] = r; cand[cn][1] = c; cn++; }
  }
  auto whiteCanFork = [&](int* wr, int* wc) -> bool {
    for (int a = 0; a < cn; a++) for (int b = a+1; b < cn; b++) {
      b6_play(cand[a][0], cand[a][1], 1);
      b6_play(cand[b][0], cand[b][1], 1);
      float f[14]; b6_get_feat12(f);
      bool fork = isFork(f[1], f[3]);            // OPP = white threats
      b6_clear(cand[a][0], cand[a][1]);
      b6_clear(cand[b][0], cand[b][1]);
      if (fork) { if (wr) { wr[0]=cand[a][0]; wr[1]=cand[a][1];
                            wc[0]=cand[b][0]; wc[1]=cand[b][1]; } return true; }
    }
    return false;
  };
  // v2.1: count ALL forking white pairs (criterion sanity, see verdict below)
  auto whiteForkCount = [&]() -> int {
    int n = 0;
    for (int a = 0; a < cn; a++) for (int b = a+1; b < cn; b++) {
      if (b6_cell(cand[a][0],cand[a][1]) || b6_cell(cand[b][0],cand[b][1])) continue;
      b6_play(cand[a][0], cand[a][1], 1);
      b6_play(cand[b][0], cand[b][1], 1);
      float f[14]; b6_get_feat12(f);
      if (isFork(f[1], f[3])) ++n;
      b6_clear(cand[a][0], cand[a][1]);
      b6_clear(cand[b][0], cand[b][1]);
    }
    return n;
  };
  int wr[2], wc[2];
  if (!whiteCanFork(wr, wc)) {
    printf("B%d: SCENARIO BROKEN (white cannot fork even before the block) -> FAIL\n", id);
    return 0;
  }
  int forksBefore = whiteForkCount();

  int o[4] = {-1,-1,-1,-1};
  b6_search(0, 5, 500, o);            // the NETWORK chooses the blocking pair
  if (o[0] < 0 || b6_cell(o[0],o[1]) || b6_cell(o[2],o[3]) ||
      (o[0]==o[2] && o[1]==o[3])) {
    printf("B%d: engine move=[%d,%d]+[%d,%d] INVALID -> FAIL\n", id, o[0],o[1],o[2],o[3]);
    return 0;
  }
  b6_play(o[0], o[1], 0);
  b6_play(o[2], o[3], 0);
  if (b6_win_at(o[0],o[1],0) || b6_win_at(o[2],o[3],0)) {
    printf("B%d: engine move=[%d,%d]+[%d,%d] (即胜) -> PASS\n", id, o[0],o[1],o[2],o[3]);
    return 1;
  }
  // v2.1 verdict: with a 2-stone block it is IMPOSSIBLE to kill every white
  // follow-up fork pair, so the old "no residual fork at all" bar made B1/B2
  // constant FAIL for every net. Fair bar: the identified main fork pair must be
  // dismantled AND the total number of forking pairs must drop.
  int forksAfter = whiteForkCount();
  bool mainDead = b6_cell(wr[0],wr[1]) || b6_cell(wc[0],wc[1]);
  if (!mainDead) {
    b6_play(wr[0], wr[1], 1); b6_play(wc[0], wc[1], 1);
    float f[14]; b6_get_feat12(f);
    mainDead = !isFork(f[1], f[3]);
    b6_clear(wr[0], wr[1]); b6_clear(wc[0], wc[1]);
  }
  bool ok = mainDead && forksAfter < forksBefore;
  printf("B%d: engine move=[%d,%d]+[%d,%d] mainpair(%d,%d)(%d,%d) %s, forkpairs %d->%d -> %s\n",
         id, o[0],o[1],o[2],o[3], wr[0],wr[1],wc[0],wc[1],
         mainDead ? "dismantled" : "INTACT", forksBefore, forksAfter,
         ok ? "PASS(拆解主双威胁)" : "FAIL(未拆解双威胁)");
  return ok ? 1 : 0;
}

int main(int argc, char** argv) {
  int net = (argc > 1) ? atoi(argv[1]) : 0;
  b6_set_net(net);
  printf("[verify_fork v2] net=%d (%s)\n", net, net ? "B(old)" : "A(new)");

  // F1: horizontal two + vertical two converging on (9,8)+(8,9)
  Stone f1[] = {
    {9,6,0},{9,7,0},{6,9,0},{7,9,0},      // black's two pairs
    {4,4,1},{4,5,1},{14,14,1},{14,15,1}   // harmless white stones
  };
  // F2: mirrored geometry converging on (9,10)+(10,9)
  Stone f2[] = {
    {9,11,0},{9,12,0},{11,9,0},{12,9,0},
    {4,14,1},{4,15,1},{14,4,1},{14,5,1}
  };
  // B1: white holds the F1-style fork threat; black must dismantle it
  Stone b1[] = {
    {9,6,1},{9,7,1},{6,9,1},{7,9,1},      // white threatens (9,8)+(8,9)
    {4,4,0},{4,5,0},{14,14,0},{14,15,0}
  };
  // B2: white's fork point is the shared cell (9,8): gap-two row 9 + two col 8
  Stone b2[] = {
    {9,7,1},{9,9,1},{7,8,1},{8,8,1},      // white threatens (9,8)+anything
    {4,4,0},{4,5,0},{14,14,0},{14,15,0}
  };

  int score = 0;
  score += test_fork_make (1, f1, 8);
  score += test_fork_make (2, f2, 8);
  score += test_fork_block(1, b1, 8);
  score += test_fork_block(2, b2, 8);
  printf("score=%d/4\n", score);
  printf("verdict: %s\n", (score == 4) ? "PASS" : "FAIL");
  return 0;
}
