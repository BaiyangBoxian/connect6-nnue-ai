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
// analyze_game.cpp -- parse a connect6 game record, run depth-10 at each move,
// compare actual vs best move + eval gap; mark turning points (gap > 500).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
extern "C" { void b6_new_game(int); void b6_play(int,int,int); void b6_clear(int,int); int b6_cell(int,int);
  int b6_search(int,int,int,int*); int b6_eval_value(int,int); void b6_set_net(int); void b6_set_perspective(int); }
static int col(char ch){ return ch - 'A'; }   // A=0 .. S=18
static int row(const char* s){ return atoi(s) - 1; }  // 1..19 -> 0..18
int main(int argc, char** argv){
  FILE* f = fopen(argv[1], "r");
  if(!f){ printf("no file\n"); return 1; }
  b6_new_game(0); b6_set_net(0);
  char line[256]; int mvno=0; int turn=0;
  double sumGap=0; int turns=0, pivots=0;
  std::vector<int> pivotList;
  while(fgets(line, sizeof line, f)){
    // strip
    char* p = line; while(*p==' '||*p=='\t') p++;
    if(*p==0||*p=='\n'||*p=='\r') continue;
    // parse "X12" or "X12+Y34"
    char t1[8], t2[8]; int ntok=0;
    char* plus = strchr(p, '+');
    if(plus){ *plus=0; ntok=2; snprintf(t1,sizeof t1,"%s",p); snprintf(t2,sizeof t2,"%s",plus+1); }
    else { ntok=1; snprintf(t1,sizeof t1,"%s",p); }
    // strip trailing whitespace
    for(char* q=t1; *q; q++) if(*q=='\n'||*q=='\r'||*q==' ') *q=0;
    for(char* q=t2; *q; q++) if(*q=='\n'||*q=='\r'||*q==' ') *q=0;
    int r1=row(t1+1), c1=col(t1[0]);
    int r2=-1, c2=-1;
    if(ntok==2){ r2=row(t2+1); c2=col(t2[0]); }
    int colour = turn % 2;
    // depth-10 best BEFORE playing the actual move
    b6_set_perspective(colour);
    int best[4]; b6_search(colour, 10, 1200, best);
    int bestEval = b6_eval_value(colour, 4);   // quick eval for the gap
    // play the ACTUAL move
    b6_play(r1, c1, colour);
    if(ntok==2) b6_play(r2, c2, colour);
    int actualEval = b6_eval_value(colour, 4);
    int gap = bestEval - actualEval;
    if(gap < 0) gap = -gap;
    sumGap += gap; turns++;
    mvno++;
    int isPivot = (gap > 500);
    if(isPivot){ pivots++; pivotList.push_back(mvno); }
    printf("move %2d [%c] actual=%s%d%s%s best=[%c%d,%c%d] gap=%d %s\n",
      mvno, colour? 'W':'B', t1[0],r1+1, ntok==2?"+":"", ntok==2?t2:"",
      'A'+best[0], best[1]+1, 'A'+best[2], best[3]+1, gap, isPivot?"<== PIVOT":"");
    turn++;
  }
  fclose(f);
  printf("\n--- summary ---\n");
  printf("total moves=%d  avg eval gap=%.0f  pivots(gap>500)=%d\n", turns, sumGap/turns, pivots);
  printf("pivot moves: ");
  for(int x : pivotList) printf("%d ", x);
  printf("\n");
  return 0; }
