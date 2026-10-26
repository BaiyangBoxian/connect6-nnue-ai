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
// aivai10.cpp -- 10 games: new net (v6, black) vs old net (v5, white). Record stats + game records.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
extern "C" { void b6_new_game(int); void b6_play(int,int,int); int b6_cell(int,int);
  int b6_search(int,int,int,int*); void b6_set_net(int); void b6_set_perspective(int); }
static bool has6(int col){ int dr[4]={0,1,1,1},dc[4]={1,0,1,-1};
  for(int r=0;r<19;r++)for(int c=0;c<19;c++) if(b6_cell(r,c)==col)
    for(int d=0;d<4;d++){int n=0;for(int i=0;i<6;i++){int rr=r+i*dr[d],cc=c+i*dc[d];
      if(rr<0||rr>18||cc<0||cc>18||b6_cell(rr,cc)!=col)break;n++;} if(n==6)return true;} return false; }
static char L(int c){ return 'A'+c; }
int main(){
  int nW=0,oW=0,dr=0; std::vector<int> nWinLen,oWinLen,allLen;
  for(int g=0; g<10; g++){
    b6_new_game(0);
    std::vector<int> rec;  // r,c,color triples
    int mover=0; bool over=false; int winner=0; // 1=new(black), 2=old(white)
    int ply=0;
    for(ply=0; ply<200 && !over; ply++){
      int colour = mover%2;
      b6_set_net(colour==0?0:1);          // black=new(v6), white=old(v5)
      b6_set_perspective(colour);
      int o[4]={-1,-1,-1,-1};
      b6_search(colour, 5, 300, o);
      int need = (rec.empty()) ? 1 : 2;   // rule: first move 1 stone
      int placed=0;
      for(int k=0;k<2 && placed<need;k++){
        int r=(k==0)?o[0]:o[2], c=(k==0)?o[1]:o[3];
        if(r>=0&&r<19&&c>=0&&c<19&&b6_cell(r,c)==0){ b6_play(r,c,colour); rec.push_back(r); rec.push_back(c); rec.push_back(colour); placed++; }
      }
      if(placed==0) break;
      if(has6(colour+1)){ over=true; winner = (colour==0)?1:2; }
      mover++;
    }
    int stones = (int)rec.size()/3;
    allLen.push_back(stones);
    if(!over) dr++;
    else if(winner==1){ nW++; nWinLen.push_back(stones); }
    else { oW++; oWinLen.push_back(stones); }
    // write the game record (67-move format)
    char fn[128]; snprintf(fn,sizeof fn,"game_records/ai_vs_ai/game_%02d.txt",g+1);
    FILE* f=fopen(fn,"w");
    // group by turn: first = 1 stone, then 2 per turn
    for(int i=0, first=1; i<(int)rec.size(); ){
      int col=rec[i+2];
      if(first){ fprintf(f,"%c%d\n", L(rec[i+1]), rec[i]+1); i+=3; first=0; continue; }
      fprintf(f,"%c%d", L(rec[i+1]), rec[i]+1); i+=3;
      if(i<(int)rec.size() && rec[i+2]==col){ fprintf(f,"+%c%d", L(rec[i+1]), rec[i]+1); i+=3; }
      fprintf(f,"\n");
    }
    fclose(f);
    printf("game %2d: %s stones=%d\n", g+1, !over?"draw":(winner==1?"NEW(v6)":"OLD(v5)"), stones);
  }
  printf("\n--- SUMMARY ---\n");
  printf("new=%d old=%d draw=%d\n", nW,oW,dr);
  int sum=0; for(int x:allLen) sum+=x;
  printf("avg moves=%.0f\n", allLen.empty()?0:(double)sum/allLen.size());
  if(!nWinLen.empty()){
    int mn=nWinLen[0],mx=nWinLen[0]; for(int x:nWinLen){ if(x<mn)mn=x; if(x>mx)mx=x; }
    printf("NEW win: fastest=%d slowest=%d stones\n", mn, mx);
  }
  if(!oWinLen.empty()){
    int mn=oWinLen[0],mx=oWinLen[0]; for(int x:oWinLen){ if(x<mn)mn=x; if(x>mx)mx=x; }
    printf("OLD win: fastest=%d slowest=%d stones\n", mn, mx);
  }
  return 0; }
