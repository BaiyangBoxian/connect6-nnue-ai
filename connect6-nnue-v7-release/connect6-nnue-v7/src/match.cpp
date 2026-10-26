// Copyright (c) 2026 白杨博贤
// match.cpp -- new net (Run 8) vs old net (Run 1), 100 games (50 each as first player).
#include <cstdio>
#include <cstring>
#include <cstdlib>
extern "C" {
  void b6_new_game(int); void b6_play(int,int,int); int b6_cell(int,int);
  int  b6_search(int,int,int,int*);
  void b6_set_net(int); void b6_nnue_recompute();
}
static bool has6(int col) {
  int dr[4]={0,1,1,1}, dc[4]={1,0,1,-1};
  for (int r=0;r<19;r++) for (int c=0;c<19;c++) if (b6_cell(r,c)==col)
    for (int d=0;d<4;d++){ int n=0; for(int i=0;i<6;i++){int rr=r+i*dr[d],cc=c+i*dc[d];
      if(rr<0||rr>18||cc<0||cc>18||b6_cell(rr,cc)!=col)break; n++;} if(n==6)return true; }
  return false;
}
int main(int argc, char** argv) {
  int NGAMES = (argc>1)?atoi(argv[1]):100;
  int TP = (argc>2)?atoi(argv[2]):300;   // time per move (ms) for a quick match
  int aW=0,bW=0,dr=0;
  for (int g=0; g<NGAMES; g++) {
    // A (new) is first player in games 0..NGAMES/2-1, B (old) first in the rest
    int mode = (argc>3)?atoi(argv[3]):0;   // 0=balanced, 1=A always first, 2=A always second
    bool aFirst = (mode==1) ? true : (mode==2) ? false : (g < NGAMES/2);
    b6_new_game(aFirst ? 0 : 1);
    int mover = 0; // 0 = black(=first player)
    bool over = false; int winner = 0; // 1=A, 2=B
    for (int ply=0; ply<180 && !over; ply++) {
      int isA = aFirst ? (mover==0) : (mover==1);
      b6_set_net(isA ? 0 : 1);           // A=new(0), B=old(1)
      int o[4]; int colour = mover?1:0;
      b6_search(colour, 3, TP, o);
      if (b6_cell(o[0],o[1]) || b6_cell(o[2],o[3])) { break; }
      b6_play(o[0],o[1],colour); b6_play(o[2],o[3],colour);
      if (has6(colour)) { over=true; winner = isA ? 1 : 2; }
      mover ^= 1;
    }
    if (!over) { dr++; }
    else if (winner==1) aW++; else bW++;
    if ((g+1)%10==0) fprintf(stderr, "[%d/%d] A(new)=%d B(old)=%d draw=%d\n", g+1,NGAMES,aW,bW,dr);
  }
  printf("RESULT A(new,Run8)=%d B(old,Run1)=%d draw=%d (of %d)\n", aW,bW,dr,NGAMES);
  return 0;
}
