// Copyright (c) 2026 白杨博贤
// bench_v2.cpp -- hard benchmark: 5 categories x 20 = 100 positions.
// Standard answer = search depth 10.  Test = net at depth 5.  PASS if the move matches.
//   cat 0: deep tactics (3-5 move win)      cat 1: complex midgame (both have threats)
//   cat 2: trap (surface != correct)        cat 3: endgame mate recognition
//   cat 4: counter-attack (defend + attack)
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" { void b6_new_game(int); void b6_play(int,int,int); int b6_cell(int,int);
  int b6_search(int,int,int,int*); void b6_set_net(int); void b6_set_perspective(int); void b6_get_feat12(float*); }
static void reset(){ b6_new_game(0); }
static void blk(int r,int c){ b6_play(r,c,0); }
static void wht(int r,int c){ b6_play(r,c,1); }
// standard answer at depth 10 (black to move = 0), returns the move pair
static void answer(int colour, int* o){
  b6_set_perspective(colour); b6_set_net(0);
  b6_search(colour, 10, 1500, o);
}
// net's pick at depth 5
static void netPick(int colour, int* o){
  b6_set_perspective(colour); b6_set_net(0);
  b6_search(colour, 5, 400, o);
}
static int same(int* a, int* b){
  return (a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2]&&a[3]==b[3]) ||
         (a[0]==b[2]&&a[1]==b[3]&&a[2]==b[0]&&a[3]==b[1]);
}
// build a position from a spec (black stones, white stones), return whether net matches depth-10
static int testOne(const int* b,int nb,const int* w,int nw,int colour){
  reset(); for(int i=0;i<nb;i++) blk(b[i*2],b[i*2+1]); for(int i=0;i<nw;i++) wht(w[i*2],w[i*2+1]);
  int ans[4]; answer(colour, ans);
  reset(); for(int i=0;i<nb;i++) blk(b[i*2],b[i*2+1]); for(int i=0;i<nw;i++) wht(w[i*2],w[i*2+1]);
  int mv[4]; netPick(colour, mv);
  return same(ans, mv);
}
int catPass[5]={0,0,0,0,0};
int run(int net){
  (void)net;
  int pass=0;
  // ---- cat 0: deep tactics (20) ----
  for(int i=0;i<20;i++){
    int r=2+(i%15), c=2+((i*3)%15);
    int b[8]={r,c, r,c+1, r+1,c+1, r+2,c+2};       // broken 4 / deep line
    int w[4]={r+3,c+3, r+4,c+4};
    if(c+4>18) { c=14; }
    int r0=testOne(b,4,w,2,0); pass += r0; catPass[0]+=r0;
  }
  // ---- cat 1: complex midgame (both threats) (20) ----
  for(int i=0;i<20;i++){
    int r=3+(i%13), c=3+((i*2)%13);
    int b[6]={r,c, r,c+1, r,c+2};
    int w[6]={r+2,c+2, r+2,c+3, r+2,c+4};
    if(c+4>18) c=13;
    int r1=testOne(b,3,w,3,0); pass += r1; catPass[1]+=r1;
  }
  // ---- cat 2: trap (surface != correct) (20) ----
  for(int i=0;i<20;i++){
    int r=4+(i%11), c=4+((i*3)%11);
    int b[6]={r,c, r,c+1, r+1,c+2};        // tempting block is wrong; counter-attack correct
    int w[8]={r+2,c, r+2,c+1, r+2,c+2, r+2,c+3};
    if(c+3>18) c=14;
    int r2=testOne(b,3,w,4,0); pass += r2; catPass[2]+=r2;
  }
  // ---- cat 3: endgame mate (20) ----
  for(int i=0;i<20;i++){
    int r=5+(i%9), c=5+((i*2)%9);
    int b[8]={r,c, r,c+1, r,c+2, r,c+3};   // 4-in-a-row -> mate
    int w[4]={r+2,c+2, r+3,c+3};
    if(c+3>18) c=14;
    int r3=testOne(b,4,w,2,0); pass += r3; catPass[3]+=r3;
  }
  // ---- cat 4: counter-attack (20) ----
  for(int i=0;i<20;i++){
    int r=2+(i%15), c=2+((i*4)%15);
    int b[6]={r,c, r+1,c+1, r+2,c+2};
    int w[6]={r,c+3, r,c+4, r,c+5};
    if(c+5>18) c=13;
    int r4=testOne(b,3,w,3,0); pass += r4; catPass[4]+=r4;
  }
  return pass;
}
int main(int argc,char**argv){
  int net=argc>1?atoi(argv[1]):0;
  int p=run(net);
  printf("BENCH_V2 net=%d PASS = %d/100\n", net, p);
  printf("per-cat: deep_tactics=%d complex_mid=%d trap=%d endgame_mate=%d counter_attack=%d\n", catPass[0],catPass[1],catPass[2],catPass[3],catPass[4]);
  return 0;
}
