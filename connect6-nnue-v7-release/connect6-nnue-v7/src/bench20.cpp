// Copyright (c) 2026 白杨博贤
// bench20.cpp -- 20-position benchmark: 5 block-3 + 5 double-3 + 5 open-4 + 5 mid-game.
// PASS criteria per type: block-3/open-4 = the move blocks an extension point;
// double-3 = the move creates 2+ open-threes; mid-game = the move creates >=1 open-three.
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" { void b6_new_game(int); void b6_play(int,int,int); int b6_cell(int,int);
  int b6_search(int,int,int,int*); void b6_set_net(int); void b6_set_perspective(int); void b6_get_feat12(float*); }
static float SELF_O3(){ float f[14]; b6_get_feat12(f); return f[6]; } // data-conv: f[5..9]=SELF
static void reset(){ b6_new_game(0); }
static void blk(int r,int c){ b6_play(r,c,0); }   // black (side flag 0)
static void wht(int r,int c){ b6_play(r,c,1); }   // white (side flag 1)
// block-3: black 3-in-a-row; white to move. PASS if the move is at a run end (blocks extension).
static int testBlock3(const int* b3, int n3, int net){
  reset(); for(int i=0;i<n3;i++) blk(b3[i*2],b3[i*2+1]); wht(0,0); wht(0,1);
  b6_set_net(net); b6_set_perspective(1); int o[4]; b6_search(1,DEPTH,400,o);
  // run ends
  int r0=b3[0],c0=b3[1],r1=b3[(n3-1)*2],c1=b3[(n3-1)*2+1];
  int dr=(r1>r0)-(r1<r0), dc=(c1>c0)-(c1<c0);
  int e1r=r0-dr,e1c=c0-dc, e2r=r1+dr,e2c=c1+dc;
  bool hit=false;
  for(int k=0;k<2;k++){ int rr=k?o[0]:o[2], cc=k?o[1]:o[3];
    if((rr==e1r&&cc==e1c)||(rr==e2r&&cc==e2c)) hit=true; }
  return hit?1:0;
}
// double-3: black has 2+2 stones; black to move. PASS if the move creates 2+ open-threes.
static int testDouble3(const int* st, int nst, int net){
  reset(); for(int i=0;i<nst;i++) blk(st[i*2],st[i*2+1]); wht(0,0); wht(0,1);
  b6_set_net(net); b6_set_perspective(0); int o[4]; b6_search(0,DEPTH,400,o);
  b6_play(o[0],o[1],0); b6_play(o[2],o[3],0);
  return (SELF_O3()>=2)?1:0;
}
// open-4: black 4-in-a-row (open); white to move. PASS if the move blocks a run end.
static int testOpen4(const int* b4, int net){
  reset(); for(int i=0;i<4;i++) blk(b4[i*2],b4[i*2+1]); wht(0,0); wht(0,1);
  b6_set_net(net); b6_set_perspective(1); int o[4]; b6_search(1,DEPTH,400,o);
  int r0=b4[0],c0=b4[1],r1=b4[6],c1=b4[7];
  int dr=(r1>r0)-(r1<r0), dc=(c1>c0)-(c1<c0);
  int e1r=r0-dr,e1c=c0-dc, e2r=r1+dr,e2c=c1+dc;
  bool hit=false;
  for(int k=0;k<2;k++){ int rr=k?o[0]:o[2], cc=k?o[1]:o[3];
    if((rr==e1r&&cc==e1c)||(rr==e2r&&cc==e2c)) hit=true; }
  return hit?1:0;
}
// mid-game: mixed; black to move. PASS if the move creates >=1 open-three.
static int testMid(const int* b,int nb,const int* w,int nw,int net){
  reset(); for(int i=0;i<nb;i++) blk(b[i*2],b[i*2+1]); for(int i=0;i<nw;i++) wht(w[i*2],w[i*2+1]);
  b6_set_net(net); b6_set_perspective(0); int o[4]; b6_search(0,DEPTH,400,o);
  b6_play(o[0],o[1],0); b6_play(o[2],o[3],0);
  return (SELF_O3()>=1)?1:0;
}
int DEPTH=5;
int run(int net){
  int pass=0;
  // 5 block-3
  {int a[]={9,8,9,9,9,10}; pass+=testBlock3(a,3,net);}            // horiz
  {int a[]={5,5,6,6,7,7}; pass+=testBlock3(a,3,net);}             // diag
  {int a[]={2,10,3,10,4,10}; pass+=testBlock3(a,3,net);}          // vert
  {int a[]={9,3,9,4,9,5}; pass+=testBlock3(a,3,net);}             // horiz2
  {int a[]={15,15,16,16,17,17}; pass+=testBlock3(a,3,net);}       // diag2
  // 5 double-3
  {int s[]={9,9,9,10,7,9,8,9}; pass+=testDouble3(s,4,net);}
  {int s[]={4,4,4,5,2,4,3,4}; pass+=testDouble3(s,4,net);}
  {int s[]={12,12,12,13,10,12,11,12}; pass+=testDouble3(s,4,net);}
  {int s[]={6,14,6,15,4,14,5,14}; pass+=testDouble3(s,4,net);}
  {int s[]={14,3,14,4,12,3,13,3}; pass+=testDouble3(s,4,net);}
  // 5 open-4
  {int a[]={9,7,9,8,9,9,9,10}; pass+=testOpen4(a,net);}
  {int a[]={3,3,4,4,5,5,6,6}; pass+=testOpen4(a,net);}
  {int a[]={7,12,8,12,9,12,10,12}; pass+=testOpen4(a,net);}
  {int a[]={11,2,11,3,11,4,11,5}; pass+=testOpen4(a,net);}
  {int a[]={5,16,6,16,7,16,8,16}; pass+=testOpen4(a,net);}
  // 5 mid-game
  {int b[]={8,8,8,9,10,10}, w[]={9,9,9,10,7,7,7,8}; pass+=testMid(b,3,w,4,net);}
  {int b[]={4,7,5,7,5,8}, w[]={4,8,6,7,6,8,3,6}; pass+=testMid(b,3,w,4,net);}
  {int b[]={13,13,13,14,15,15}, w[]={14,14,14,15,12,12,12,13}; pass+=testMid(b,3,w,4,net);}
  {int b[]={6,3,7,3,7,4}, w[]={6,4,8,3,8,4,5,2}; pass+=testMid(b,3,w,4,net);}
  {int b[]={10,10,11,10,11,11}, w[]={10,11,12,10,12,11,9,9}; pass+=testMid(b,3,w,4,net);}
  return pass;
}
int main(int argc,char**argv){
  int net = argc>1?atoi(argv[1]):0; DEPTH = argc>2?atoi(argv[2]):5;
  int p = run(net);
  printf("net=%d depth=%d PASS = %d/20\n", net, DEPTH, p);
  return 0;
}
