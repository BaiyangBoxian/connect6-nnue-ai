// Copyright (c) 2026 白杨博贤
#include <cstdio>
#include <cstdlib>
extern "C" { void b6_new_game(int); void b6_play(int,int,int); int b6_cell(int,int);
  int b6_search(int,int,int,int*); void b6_set_net(int); void b6_set_perspective(int); }
int main(){
  b6_new_game(0); b6_set_net(0);
  int stones_per_move[10]; int nm=0;
  for(int ply=0; ply<10; ply++){
    int colour=ply%2;
    b6_set_perspective(colour);
    int o[4]={-1,-1,-1,-1};
    b6_search(colour,5,300,o);
    int placed=0;
    if(o[0]>=0 && b6_cell(o[0],o[1])==0){ b6_play(o[0],o[1],colour); placed++; }
    if(o[2]>=0 && b6_cell(o[2],o[3])==0){ b6_play(o[2],o[3],colour); placed++; }
    stones_per_move[nm++]=placed;
    printf("move %d: %d stone(s) [%d,%d]%s[%d,%d]\n", ply+1, placed, o[0],o[1], placed>1?"+":" ", o[2],o[3]);
    if(placed==0) break;
  }
  printf("format: ");
  for(int i=0;i<nm;i++) printf("%d",stones_per_move[i]);
  printf("\n");
  int ok=(nm>=5 && stones_per_move[0]==1 && stones_per_move[1]==2 && stones_per_move[2]==2 && stones_per_move[3]==2 && stones_per_move[4]==2);
  printf("RULE 1-2-2-2-2: %s\n", ok?"PASS ✅":"FAIL");
  return 0; }
