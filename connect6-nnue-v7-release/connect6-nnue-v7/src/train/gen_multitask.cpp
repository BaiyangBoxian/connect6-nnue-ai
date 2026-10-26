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
// gen_multitask.cpp -- KataGo-style MULTI-TASK training data for Connect6.
// Single scalar target (value/WDL) lets the net "shortcut"; auxiliary tasks
// (WDL, TTE, ownership, policy) force rich representations.  Positions come from
// self-play with DEEP search-value labels (the "teacher").
//
// Per-position record (binary):
//   uint8  board[361]    (0=empty 1=black 2=white, absolute)
//   float32 value        (deep search value from BLACK's perspective)
//   uint8  wdl           (2=black win 1=draw 0=white win -- final outcome)
//   float32 tte          (plies remaining until game end, normalised later)
//   uint8  pol_r, pol_c  (the move actually played here, 0xff if game over)
//   uint8  owner[361]    (final ownership: 1=black 2=white 0=neutral/empty)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
extern "C" {
  void b6_new_game(int); void b6_play(int,int,int); int b6_cell(int,int); void b6_set_perspective(int);
  int  b6_search(int,int,int,int*); int b6_eval_value(int,int);
}
static const int LABEL_DEPTH = 2;   // Step1: fast labels for volume (Step3 raises depth)

struct Pos {
    uint8_t board[361];
    float   value;
    uint8_t wdl;
    float   tte;
    uint8_t pol_r, pol_c;
    float feat[14];
    uint8_t faced_threat;   // opponent had open3/open4 at this position
    uint8_t mover;   // side to move at this position (0=black 1=white) -- WDL is from THEIR view
};
struct Game { std::vector<Pos> pos; int winner; uint8_t finalOwner[361]; };


// ===== 12 hand-crafted tactical features (root-cause fix #2) =====
// 0-4: self open2/open3/sleep3/open4/double3   5-9: opp same
// 10: whose turn (1=black to move)             11: stone count / 361
static void countThreats(const unsigned char* b, int col, int* out) {
    int o2=0,o3=0,s3=0,o4=0,d3=0;
    int dr[4]={0,1,1,1}, dc[4]={1,0,1,-1};
    for (int r=0;r<19;r++) for (int c=0;c<19;c++) {
        if (b[r*19+c]!=(unsigned char)col) continue;
        for (int d=0;d<4;d++) {
            int pr=r-dr[d], pc=c-dc[d];
            if (pr>=0&&pr<19&&pc>=0&&pc<19&&b[pr*19+pc]==(unsigned char)col) continue; // only run starts
            int n=0, rr=r, cc=c;
            while (rr>=0&&rr<19&&cc>=0&&cc<19&&b[rr*19+cc]==(unsigned char)col) { n++; rr+=dr[d]; cc+=dc[d]; }
            int r0=r-dr[d], c0=c-dc[d];
            bool openB=(r0>=0&&r0<19&&c0>=0&&c0<19&&b[r0*19+c0]==0);
            bool openA=(rr>=0&&rr<19&&cc>=0&&cc<19&&b[rr*19+cc]==0);
            int opens=openB+openA;
            // SUB-RUN counting: a run of length n counts 1 x openN + (n-1) x openN-1 + ... + (n-1) x open2
            for (int k=n; k>=2; --k) {
                int sub = n - k + 1;   // number of length-k sub-runs
                if (opens==2) { if (k==2) o2+=sub; else if (k==3) o3+=sub; else if (k>=4) o4+=sub; }
                else if (opens==1) { if (k==3) s3+=sub; else if (k==2) o2+=sub; else if (k>=4) o4+=sub; }
            }
        }
    }
    // double3: empty cells where placing `col` would create >=2 open-threes (fork point)
    for (int r=0;r<19;r++) for (int c=0;c<19;c++) {
        if (b[r*19+c]!=0) continue;
        int made=0;
        for (int d=0;d<4;d++) {
            int n=1, rr=r+dr[d], cc=c+dc[d];
            while (rr>=0&&rr<19&&cc>=0&&cc<19&&b[rr*19+cc]==(unsigned char)col) { n++; rr+=dr[d]; cc+=dc[d]; }
            int r0=r-dr[d], c0=c-dc[d];
            while (r0>=0&&r0<19&&c0>=0&&c0<19&&b[r0*19+c0]==(unsigned char)col) { n++; r0-=dr[d]; c0-=dc[d]; }
            bool openB=(r0>=0&&r0<19&&c0>=0&&c0<19&&b[r0*19+c0]==0);
            bool openA=(rr>=0&&rr<19&&cc>=0&&cc<19&&b[rr*19+cc]==0);
            if (n==3 && openB && openA) made++;
        }
        if (made>=2) d3++;
    }
    out[0]=o2; out[1]=o3; out[2]=s3; out[3]=o4; out[4]=d3;
}
static void computeFeat14(const unsigned char* b, int mover, float* f) {
    int selfCol = mover?1:2, oppCol = mover?2:1;   // mover: 0=black 1=white
    int ts[5], to[5];
    countThreats(b, selfCol, ts);
    countThreats(b, oppCol, to);
    for (int i=0;i<5;i++){ f[i]=(float)ts[i]; f[5+i]=(float)to[i]; }
    int stones=0; for (int i=0;i<361;i++) if (b[i]!=0) stones++;
    f[10]=mover?0.0f:1.0f;          // 1 = black to move
    f[11]=(float)stones/361.0f;
    // P1: 2 defense signals
    f[12] = (to[3] > 0 || to[1] >= 2) ? 1.0f : 0.0f;   // opp_threatens_win (open4 or 2+ open3)
    f[13] = (to[3] > 0) ? 1.0f : 0.0f;                  // opp_has_open4
}

static void snapshot(std::vector<Pos>& pos, int pr, int pc, int mover) {
    Pos p;
    for (int i = 0; i < 361; i++) p.board[i] = (uint8_t)b6_cell(i/19, i%19);
    float v = (float)b6_eval_value(0, LABEL_DEPTH);  // black's perspective
    // clamp search WIN scores: tanh(value/VSCALE=200000) saturates otherwise
    if (v >  800000.f) v =  800000.f;
    if (v < -800000.f) v = -800000.f;
    p.value = v;
    p.wdl = 0; p.tte = 0; p.pol_r = (uint8_t)pr; p.pol_c = (uint8_t)pc; p.mover = (uint8_t)mover;
    computeFeat14(p.board, mover, p.feat);
    // sample-weighting: did the OPPONENT have an open3/open4 threat here? (feat[5..9]=SELF, feat[0..4]=OPP in data conv)
    p.faced_threat = (p.feat[1] > 0 || p.feat[3] > 0) ? 1 : 0;   // opp open3 or open4
    pos.push_back(p);
}
static bool has6(int col) {
    int dr[4]={0,1,1,1}, dc[4]={1,0,1,-1};
    for (int r=0;r<19;r++) for (int c=0;c<19;c++) if (b6_cell(r,c)==col)
        for (int d=0;d<4;d++){ int n=0; for(int i=0;i<6;i++){int rr=r+i*dr[d],cc=c+i*dc[d]; if(rr<0||rr>18||cc<0||cc>18||b6_cell(rr,cc)!=col)break; n++;} if(n==6)return true; }
    return false;
}
// simple ownership at game end: nearest-stone influence (teaches spatial control)
static void computeOwnership(uint8_t owner[361]) {
    for (int i=0;i<361;i++) {
        int r=i/19, c=i%19; int bestB=999, bestW=999;
        for (int rr=0;rr<19;rr++) for (int cc=0;cc<19;cc++) {
            int v=b6_cell(rr,cc); if(!v) continue;
            int d=abs(rr-r)+abs(cc-c);
            if (v==1 && d<bestB) bestB=d;
            if (v==2 && d<bestW) bestW=d;
        }
        owner[i] = (bestB<bestW)?1:((bestW<bestB)?2:0);
    }
}
int main(int argc, char** argv) {
    int nGames=(argc>1)?atoi(argv[1]):60;
    const char* outpath=(argc>2)?argv[2]:"engine_data/mt_1.bin";
    int seed=(argc>3)?atoi(argv[3]):12345;
    srand(seed);
    FILE* f=fopen(outpath,"wb"); if(!f){fprintf(stderr,"no write %s\n",outpath);return 1;}
    int32_t total=0; fwrite(&total,4,1,f);   // placeholder count
    for (int g=0; g<nGames; g++) {
        int engine_white=g&1; b6_new_game(engine_white);
        int extra=rand()%5;
        for (int k=0;k<extra;k++){ int r,c; do{r=rand()%19;c=rand()%19;}while(b6_cell(r,c)); b6_play(r,c,(k&1)?engine_white:(engine_white^1)); }
        std::vector<Pos> pos; int mover=rand()%2;  /* SWAP rule: random first player (balances first-move advantage) */ int plies=0; bool over=false; int winCode=0;
        while(!over && plies<70) {
            int colour=mover?1:0; int o[4]={-1,-1,-1,-1};
            b6_set_perspective(colour);   // search plays for the side to move
            b6_search(colour,6,600,o);   // teaching depth 6 (see block->open4->lose chain)
            // GUARD (match_bal pattern): b6_search can return its dummy 9,9,9,10
            // fallback or leave out untouched on failure; never index the board with
            // out-of-range coords (s12 gens died mid-run at 22:15 leaving unpatched
            // headers + ragged tails -- suspect OOB board access here).
            if(o[0]<0||o[0]>18||o[1]<0||o[1]>18||o[2]<0||o[2]>18||o[3]<0||o[3]>18||
               b6_cell(o[0],o[1])||b6_cell(o[2],o[3])) break;
            snapshot(pos, o[0], o[1], mover);   // snapshot BEFORE the move: policy on empty cell
            if (plies == 0) { b6_play(o[0],o[1],colour); }        // RULE: first move = 1 stone
            else { b6_play(o[0],o[1],colour); b6_play(o[2],o[3],colour); }   // then 2/turn
            plies++;
            // BUGFIX: has6 must compare BOARD CODES (b6_cell returns 1=black/2=white),
            // not the 0/1 side flag -- the old has6(0) matched runs of EMPTY cells and
            // ended every game at ply 2 with a constant WDL label (shortcut learning).
            winCode = b6_cell(o[0],o[1]);   // code of the mover's just-placed stones
            if (has6(winCode)) { over=true; }
            mover^=1;
        }
        // fill WDL + TTE + ownership from the finished game
        // encoding matches train_multitask.py: 2=black win, 1=draw, 0=white win
        int winner = 1;
        if (over) {
            winner = (winCode==1) ? 2 : 0;
        } else if (!pos.empty()) {
            // Truncated game (70-ply cap / invalid search move): true outcome unknown.
            // Labeling every such game "draw" made WDL labels constant -> shortcut
            // learning (wdl cross-entropy collapsed to exactly 0). Use the teacher's
            // final deep eval as outcome proxy instead; deadband keeps genuinely
            // balanced games labeled draw.
            float lv = pos.back().value;
            winner = (lv >  100000.f) ? 2 : (lv < -100000.f) ? 0 : 1;
        }
        uint8_t owner[361]; computeOwnership(owner);
        int n=(int)pos.size();
        for (int i=0;i<n;i++) {
            // WDL from the SIDE-TO-MOVE's perspective (Step 1.1): 2=that side wins, 1=draw, 0=loss
            int wdl;
            if (winner == 1) wdl = 1;                       // draw
            else {
                int winner_mover = (winner == 2) ? 0 : 1;    // winner==2 -> black(mover 0) won; ==0 -> white(mover 1) won
                wdl = (pos[i].mover == winner_mover) ? 2 : 0;
            }
            pos[i].wdl = (uint8_t)wdl;
            pos[i].tte = (float)(n - i);
            fwrite(pos[i].board,1,361,f);
            fwrite(&pos[i].value,4,1,f);
            fwrite(&pos[i].wdl,1,1,f);
            fwrite(&pos[i].tte,4,1,f);
            fwrite(&pos[i].pol_r,1,1,f); fwrite(&pos[i].pol_c,1,1,f); fwrite(&pos[i].mover,1,1,f);
            fwrite(pos[i].feat,sizeof(float),14,f);   // 14 hand-crafted features
            fwrite(owner,1,361,f);
            // SAMPLE WEIGHT x3: "didn't block -> lost" (faced threat + the mover lost)
            if (pos[i].faced_threat && ((pos[i].mover == 0 && winner == 0) || (pos[i].mover == 1 && winner == 2))) {
                for (int rep = 0; rep < 2; ++rep) {   // 2 extra copies = x3 total
                    fwrite(pos[i].board,1,361,f);
                    fwrite(&pos[i].value,4,1,f); fwrite(&pos[i].wdl,1,1,f); fwrite(&pos[i].tte,4,1,f);
                    fwrite(&pos[i].pol_r,1,1,f); fwrite(&pos[i].pol_c,1,1,f); fwrite(&pos[i].mover,1,1,f);
                    fwrite(pos[i].feat,sizeof(float),14,f); fwrite(owner,1,361,f);
                    total++;
                }
            }
            // P0.2: oversample 2x when the opponent has open3/open4 (defense-critical positions)
            if (pos[i].feat[6] > 0 || pos[i].feat[8] > 0) {
                fwrite(pos[i].board,1,361,f);
                fwrite(&pos[i].value,4,1,f); fwrite(&pos[i].wdl,1,1,f); fwrite(&pos[i].tte,4,1,f);
                fwrite(&pos[i].pol_r,1,1,f); fwrite(&pos[i].pol_c,1,1,f); fwrite(&pos[i].mover,1,1,f);
                fwrite(pos[i].feat,sizeof(float),14,f);
                fwrite(owner,1,361,f);
                total++;
            }
            total++;
        }
        if((g+1)%10==0) fprintf(stderr,"games %d samples %d\n",g+1,total);
    }
    fseek(f,0,SEEK_SET); fwrite(&total,4,1,f); fclose(f);
    fprintf(stderr,"wrote %d samples to %s (%.1f MB)\n",total,outpath,total*733/1e6);
    return 0;
}
