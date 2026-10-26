// Copyright (c) 2026 白杨博贤
// main.cpp -- connect6-engine.exe: line-based stdin/stdout protocol wrapper
// around the NNUE alpha-beta engine in engine/engine_ab.cpp.
//
// Protocol (one command per line, one response per line):
//   START 19                  -> OK
//   PING                      -> OK
//   RESET                     -> OK
//   BOARD <361 chars> <side>  -> OK    cells row-major: '0' empty, '1' black, '2' white
//                                     side: which colour moves next (1 black / 2 white)
//   GO <timelimit_ms> <depth> -> = r1 c1 r2 c2   (0-based; first move returns r2 c2 = -1 -1)
//   QUIT                      -> (exit)
//
// Every GO rebuilds the engine position from the BOARD payload, so the engine is
// always in sync with the GUI (undo / restart correctness by construction).
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>

extern "C" {
    void b6_new_game(int);
    void b6_play(int, int, int);
    void b6_clear(int, int);
    int  b6_cell(int, int);
    void b6_set_perspective(int);
    int  b6_search(int, int, int, int*);
    int  b6_opening(int*);
}

static int g_board[19][19];   // 0 empty, 1 black, 2 white
static int g_side = 1;        // side to move: 1 black, 2 white

static void reply(const char* s) { std::printf("%s\n", s); std::fflush(stdout); }

// protocol colour (1 black / 2 white) -> engine colour (0 black / 1 white)
static int eng_colour(int v) { return v == 1 ? 0 : 1; }

static void apply_board() {
    // colour 0 (black) <-> SELF after b6_new_game(0); b6_play maps by colour.
    b6_new_game(0);
    for (int r = 0; r < 19; ++r)
        for (int c = 0; c < 19; ++c)
            if (g_board[r][c]) b6_play(r, c, eng_colour(g_board[r][c]));
}

static int in_board(int r, int c) { return r >= 0 && r < 19 && c >= 0 && c < 19; }

static void cmd_board(const std::string& cells, int side) {
    if (cells.size() != 361 || (side != 1 && side != 2)) { reply("ERROR bad_board"); return; }
    for (int i = 0; i < 361; ++i) {
        char ch = cells[(size_t)i];
        if (ch < '0' || ch > '2') { reply("ERROR bad_board"); return; }
        g_board[i / 19][i % 19] = ch - '0';
    }
    g_side = side;
    apply_board();
    reply("OK");
}

static void cmd_go(int ms, int depth) {
    int stones = 0;
    for (int r = 0; r < 19; ++r) for (int c = 0; c < 19; ++c) if (g_board[r][c]) ++stones;
    if (stones >= 361) { reply("= -1 -1 -1 -1"); return; }
    if (stones == 0) {
        // connect6 first move = 1 stone: use the engine's own opening point
        int o[2];
        b6_opening(o);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "= %d %d -1 -1", o[0], o[1]);
        reply(buf);
        return;
    }

    apply_board();
    int mover = eng_colour(g_side);
    b6_set_perspective(mover);          // search must play for the side to move

    int out[4] = { -1, -1, -1, -1 };
    int ok = b6_search(mover, depth > 0 ? depth : 0, ms > 0 ? ms : 100, out);

    int r1 = out[0], c1 = out[1], r2 = out[2], c2 = out[3];
    // validate: returned points must be legal and empty
    if (!ok || !in_board(r1, c1) || g_board[r1][c1] != 0) {
        if (stones == 0) { b6_opening(out); r1 = out[0]; c1 = out[1]; r2 = -1; c2 = -1; }
        else {
            r1 = c1 = -1;
            for (int r = 0; r < 19 && r1 < 0; ++r)
                for (int c = 0; c < 19; ++c)
                    if (g_board[r][c] == 0) { r1 = r; c1 = c; break; }
            r2 = c2 = -1;
        }
    }
    if (in_board(r2, c2) && (g_board[r2][c2] != 0 || (r2 == r1 && c2 == c1))) { r2 = c2 = -1; }
    if (stones == 0) { r2 = c2 = -1; }   // connect6 first move = 1 stone

    // single-stone move on a non-empty board: pick any sensible second stone
    if (!in_board(r2, c2) && stones > 0 && in_board(r1, c1)) {
        static const int DR[4] = { 0, 1, 1, 1 }, DC[4] = { 1, 0, 1, -1 };
        int bestR = -1, bestC = -1, bestScore = -1;
        for (int d = 0; d < 4; ++d) {
            for (int s = -5; s <= 5; ++s) {
                if (!s) continue;
                int rr = r1 + s * DR[d], cc = c1 + s * DC[d];
                if (!in_board(rr, cc) || g_board[rr][cc] != 0) continue;
                int score = 5 - (s > 0 ? s : -s);
                if (score > bestScore) { bestScore = score; bestR = rr; bestC = cc; }
            }
        }
        r2 = bestR; c2 = bestC;
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "= %d %d %d %d", r1, c1, r2, c2);
    reply(buf);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        char cmd[32] = { 0 };
        if (std::sscanf(line.c_str(), "%31s", cmd) != 1) { reply("ERROR empty"); continue; }

        if (!std::strcmp(cmd, "QUIT")) break;
        else if (!std::strcmp(cmd, "PING")) reply("OK");
        else if (!std::strcmp(cmd, "START")) reply("OK");
        else if (!std::strcmp(cmd, "RESET")) { b6_new_game(0); std::memset(g_board, 0, sizeof(g_board)); g_side = 1; reply("OK"); }
        else if (!std::strcmp(cmd, "BOARD")) {
            char cells[384] = { 0 };
            int side = 0;
            if (std::sscanf(line.c_str(), "%*s %383s %d", cells, &side) == 2) cmd_board(cells, side);
            else reply("ERROR bad_board");
        }
        else if (!std::strcmp(cmd, "GO")) {
            int ms = 0, depth = 0;
            if (std::sscanf(line.c_str(), "%*s %d %d", &ms, &depth) >= 1) cmd_go(ms, depth);
            else reply("ERROR bad_go");
        }
        else reply("ERROR unknown_command");
    }
    return 0;
}
