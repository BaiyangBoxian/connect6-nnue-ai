# Connect6 NNUE Engine v7

六子棋（Connect6，19×19）AI 引擎：多任务 NNUE（KataGo式5头）+ α-β 搜索。

## 规则
- 19×19 棋盘，黑先，**第一手 1 子、之后每手 2 子**（1-2-2-2）
- 六子连线（横/竖/斜）即胜

## 架构（固定）
```
输入 736 维 = 722 棋盘(19×19×2) + 14 手工特征
4 层 MLP: 736 → 256 → 64 → 1   (v7 = weights_v7.npz)
5 个头: value(tanh) / WDL(3) / TTE(1) / ownership(361×3) / policy(361)
搜索: α-β + 迭代加深 + 静态搜索 + 杀手启发 + 历史启发 + 空着裁剪
```

## 14 维特征
```
f[0..4]  OPP 威胁子run: open2/open3/sleep3/open4/double3
f[5..9]  SELF 威胁子run
f[10]    轮到谁走 (1=黑)
f[11]    落子数/361
f[12]    opp_threatens_win
f[13]    opp_has_open4
```

## 目录
```
src/engine/   engine_ab.cpp + nnue_A.h + nnue_B.h  (α-β + NNUE)
src/train/    gen_multitask.cpp(数据) + train_multitask.py(训练) + merge_data.py
src/          verify_rule/fork/match, bench_v2, aivai10, analyze_game
src/scripts/  gen1m.sh(生成) pipeline_1m.sh(管线) retrain*.sh(训练)
models/       weights_v7.npz(最终) + benchmark_v2.json(考卷) + progress_log.txt
game_records/ AI自对弈 + 大师棋谱
web/          pack_A/B/C (HTML + asm.js 引擎)
docs/         README / BUGS_FOUND / TRAINING_LOG
```

## 怎么跑
```bash
# 编译引擎
g++ -O2 -std=c++14 -o verify_rule engine_src/engine_ab.cpp verify_rule.cpp

# 打包网页版 (asm.js, 无wasm)
emcc -O2 -std=c++14 -s WASM=0 -s EXPORTED_FUNCTIONS='["_b6_new_game","_b6_play","_b6_clear","_b6_cell","_b6_search","_b6_set_net","_b6_set_perspective","_b6_eval_value","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue"]' -s ALLOW_MEMORY_GROWTH=1 \
  --memory-init-file 1 --preload-file weights_v7.npz -o pack_A/betasix_v2.js engine_src/engine_ab.cpp

# 启动网页版
cd pack_A && python3 -m http.server 8003   # → http://localhost:8003

# benchmark v2 (100 面考卷, depth10 标准答案)
g++ -O2 -std=c++14 -o bench_v2 engine_src/engine_ab.cpp bench_v2.cpp && ./bench_v2 0
```

## benchmark v2 = 90/100 (weights_v7)
```
深层战术 14 | 复杂中盘 19 | 诱导 18 | 终盘必杀 20 | 防守反击 19
```
