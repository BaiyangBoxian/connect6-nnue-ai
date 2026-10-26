# TRAINING_LOG.md — 从 0 到 110 万样本

## 阶段 0：基线
- 引擎：α-β + 迭代加深 + 静态搜索 + 杀手/历史启发 + 空着裁剪
- NNUE：736 → 256 → 64 → 1（value 主评估），55% eval 权重
- 旧有监督训练（bug 时代 160k）：value 81.6% 但**策略头没学到**（多分类错配 bug），实际 0% 抓 fork

## 阶段 1：多任务（KataGo式5头）
- 头：value(tanh)/WDL/TTE/ownership/policy
- 数据：`gen_multitask.cpp` 限深6 自对弈（`b6_search(colour,6,600,o)`）
- 样本格式 790 字节：board(361) + value(1) + wdl(1) + tte(1) + polr+polc(2) + mover(1) + 14feat + owner(361)
- 训练：`train_multitask.py`，value MSE + wdl KL + tte MSE + policy CE + owner CE

## 关键修复迭代
1. iterBest bug → 多任务 net 从 0% 到 60% 抓 fork（archive dataset）
2. eval 非对称 + WDL 方向 → 自对弈平衡
3. 规则修正（首手1子）→ 旧数据作废，重生成
4. 特征编码（f[12]/f[13] OPP vs SELF）→ 防守修复，benchmark 85→90

## 数据规模
```
d5000 (作废)   510,788
d750k (作废)   769,123
r5000 (正确)   292,990
d1m (正确)     425,847
合并           1,101,585 (110万)
```

## benchmark v2 演进
```
weights_v5 (67万)  84/100
weights_v6 (110万) 85/100
weights_v7 (110万+编码修复) 90/100  ← 最终
宽网 512/256/128   66/100  (放弃)
```

## 阶段 2 尝试（宽网）
- 736 → 512 → 256 → 128 → 1（4层MLP，5头不变）
- 结果 benchmark 66/100（深层战术 4/20）< 90 → **放弃，回滚 v7**

## 自对弈验证
- AI vs AI 10 局：v6 胜 6 / v5 胜 4
- 防守反击类 benchmark 19/20（编码修复后）

## 最终
- **weights_v7.npz**：v2 = 90/100，110 万样本，编码修复版
- 生成管线继续，目标 150 万
