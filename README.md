# connect6-nnue-ai
Neural network + alpha-beta search board game Al, trained with self-play

BetaSix: 面向十九路六子棋的多任务 NNUE 高效评估与 α-β 搜索

BetaSix Multi-task NNUE Evaluation and Efficient Alpha-Beta Search for 19×19 Connect6

Copyright (C) 2026  白杨博贤

As far as we know, BetaSix is the first fully open-source (including training weights) NNUE six-in-a-row game-playing engine. Note: The KataGomo project by hzyhhzy implemented NNUE evaluation code in its ConnectSix_NNUE_VCF branch (without releasing weights) for VCF winning search.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

## 权重许可 Weights License

本仓库源代码以 GNU GPL v3 发布。

神经网络权重文件（*.nnue / *.pth，含 weights_v7）不适用 GPL v3，
另受以下条款约束：
1. 个人学习、研究与非商业用途：自由使用与再分发；
2. 任何商业用途（包括但不限于：嵌入付费软件、用于对弈平台收费服务、
   商业棋具）需获得版权人 白杨博贤 的书面授权；
3. 使用本权重即视为同意上述条款。

The neural network weights in this repository are NOT licensed under GPL v3.
They may be used and redistributed freely for personal, academic and
non-commercial purposes only. Commercial use requires written permission
from the copyright holder (Boxian Baiyang).

中国主站：Gitee(https://gitee.com/BaiyangBoxian/connect6-nnue-ai)
