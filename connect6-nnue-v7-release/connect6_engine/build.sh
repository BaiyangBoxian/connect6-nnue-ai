#!/usr/bin/env bash
# connect6-nnue-ai
# Neural network + alpha-beta search board game AI, trained with self-play
#
# BetaSix: 面向十九路六子棋的多任务 NNUE 高效评估与 α-β 搜索
# Copyright (C) 2026  白杨博贤
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Copyright (c) 2026 白杨博贤
# build.sh -- build the connect6 NNUE engine wrapper
#   ./build.sh          -> Linux binary (connect6-engine-linux) + Windows exe if mingw exists
set -e
cd "$(dirname "$0")"

g++ -O2 -std=c++14 -msse4.1 -DNDEBUG -o connect6-engine-linux engine/engine_ab.cpp main.cpp
echo "built: connect6-engine-linux"

if command -v x86_64-w64-mingw32-g++ >/dev/null 2>&1; then
    x86_64-w64-mingw32-g++ -O2 -std=c++14 -msse4.1 -DNDEBUG -static \
        -o connect6-engine-win-sse4.1.exe engine/engine_ab.cpp main.cpp
    echo "built: connect6-engine-win-sse4.1.exe"
else
    echo "x86_64-w64-mingw32-g++ not found -- Windows exe skipped"
fi
