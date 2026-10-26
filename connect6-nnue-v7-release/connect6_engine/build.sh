#!/usr/bin/env bash
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
