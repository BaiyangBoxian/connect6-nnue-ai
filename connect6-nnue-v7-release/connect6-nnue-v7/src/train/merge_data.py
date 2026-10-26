#!/usr/bin/env python3
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
# merge_data.py -- merge multiple NNUE .bin record files into one.
import struct, sys, glob
outs = sys.argv[1]
args = sys.argv[2:]
rec = None  # auto-detect per file (old format 369B, multitask format 733B)
if len(args) >= 2 and args[0] == '--rec':
    rec = int(args[1]); args = args[2:]   # e.g. --rec 790 for unpatched-header salvage
ins = []
for a in args:
    ins += sorted(glob.glob(a))
chunks = []
total = 0
for path in ins:
    with open(path, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        rest = f.read()
    if n > 0:
        r = len(rest) // n
        if r == 0 or (rec is not None and r != rec):
            print(f"  ! {path}: rec size {r} != {rec}, skipped")
            continue
        if rec is None:
            rec = r
        if len(rest) > n * rec:
            print(f"  ~ {path}: ragged tail ({len(rest) - n * rec}B) dropped")
    else:
        # header unpatched (generator killed before finalize) -> salvage complete
        # records instead of silently dropping the whole file (s12 round bug)
        if rec is None:
            print(f"  ! {path}: unpatched header and rec size unknown, skipped (use --rec N)")
            continue
        n = len(rest) // rec
        print(f"  ~ {path}: unpatched header -> salvaged {n} samples (partial tail dropped)")
    if n <= 0:
        print(f"  ! {path}: no complete records, skipped")
        continue
    data = rest[:n * rec]
    chunks.append(data); total += n
    print(f"  + {path}: {n} samples")
with open(outs, 'wb') as f:
    f.write(struct.pack('<i', total))
    for c in chunks: f.write(c)
print(f"merged -> {outs}: {total} samples")
