# Copyright (c) 2026 白杨博贤
import tkinter as tk
from tkinter import messagebox
import random
import time
import subprocess
import os
import sys
import threading
import queue

# ===================== 六子棋 NNUE 引擎集成配置 =====================
# 引擎可执行文件查找目录（相对本文件所在目录）：
#   Windows: connect6_engine/connect6-engine-win-sse4.1.exe
#   其他平台: connect6_engine/connect6-engine-linux
# 也可用环境变量 CONNECT6_ENGINE_PATH 直接指定 exe 路径。
ENGINE_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "connect6_engine")
CONNECT6_ENGINE_PATH = os.environ.get("CONNECT6_ENGINE_PATH", "")
ENGINE_HARD_TIMEOUT_PAD = 5.0

BOARD_SIZE = 19   # 标准 19 路六子棋棋盘


def find_engine_executable():
    if sys.platform.startswith("win"):
        names = ["connect6-engine-win-sse4.1.exe", "connect6-engine-win.exe", "connect6-engine.exe"]
    else:
        names = ["connect6-engine-linux", "connect6-engine"]
    candidates = []
    if CONNECT6_ENGINE_PATH:
        candidates.append(CONNECT6_ENGINE_PATH)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for n in names:
        candidates.append(os.path.join(ENGINE_DIR_DEFAULT, n))
        candidates.append(os.path.join(base_dir, n))
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


class Connect6Engine:
    """六子棋 NNUE 引擎（exe 子进程）的 Python 包装。

    通过行式文本协议（stdin/stdout）与一个持久子进程通信：
        START 19                  -> OK
        PING                      -> OK
        RESET                     -> OK
        BOARD <361 字符> <side>   -> OK     全量同步棋盘：'0'空 '1'黑 '2'白，
                                            side = 行棋方（1 黑 / 2 白）
        GO <时限ms> <深度>        -> = r1 c1 r2 c2
                                            0 基坐标；首手只返回 1 子（r2 c2 = -1 -1）
        QUIT                      -> 退出

    每次思考都发送完整棋盘并在引擎内部重建局面，因此悔棋、重新开始之后
    引擎看到的棋局必然与界面完全一致，无需任何增量同步状态。
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.exe = find_engine_executable()
        self.proc = None
        self._q = queue.Queue()
        self._reader = None
        self._broken = False
        if self.exe:
            try:
                self._start()
            except Exception as e:
                print("[AI] 引擎启动失败:", e)
                self._broken = True
        else:
            print("[AI] 未找到六子棋引擎可执行文件（connect6_engine/ 目录）")
            self._broken = True

    def _start(self):
        kwargs = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, bufsize=0)
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = 0x08000000   # CREATE_NO_WINDOW
        else:
            kwargs["close_fds"] = True
        self.proc = subprocess.Popen([self.exe], **kwargs)
        self._q = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop,
                                        args=(self._q, self.proc), daemon=True)
        self._reader.start()
        self._send("START 19")
        if not self._expect("OK", 15.0):
            try:
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
            except Exception:
                pass
            self._broken = True
            raise RuntimeError("引擎 START 未返回 OK (%s)" % os.path.basename(self.exe))
        self._broken = False
        print("[AI] 六子棋引擎已启动: %s" % os.path.basename(self.exe))

    def _read_loop(self, q, proc):
        try:
            while proc and proc.poll() is None:
                line = proc.stdout.readline()
                if not line:
                    break
                q.put(line)
        except Exception:
            pass

    def _send(self, cmd):
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError("引擎进程未运行")
        self.proc.stdin.write((cmd + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _drain(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _readline_until(self, deadline, predicate):
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                line = self._q.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            text = line.decode("utf-8", "ignore").strip()
            if predicate(text):
                return text

    def _expect(self, token, timeout=5.0):
        return self._readline_until(time.time() + timeout, lambda t: t == token) is not None

    def reset(self):
        """悔棋 / 重置对局后调用：清空引擎内部缓存。

        发出即忘（不等待应答，避免阻塞界面）；下一次思考仍会用 BOARD 命令
        全量同步真实棋盘，保证 exe 思考的棋局正确。"""
        try:
            if self.proc and self.proc.poll() is None:
                self._send("RESET")
        except Exception:
            self._broken = True

    def search(self, board, player, time_limit_ms, depth=0):
        """思考一手棋，返回 [(r1,c1), (r2,c2)] 或首手 [(r,c)]；失败返回 None。

        board: 19x19 列表，0=空 1=黑 2=白；player: 行棋方 1/2。
        """
        if self._broken or not self.proc or self.proc.poll() is not None:
            try:
                self._start()
            except Exception:
                return None
        cells = "".join(str(board[r][c]) for r in range(19) for c in range(19))
        try:
            self._drain()
            self._send("BOARD %s %d" % (cells, player))
            if not self._expect("OK", 10.0):
                raise RuntimeError("BOARD 未返回 OK")
            self._send("GO %d %d" % (int(time_limit_ms), int(depth)))
            line = self._readline_until(
                time.time() + time_limit_ms / 1000.0 + ENGINE_HARD_TIMEOUT_PAD,
                lambda t: t.startswith("="))
            if line is None:
                raise RuntimeError("GO 超时无响应")
            parts = line[1:].split()
            if len(parts) < 4:
                raise RuntimeError("GO 响应格式错误: %s" % line)
            r1, c1, r2, c2 = (int(x) for x in parts[:4])
            if not (0 <= r1 < 19 and 0 <= c1 < 19) or board[r1][c1] != 0:
                raise RuntimeError("GO 返回非法着法: %s" % line)
            moves = [(r1, c1)]
            if (0 <= r2 < 19 and 0 <= c2 < 19 and board[r2][c2] == 0
                    and (r2, c2) != (r1, c1)):
                moves.append((r2, c2))
            return moves
        except Exception as e:
            print("[AI] 引擎通信失败: %s" % e)
            self._broken = True
            try:
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
            except Exception:
                pass
            return None

    def stop(self):
        try:
            if self.proc and self.proc.poll() is None:
                self._send("QUIT")
                self.proc.wait(timeout=2)
        except Exception:
            try:
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
            except Exception:
                pass


def get_connect6_engine():
    with Connect6Engine._lock:
        if Connect6Engine._instance is None or Connect6Engine._instance._broken:
            Connect6Engine._instance = Connect6Engine()
        return Connect6Engine._instance


class Connect6EngineAI:
    """六子棋 AI：着法全部由 connect6 NNUE 引擎 exe 计算。"""

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size

    def get_best_move(self, board, player, max_time=7.0):
        """返回 (moves, elapsed)；moves 为 [(r,c), ...]（首手 1 点、其余 2 点）。"""
        engine = get_connect6_engine()
        t0 = time.time()
        moves = engine.search(board, player, int(max_time * 1000))
        elapsed = time.time() - t0
        if moves:
            print(f"[AI决策: {moves}, 耗时: {elapsed:.2f}秒]")
        else:
            print("[AI 引擎未返回着法，将随机合法落子兜底]")
        return moves, elapsed


class GobangGame:

    def __init__(self, root):
        self.root = root
        self.root.title("六子棋(游戏制作:白杨博贤)")
        self.root.geometry("950x950")

        self.board_size = BOARD_SIZE
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.game_over = False
        self.piece_count = 0

        self.cell_size = 40
        self.offset = 30
        self.last_moves = []      # 最近一手的落点（第一手 1 点、其余每手 2 点）
        self.win_line = []
        self.move_stack = []
        self.turn_moves = []      # 本回合已落的子（六子棋每回合 1 或 2 子）
        self.ai_thinking = False
        self._ai_gen = 0
        self._ai_inflight = False
        self._ai_result = None    # 工作线程写入，主线程轮询取回（tkinter 非线程安全）

        self.mode = "人机对战"
        self.difficulty = "普通"
        self.my_color = 1
        self.ai_color = 2
        self.time_limit = 7.0

        self._create_ui()
        self._bind_events()

    def _turn_quota(self):
        """本回合应落子数：首回合 1 子，之后每回合 2 子（1-2-2-2 规则）。"""
        return 1 if (self.piece_count == 0 and not self.turn_moves) else 2

    def _create_ui(self):
        self.control_frame = tk.Frame(self.root, bg="#dcdcdc", pady=5)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(self.control_frame, text="模式:", bg="#dcdcdc", font=("黑体", 11)).pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="人机对战")
        for m in ["人机对战", "双人对战", "双机博弈"]:
            tk.Radiobutton(self.control_frame, text=m, variable=self.mode_var, value=m,
                          bg="#dcdcdc", command=self.reset_game).pack(side=tk.LEFT)

        tk.Label(self.control_frame, text=" | 难度:", bg="#dcdcdc", font=("黑体", 11)).pack(side=tk.LEFT, padx=5)
        self.diff_var = tk.StringVar(value="普通")
        for d in ["留手模式", "简单", "普通", "高级"]:
            tk.Radiobutton(self.control_frame, text=d, variable=self.diff_var, value=d,
                          bg="#dcdcdc", command=self._update_settings).pack(side=tk.LEFT)

        tk.Label(self.control_frame, text=" | 玩家执:", bg="#dcdcdc", font=("黑体", 11)).pack(side=tk.LEFT, padx=5)
        self.color_var = tk.StringVar(value="黑方")
        for txt, val in [("黑方", "黑方"), ("白方", "白方")]:
            tk.Radiobutton(self.control_frame, text=txt, variable=self.color_var, value=val,
                          bg="#dcdcdc", command=self.reset_game).pack(side=tk.LEFT)

        tk.Button(self.control_frame, text="悔棋", command=self._undo_move,
                 bg="#87CEEB", font=("黑体", 10, "bold")).pack(side=tk.RIGHT, padx=5)
        tk.Button(self.control_frame, text="重置对局", command=self.reset_game,
                 bg="#FFD700", font=("黑体", 10, "bold")).pack(side=tk.RIGHT, padx=20)

        self.canvas = tk.Canvas(self.root, bg="#E3CF57")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_label = tk.Label(self.root, text="请黑方落子", font=("楷体", 16, "bold"), pady=8)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.algo_label = tk.Label(self.root, text="",
                                  font=("Arial", 10), fg="gray")
        self.algo_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self):
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Button-1>", self._on_click)
        self.root.bind('<Key>', self._on_key_press)

    def _on_click(self, event):
        if self.game_over or self.mode == "双机博弈":
            return

        if self.mode == "人机对战" and self.current_player != self.my_color:
            return

        col = int(round((event.x - self.offset_x) / self.cell_size))
        row = int(round((event.y - self.offset_y) / self.cell_size))

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if self.board[row][col] == 0:
                self._place_piece(row, col)

    def _on_resize(self, event):
        w, h = event.width, event.height
        min_dim = min(w, h)
        self.cell_size = (min_dim - 40) / (self.board_size - 1)
        self.offset = (min_dim - (self.board_size - 1) * self.cell_size) / 2
        self.offset_x = self.offset + (w - min_dim) / 2
        self.offset_y = self.offset + (h - min_dim) / 2
        self._draw_board()

    def _draw_board(self):
        self.canvas.delete("all")

        for i in range(self.board_size):
            sx = self.offset_x
            ex = self.offset_x + (self.board_size - 1) * self.cell_size
            y = self.offset_y + i * self.cell_size
            self.canvas.create_line(sx, y, ex, y)

            sy = self.offset_y
            ey = self.offset_y + (self.board_size - 1) * self.cell_size
            x = self.offset_x + i * self.cell_size
            self.canvas.create_line(x, sy, x, ey)

        stars = [3, 9, 15]
        for r in stars:
            for c in stars:
                x = self.offset_x + c * self.cell_size
                y = self.offset_y + r * self.cell_size
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")

        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r][c] != 0:
                    self._draw_piece(r, c, self.board[r][c])

        for (r, c) in self.last_moves:
            x = self.offset_x + c * self.cell_size
            y = self.offset_y + r * self.cell_size
            self.canvas.create_text(x, y, text="+", fill="red", font=("Arial", 16, "bold"))

        if self.win_line:
            self._draw_win_line()

    def _draw_piece(self, row, col, player):
        x = self.offset_x + col * self.cell_size
        y = self.offset_y + row * self.cell_size
        radius = self.cell_size * 0.42
        color = "black" if player == 1 else "white"
        outline = "#555555" if player == 2 else "black"
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                               fill=color, outline=outline, width=1)

    def _draw_win_line(self):
        if not self.win_line: return
        start_r, start_c = self.win_line[0]
        end_r, end_c = self.win_line[-1]
        sx = self.offset_x + start_c * self.cell_size
        sy = self.offset_y + start_r * self.cell_size
        ex = self.offset_x + end_c * self.cell_size
        ey = self.offset_y + end_r * self.cell_size
        self.canvas.create_line(sx, sy, ex, ey, fill="#FF1493", width=5, capstyle=tk.ROUND)

    def _on_key_press(self, event):
        if event.keysym in ('r', 'R'):
            self.reset_game()
        elif event.keysym == 'space':
            if self.mode == "双机博弈" and not self.game_over:
                self._schedule_ai(50)
        elif event.keysym == 'q':
            try:
                get_connect6_engine().stop()
            except Exception:
                pass
            self.root.quit()

    def _update_settings(self):
        self.difficulty = self.diff_var.get()
        self.time_limit = {
            "留手模式": 1.0,
            "简单": 3.0,
            "普通": 7.0,
            "高级": 15.0,
        }.get(self.difficulty, 15.0)

    def _reset_engine_state(self):
        """棋盘发生回退（悔棋/重置对局）后调用。

        引擎每次思考都会用 BOARD 命令全量同步真实棋盘，所以这里只需让引擎
        清空内部缓存；下一次思考必然按当前界面棋盘进行，保证 exe 思考正确。"""
        try:
            get_connect6_engine().reset()
        except Exception:
            pass

    def reset_game(self):
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.game_over = False
        self.last_moves = []      # 最近一手的落点（第一手 1 点、其余每手 2 点）
        self.win_line = []
        self.piece_count = 0
        self.move_stack = []
        self.turn_moves = []
        self.ai_thinking = False
        self._ai_gen += 1
        self.mode = self.mode_var.get()
        self.my_color = 1 if self.color_var.get() == "黑方" else 2
        self.ai_color = 3 - self.my_color

        # 重置对局也让棋盘完全变化 → 同步重置引擎状态
        self._reset_engine_state()

        self._draw_board()
        self.status_label.config(text="黑方回合")

        if (self.mode == "人机对战" and self.my_color == 2) or self.mode == "双机博弈":
            self._schedule_ai(500)

    def _check_win(self, r, c, player):
        """六子连线（横/竖/两斜）即胜。"""
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            line_coords = [(r, c)]
            for i in range(1, 6):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    line_coords.append((nr, nc))
                else: break
            for i in range(1, 6):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr][nc] == player:
                    line_coords.append((nr, nc))
                else: break
            if len(line_coords) >= 6:
                line_coords.sort(key=lambda x: (x[0], x[1]))
                self.win_line = line_coords
                return True
        return False

    def _place_piece(self, row, col):
        if self.board[row][col] != 0:
            return

        quota = self._turn_quota()
        self.move_stack.append((row, col, self.current_player,
                                list(self.last_moves), self.piece_count, self.game_over,
                                len(self.turn_moves)))
        self.board[row][col] = self.current_player
        self.piece_count += 1
        if len(self.turn_moves) == 0:
            self.last_moves = []          # 新一手开始，标记换组
        self.last_moves.append((row, col))
        self.turn_moves.append((row, col))

        win = self._check_win(row, col, self.current_player)
        self._draw_board()
        self.root.update_idletasks()

        if win:
            winner = "黑方" if self.current_player == 1 else "白方"
            self.status_label.config(text=f"{winner} 获胜!")
            self.game_over = True
            messagebox.showinfo("游戏结束", f"{winner} 获胜!")
            return

        if self.piece_count >= self.board_size * self.board_size:
            self.status_label.config(text="平局!")
            self.game_over = True
            messagebox.showinfo("结束", "平局！")
            return

        turn_text = "黑方" if self.current_player == 1 else "白方"
        if len(self.turn_moves) < quota:
            remain = quota - len(self.turn_moves)
            self.status_label.config(text=f"{turn_text} 回合，请继续落子（还需 {remain} 子）")
            return

        # 本回合落子完成 → 换手
        self.turn_moves = []
        self.current_player = 3 - self.current_player
        turn_text = "黑方" if self.current_player == 1 else "白方"
        if self.mode == "双人对战":
            self.status_label.config(text=f"{turn_text}回合，请落子")
        else:
            self.status_label.config(text=f"{turn_text} 思考中...")

        self._schedule_ai(50)

    def _schedule_ai(self, delay=50):
        if self.game_over:
            self.ai_thinking = False
            return
        is_ai_turn = (self.mode == "人机对战" and self.current_player == self.ai_color) or (self.mode == "双机博弈")
        if not is_ai_turn:
            self.ai_thinking = False
            return
        self.ai_thinking = True
        if self._ai_inflight:
            return
        self._ai_inflight = True
        self.root.after(delay, self._ai_move)

    def _ai_move(self):
        if self.game_over:
            self._ai_inflight = False
            self.ai_thinking = False
            return
        is_ai_turn = (self.mode == "人机对战" and self.current_player == self.ai_color) or (self.mode == "双机博弈")
        if not is_ai_turn:
            self._ai_inflight = False
            self.ai_thinking = False
            return

        self.ai_thinking = True
        ai_player = self.current_player
        gen = self._ai_gen
        current_time = self.time_limit

        print(f"\n{'='*60}")
        print(f"{'黑方' if ai_player == 1 else '白方'} AI 启动引擎...")

        board_copy = [row[:] for row in self.board]

        def _worker():
            try:
                ai_engine = Connect6EngineAI(self.board_size)
                moves, elapsed = ai_engine.get_best_move(board_copy, ai_player, current_time)
            except Exception:
                moves = None
            self._ai_result = (moves, gen)

        threading.Thread(target=_worker, daemon=True).start()
        self.root.after(100, self._poll_ai_result)

    def _poll_ai_result(self):
        """主线程轮询 AI 结果（tkinter 只能在主线程更新界面）。"""
        if self._ai_result is None:
            self.root.after(100, self._poll_ai_result)
            return
        moves, gen = self._ai_result
        self._ai_result = None
        self._on_ai_done(moves, gen)

    def _on_ai_done(self, moves, gen):
        self._ai_inflight = False
        if gen != self._ai_gen:
            self._schedule_ai(100)
            return
        self.ai_thinking = False

        if self.game_over:
            return

        # 本回合尚需落的子数（引擎应一手给足：首手 1 子、其余 2 子）
        need = self._turn_quota() - len(self.turn_moves)
        placed = 0
        if moves:
            print("="*60)
            for (r, c) in moves[:need]:
                if self.game_over:
                    break
                if (0 <= r < self.board_size and 0 <= c < self.board_size
                        and self.board[r][c] == 0):
                    self._place_piece(r, c)
                    placed += 1

        # 引擎不可用 / 返回子数不足时，随机合法落子补齐本回合，避免卡住
        for _ in range(need - placed):
            if self.game_over:
                break
            candidates = [(r, c) for r in range(self.board_size)
                         for c in range(self.board_size) if self.board[r][c] == 0]
            if not candidates:
                break
            self._place_piece(*random.choice(candidates))

    def _undo_move(self):
        if not self.move_stack:
            messagebox.showinfo("提示", "没有可悔的棋步")
            return

        # AI 思考中也可悔棋：_ai_gen 递增会使在途思考结果作废，
        # 并在新棋盘上重新调用 exe 思考（每次思考都全量同步棋盘，安全）。

        self._ai_gen += 1

        def _pop_turn():
            """弹出栈顶一整个回合（含其全部棋子）并恢复局面。"""
            while self.move_stack:
                (row, col, player, prev_last_moves, prev_piece_count,
                 prev_game_over, seq) = self.move_stack.pop()
                self.board[row][col] = 0
                self.last_moves = prev_last_moves
                self.piece_count = prev_piece_count
                self.game_over = prev_game_over
                self.current_player = player
                if seq == 0:
                    break

        if self.turn_moves:
            # 本回合已落子但未走完（中途悔棋）→ 只取消本回合已落的子
            _pop_turn()
        else:
            # 人机对战回退 2 个完整回合（AI 回合 + 玩家回合），其余模式回退 1 回合
            turns = 2 if self.mode == "人机对战" else 1
            for _ in range(turns):
                if not self.move_stack:
                    break
                _pop_turn()

        self.turn_moves = []
        self.win_line = []

        # === 关键：悔棋使棋盘回退，同步重置引擎状态 ===
        # 引擎每次思考都会用 BOARD 命令全量同步整个棋盘，加上这里重置缓存，
        # 保证 exe 思考的棋局与真实棋盘完全一致
        self._reset_engine_state()

        self._draw_board()

        turn_text = "黑方" if self.current_player == 1 else "白方"
        if self.mode == "双人对战":
            self.status_label.config(text=f"{turn_text}回合，请落子")
        else:
            self.status_label.config(text=f"{turn_text} 回合")

        self._schedule_ai(500)

if __name__ == "__main__":
    root = tk.Tk()
    game = GobangGame(root)

    print("="*60)
    print("六子棋(19路，1-2-2-2 规则)(游戏制作:白杨博贤)")
    print("="*60)

    root.mainloop()
