# board + live fly brain view (3d: drag to rotate, scroll to zoom, double-click to reset)
# run: python fly_gui.py
# the fly learns from your games (mushroom body), saved to mb_weights.npz
import os, threading, time, tkinter as tk
from collections import deque
import numpy as np
import chess
from fly_player import Fly
from reservoir import board_features
from brainviz import BrainView, BrainView3D

from pathlib import Path

SQ = 96                                   # square size
VIEW_W, VIEW_H = 700, 460                 # brain view
FPS = 20                                  # brain animation
FADE_S = 0.35                             # spikes fade out with this time constant
LIVE = 0.35                               # brightness of live spikes vs the picked move's 50ms total
SPIN, IDLE_S = 0.3, 3.0                   # auto-rotation (rad/s), starts this long after you let go
BG, PANEL, TXT, DIM = "#161512", "#262421", "#bababa", "#8b8987"   # lichess dark UI


def _mix(a, b, t):
    """Blend hex color a with b at opacity t (tk has no alpha, so we pre-blend)."""
    a, b = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in (a, b)]
    return "#%02x%02x%02x" % tuple(round(x * (1 - t) + y * t) for x, y in zip(a, b))


# lichess "brown" board + its overlay colors
LIGHT, DARK = "#f0d9b5", "#b58863"
LAST = {True: _mix(LIGHT, "#9bc700", 0.41), False: _mix(DARK, "#9bc700", 0.41)}
SEL = {True: _mix(LIGHT, "#14551e", 0.5), False: _mix(DARK, "#14551e", 0.5)}
DEST = {True: _mix(LIGHT, "#14551e", 0.5), False: _mix(DARK, "#14551e", 0.5)}
CHECK = ["#ff0000", "#e70000", "#a90000", "#9e0000"]
UI = "Segoe UI"
PIECE_DIR = Path(__file__).parent / "pieces" / "big"   # 90px versions of the 68px set
REWARD, PAIN = (90, 255, 120), (255, 40, 40)   # PAM / PPL1 flash colors
FLASH_S = 2.0


class App:
    def __init__(self, root):
        self.root = root
        root.title("FlyGM 🪰 - chess vs a fruit fly brain")
        root.configure(bg=BG)

        # board (left)
        self.canvas = tk.Canvas(root, width=8 * SQ, height=8 * SQ, highlightthickness=0, bg=BG)
        self.canvas.grid(row=0, column=0, rowspan=2, padx=(14, 8), pady=(14, 6))
        self.status = tk.Label(root, text="", fg=TXT, bg=BG, font=(UI, 13))
        self.status.grid(row=2, column=0, pady=4)
        bar = tk.Frame(root, bg=BG); bar.grid(row=3, column=0, pady=(0, 14))
        mk = dict(bg="#302e2c", fg=TXT, activebackground="#3692e7", activeforeground="#ffffff",
                  relief="flat", font=(UI, 11), padx=12, pady=4, bd=0, cursor="hand2")
        tk.Button(bar, text="New game as White", command=lambda: self.new_game(True), **mk).pack(side="left", padx=4)
        tk.Button(bar, text="New game as Black", command=lambda: self.new_game(False), **mk).pack(side="left", padx=4)
        tk.Button(bar, text="Undo move", command=self.undo, **mk).pack(side="left", padx=4)
        self.canvas.bind("<Button-1>", self.click)
        # cburnett piece set (the lichess default), pre-rendered to PNG
        self.piece_img = {}
        for sym in "KQRBNPkqrbnp":
            name = ("w" if sym.isupper() else "b") + sym.upper()
            self.piece_img[sym] = tk.PhotoImage(file=str(PIECE_DIR / f"{name}.png"))

        # brain (right)
        right = tk.Frame(root, bg=PANEL); right.grid(row=0, column=1, rowspan=4, sticky="n", padx=(8, 14), pady=14)
        top = tk.Frame(right, bg=PANEL); top.pack(fill="x", padx=12, pady=(10, 4))
        self.title = tk.Label(top, text="", fg=TXT, bg=PANEL, font=(UI, 12, "bold"))
        self.title.pack(side="left")
        tk.Button(top, text="2D / 3D", command=self.toggle_view, **dict(mk, font=(UI, 9), padx=8, pady=1)).pack(side="right")
        self.view3d, self.view2d, self.use3d = BrainView3D(VIEW_W, VIEW_H), BrainView(VIEW_W), True
        box = tk.Frame(right, bg=PANEL, width=VIEW_W, height=VIEW_H)   # fixed size, the 2d view is shorter
        box.pack(padx=12); box.pack_propagate(False)
        self.brain_img = tk.PhotoImage(data=self.view3d.render(), format="PPM")
        self.brain_lbl = tk.Label(box, image=self.brain_img, bg=PANEL, bd=0)
        self.brain_lbl.pack(expand=True)
        for ev, fn in (("<ButtonPress-1>", self.grab), ("<B1-Motion>", self.turn), ("<ButtonRelease-1>", self.let_go),
                       ("<MouseWheel>", self.wheel), ("<Double-Button-1>", self.reset_view)):
            self.brain_lbl.bind(ev, fn)
        self.brain_cap = tk.Label(right, text="brain at rest", fg=DIM, bg=PANEL, font=(UI, 10))
        self.brain_cap.pack(anchor="w", padx=12, pady=(4, 0))
        tk.Label(right, text="blue dots: neurons that receive the board    fire: spikes    drag: rotate, scroll: zoom, double-click: reset",
                 fg=DIM, bg=PANEL, font=(UI, 9), wraplength=VIEW_W, justify="left").pack(anchor="w", padx=12)
        self.mem_lbl = tk.Label(right, text="", fg=DIM, bg=PANEL, font=(UI, 10), wraplength=VIEW_W, justify="left")
        self.mem_lbl.pack(anchor="w", padx=12, pady=(2, 0))
        tk.Label(right, text="How the fly feels about each move", fg=TXT, bg=PANEL,
                 font=(UI, 12, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        self.bars = tk.Canvas(right, width=VIEW_W, height=200, bg=PANEL, highlightthickness=0)
        self.bars.pack(padx=12, pady=(0, 12))

        self.fly, self.board, self.you = None, chess.Board(), chess.WHITE
        self.sel, self.thinking, self.result = None, False, None
        self.fly_said = ""
        self.flash, self.flash_until = None, 0.0
        # what the brain view shows: fading activity per neuron, fed by the fly's thread
        self.act = np.zeros(self.view3d.p.shape[0], np.float32)
        self.feed = deque()          # spike frames from the thinking thread
        self.pick = None             # (spikes, inputs, caption) of the move it chose
        self.hold, self.lit = False, False
        self.inputs, self.caption = None, "brain at rest"
        self.drag, self.touched = None, 0.0
        self.t_frame, self.frames = time.time(), 0
        self.show_mode()
        self.set_status("🪰 Loading the fly brain...")
        self.draw()
        threading.Thread(target=self.load_fly, daemon=True).start()
        self.animate()

    # brain
    def load_fly(self):
        # best setup from fly_player (see README), replies are tasted in parallel by worker processes
        self.fly = Fly(mb=True, workers=max(1, (os.cpu_count() or 2) - 2))
        self.fly.note = lambda t: setattr(self, "caption", t)
        self.root.after(0, lambda: (self.set_status("Your move"), self.show_mem(), self.draw()))

    def animate(self):
        # draws the brain ~20x a second on the main thread, the fly's thread only hands over spikes
        t = time.time()
        dt, self.t_frame = min(t - self.t_frame, 0.25), t
        if self.pick is not None:   # the move it chose: keep its spikes on screen until something happens
            self.act[:], self.inputs, self.caption = self.pick
            self.pick, self.hold, self.lit = None, True, True
        while self.feed:
            self.act += self.feed.popleft(); self.lit = True
        if not self.hold and self.lit:
            self.act *= np.float32(np.exp(-dt / FADE_S))
            if self.act.max() < 0.05: self.act[:] = 0; self.lit = False
        v = self.view3d if self.use3d else self.view2d
        if self.use3d and self.drag is None and t - self.touched > IDLE_S: v.yaw += SPIN * dt
        fl = self.flashing()
        if fl is not None:   # dopamine / pain glow fades out at the end
            fl = (fl[0], np.asarray(fl[1], np.float32) * min(1.0, (self.flash_until - t) / 0.7))
        ppm = v.render(self.act if self.lit else None, self.inputs, gain=LIVE, flash=fl)
        self.brain_img.configure(data=ppm, format="PPM")
        self.brain_cap.config(text=self.caption)
        self.frames += 1
        self.root.after(max(5, int(1000 / FPS - (time.time() - t) * 1000)), self.animate)

    def toggle_view(self):
        self.use3d = not self.use3d
        self.show_mode()

    def show_mode(self):
        self.title.config(text="Fly brain | 3D, drag to rotate | 127,400 neurons" if self.use3d
                          else "Fly brain | front view | 127,400 neurons")
        self.brain_lbl.config(cursor="fleur" if self.use3d else "")

    # 3d view: drag rotates, wheel zooms, double-click resets, it spins again a few seconds after you let go
    def grab(self, e):
        self.drag = (e.x, e.y, self.view3d.yaw, self.view3d.pitch)

    def turn(self, e):
        if self.drag is None or not self.use3d: return
        x, y, yaw, pitch = self.drag
        self.view3d.yaw = yaw - (e.x - x) * 0.01
        self.view3d.pitch = min(1.5, max(-1.5, pitch - (e.y - y) * 0.01))

    def let_go(self, e):
        self.drag, self.touched = None, time.time()

    def wheel(self, e):
        v = self.view3d
        v.zoom = min(4.0, max(0.5, v.zoom * 1.1 ** (e.delta / 120)))
        self.touched = time.time()

    def reset_view(self, e):
        v = self.view3d
        v.yaw = v.pitch = 0.0; v.zoom = 1.0
        self.drag, self.touched = None, time.time()

    def close(self):
        if self.fly: self.fly.close()
        self.root.destroy()

    def flashing(self):
        return self.flash if time.time() < self.flash_until else None

    def fly_think(self, board):
        f = self.fly

        def watch(b, tag):   # live spikes while the brain runs on position b (drawn by animate)
            self.inputs, self.hold = f.brain.groups[board_features(b)].ravel(), False

            def on_frame(recent, t_ms):
                self.feed.append(recent)
                self.caption = f"{tag} | {t_ms:.0f} ms of simulation"
            return on_frame

        mv = f.choose(board, watch=watch)
        _, v, counts, san = f.chosen
        if counts is not None:
            board.push(mv); stim = f.brain.groups[board_features(board)].ravel(); board.pop()
            n = int(np.count_nonzero(counts))
            self.pick = (counts.astype(np.float32) / LIVE, stim, f"picked {san}: {n:,} neurons fired, {int(counts.sum()):,} spikes")
        self.result = (mv, v)

    def dopamine(self, d):
        # flash PAM (reward) or PPL1 (pain) neurons after the fly saw how its move worked out
        if d is None: return
        mb = self.fly.mb
        mb.save()
        if d > 0.02: self.flash, txt, col = (mb.pam, REWARD), f"dopamine {d:+.2f} (PAM): better than expected", "#7fc96b"
        elif d < -0.02: self.flash, txt, col = (mb.ppl1, PAIN), f"pain {d:+.2f} (PPL1): worse than expected", "#e06a5c"
        else: self.flash, txt, col = None, f"no surprise ({d:+.2f})", DIM
        self.flash_until = time.time() + FLASH_S
        self.hold, self.inputs, self.caption = False, None, txt   # the picked move's spikes fade out
        self.show_mem(txt, col)

    def show_mem(self, last="", col=DIM):
        n = self.fly.mb.n if self.fly and self.fly.mb else 0
        self.mem_lbl.config(text=f"mushroom body: {n:,} lessons" + (f"  |  {last}" if last else ""), fg=col)

    def start_fly(self):
        self.thinking, self.result = True, None
        threading.Thread(target=self.fly_think, args=(self.board.copy(),), daemon=True).start()
        self.poll()

    def poll(self):
        self.draw_bars()
        if self.result is None:
            self.set_status("🪰 The fly is thinking...")
            self.root.after(60, self.poll)
            return
        mv, feel = self.result
        self.fly_said = f"🪰 {self.board.san(mv)}  (it feels: {feel:+.2f})"
        self.board.push(mv)
        self.thinking = False
        self.draw_bars(chosen=self.fly_said.split()[1])
        if self.board.is_game_over():   # no reply coming, it learns now (e.g. it gave mate)
            self.dopamine(self.fly.learn(self.board, not self.you))
        self.after_move()

    def draw_bars(self, chosen=None):
        c = self.bars
        c.delete("all")
        scores = list(self.fly.scores) if self.fly else []
        if not scores: return
        top = sorted(scores, key=lambda s: -s[1])[:8]
        runs = f" | {self.fly.last[1]} brain runs in {self.fly.last[0]:.0f}s" if chosen and hasattr(self.fly, "last") else ""
        c.create_text(4, 2, anchor="nw", fill=DIM, font=(UI, 9),
                      text=f"moves checked: {len(scores)}{runs} | showing its top 8")
        mid, scale = int(VIEW_W * 0.55), int(VIEW_W * 0.33)
        for k, (san, v) in enumerate(top):
            y = 24 + k * 22
            vv = max(-1.0, min(1.0, v))
            col = "#7fc96b" if vv >= 0 else "#e06a5c"
            if san == chosen: col = "#f6d743"
            c.create_text(4, y + 8, anchor="w", fill=TXT if san == chosen else DIM, font=(UI, 10, "bold"), text=san)
            c.create_line(mid, y, mid, y + 16, fill="#444")
            c.create_rectangle(mid, y + 2, mid + vv * scale, y + 14, fill=col, outline="")
            c.create_text(mid + vv * scale + (6 if vv >= 0 else -6), y + 8, anchor="w" if vv >= 0 else "e",
                          fill=DIM, font=(UI, 9), text=f"{v:+.2f}")

    # game
    def new_game(self, white):
        if self.thinking or self.fly is None: return
        self.board, self.you, self.sel, self.fly_said = chess.Board(), chess.WHITE if white else chess.BLACK, None, ""
        self.fly.scores = []; self.fly.pending.clear(); self.draw_bars()
        self.hold, self.inputs = False, None
        self.draw()
        if self.board.turn != self.you: self.start_fly()
        else: self.set_status("Your move")

    def undo(self):
        if self.thinking or len(self.board.move_stack) < 2: return
        self.board.pop(); self.board.pop(); self.sel = None; self.fly_said = ""
        self.fly.pending.clear()   # don't learn from a taken-back move
        self.after_move()

    def after_move(self):
        self.draw()
        if self.board.is_game_over():
            r = self.board.result()
            if r == "1/2-1/2": winner = "Draw"
            else: winner = "You won" if (r == "1-0") == (self.you == chess.WHITE) else "The fly won"
            self.set_status(f"{winner}   ({r})")
        elif self.board.turn == self.you:
            self.set_status(f"{self.fly_said}   | your move" if self.fly_said else "Your move")
        else:
            self.start_fly()

    def click(self, e):
        if self.thinking or self.fly is None or self.board.is_game_over() or self.board.turn != self.you: return
        sq = self.xy_to_sq(e.x, e.y)
        pc = self.board.piece_at(sq)
        if self.sel is not None:
            mv = chess.Move(self.sel, sq)
            if self.board.piece_type_at(self.sel) == chess.PAWN and chess.square_rank(sq) in (0, 7):
                mv.promotion = chess.QUEEN
            if mv in self.board.legal_moves:
                self.board.push(mv); self.sel = None
                self.dopamine(self.fly.learn(self.board, not self.you))   # fly sees what your reply did
                self.after_move(); return
        self.sel = sq if pc and pc.color == self.you else None
        self.draw()

    # board drawing
    def sq_to_xy(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.you == chess.WHITE: return f * SQ, (7 - r) * SQ
        return (7 - f) * SQ, r * SQ

    def xy_to_sq(self, x, y):
        f, r = int(x // SQ), 7 - int(y // SQ)
        if self.you == chess.BLACK: f, r = 7 - f, 7 - r
        return chess.square(min(max(f, 0), 7), min(max(r, 0), 7))

    def draw(self):
        c, b = self.canvas, self.board
        c.delete("all")
        last = b.peek() if b.move_stack else None
        targets = {m.to_square for m in b.legal_moves if m.from_square == self.sel} if self.sel is not None else set()
        king_in_check = b.king(b.turn) if b.is_check() else None
        for sq in chess.SQUARES:
            x, y = self.sq_to_xy(sq)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            base = LIGHT if light else DARK
            col = base
            if last and sq in (last.from_square, last.to_square): col = LAST[light]
            if sq == self.sel: col = SEL[light]
            c.create_rectangle(x, y, x + SQ, y + SQ, fill=col, outline="")
            pc = b.piece_at(sq)
            cx, cy = x + SQ / 2, y + SQ / 2
            if sq == king_in_check:            # lichess-style red glow under the king
                for k, rc in enumerate(CHECK):
                    r = SQ * (0.5 - 0.1 * k)
                    c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=_mix(col, rc, 0.35 + 0.15 * k), outline="")
            if sq in targets:
                if pc:                         # capture: four green corners
                    c.create_rectangle(x, y, x + SQ, y + SQ, fill=DEST[light], outline="")
                    r = SQ * 0.5
                    c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=col, outline="")
                else:                          # quiet move: green dot
                    r = SQ * 0.13
                    c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=DEST[light], outline="")
            # coordinates inside the edge squares, like lichess
            txt = DARK if light else LIGHT
            if chess.square_rank(sq) == (0 if self.you == chess.WHITE else 7):
                c.create_text(x + SQ - 4, y + SQ - 2, text="abcdefgh"[chess.square_file(sq)], anchor="se", fill=txt, font=(UI, 9, "bold"))
            if chess.square_file(sq) == (7 if self.you == chess.WHITE else 0):
                c.create_text(x + SQ - 4, y + 2, text=str(chess.square_rank(sq) + 1), anchor="ne", fill=txt, font=(UI, 9, "bold"))
            if pc:
                c.create_image(cx, cy, image=self.piece_img[pc.symbol()])

    def set_status(self, t):
        self.status.config(text=t)


if __name__ == "__main__":
    try:   # real pixel sizes on scaled windows screens (else windows stretches it and it's blurry)
        import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass
    root = tk.Tk()
    root.resizable(False, False)
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
