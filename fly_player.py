# fly plays alone: fly brain + linear readout (stockfish isn't needed)
#   python fly_player.py play | vs-random | vs-sf0
#   python fly_player.py fit ds/a ds/b   -> fly_readout.npz (old stockfish-labelled data)
import os, sys, random, time, shutil, numpy as np, chess, chess.pgn, chess.engine
from pathlib import Path
from reservoir import FlyReservoir
from mb import MB, WEIGHTS, MATE, material, VALUES

HERE = Path(__file__).parent
MODEL = HERE / "fly_readout.npz"   # the original readout (one fly, 50ms total counts)
SWARM = [HERE / "readouts" / f"w{k}s_tot.npz" for k in range(8)]   # 8 flies: own wiring + senses, 50ms counts
READOUTS = SWARM   # best setup from lab.py, see README
DEPTH = 1        # search plies (1 = just taste each move). 2 plies played worse in the bench, see README
BEAM = 8         # search: only its 8 best moves (by taste) get the alpha-beta
TOP = 4         # lookahead: how many of the best moves get checked
MAX_REPLIES = 8  # lookahead: forcing replies tasted per move (biggest captures first)
WIN = 9.0        # mate, in feel units (feel is about -1..1)


def fit(prefixes):
    from sklearn.linear_model import RidgeCV
    from train import load
    X, cp, _ = load(prefixes)
    y = np.tanh(cp / 400.0)
    active = np.unique(X.indices)
    Xa = np.sqrt(X[:, active].toarray())
    m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Xa, y)
    np.savez(MODEL, idx=active, coef=m.coef_.astype(np.float32), b=m.intercept_)
    print(f"saved {MODEL.name}: {len(active)} neurons, alpha={m.alpha_:.3g}")


class Readout:
    # one fly: its brain (wiring, senses, apl) + linear readout of sqrt spike counts
    def __init__(self, path, brains, seed=None):
        d = np.load(path)
        if "wiring" in d: cfg = dict(wiring=int(d["wiring"]), seed=int(d["seed"]), senses=bool(d["senses"]), apl=bool(d["apl"]))
        else: cfg = dict(wiring=0, seed=seed or 0, senses=False, apl=False)   # old fly_readout.npz
        self.nb = int(d["nbins"]) if "nbins" in d else 5
        self.bins = "kind" in d and str(d["kind"]) == "bins"
        key = tuple(cfg.values())
        if key not in brains: brains[key] = FlyReservoir(**cfg, t_ms=10.0 * self.nb)
        self.brain = brains[key]
        self.brain.t_ms = max(self.brain.t_ms, 10.0 * self.nb)
        loc = self.brain.loc[d["idx"]]
        self.loc, self.coef, self.b = loc[loc >= 0], np.atleast_2d(d["coef"])[:, loc >= 0], float(d["b"])

    def read(self, X):
        # X: (bins, live neurons) spike counts -> feel for the side to move
        x = X[:self.nb, self.loc].astype(np.float32)
        if self.bins: return float((np.sqrt(x) * self.coef).sum() + self.b)
        return float(np.sqrt(x.sum(0)) @ self.coef[0] + self.b)


_pool_fly = None
def _pool_init(paths, seed):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    global _pool_fly
    _pool_fly = Fly(seed=seed, readouts=paths, depth=1)
def _pool_feel(fen):
    return _pool_fly.feel(chess.Board(fen))


class Fly:
    def __init__(self, seed=None, mb=False, lookahead=False, weights=WEIGHTS, readouts=None, depth=None, workers=0):
        self.paths = [str(p) for p in (readouts or READOUTS)]
        brains = {}
        self.flies = [Readout(p, brains, seed) for p in self.paths]   # several = swarm, evaluations averaged
        self.brain = self.flies[0].brain
        self.seed = seed
        self.mb = MB(weights) if mb else None
        self.lookahead = lookahead
        self.depth = DEPTH if depth is None else depth
        self.workers, self.pool = workers, None
        self.note = None     # optional callback(str), the gui shows what it's doing
        self.cache = {}      # same board always feels the same (inputs are seeded by the position)
        self.evals = 0       # brain runs so far
        self.scores = []     # (san, value) of every move it tasted, for the gui
        self.pending = {}    # color -> (material before, after own move, spike counts)

    def taste(self, board, watch=None, tag=""):
        """(feel for the side to move in [-1, 1], spike counts of all neurons of the first fly)"""
        feel, c = [], None
        for k, f in enumerate(self.flies):
            on = watch(board, tag if len(self.flies) == 1 else f"{tag}  (fly {k + 1})") if watch else None
            X = f.brain.run([board], on_frame=on)[0]
            feel.append(f.read(X))
            if k == 0:
                c = np.zeros(f.brain.N, np.int16); c[f.brain.S] = X.sum(0)
        self.evals += len(self.flies)
        v = float(np.mean(feel))
        self.cache[self.key(board)] = v
        return v, c

    @staticmethod
    def key(board):
        return board.board_fen() + "wb"[board.turn == chess.BLACK]

    def feel(self, board):
        k = self.key(board)
        if k not in self.cache:
            if len(self.cache) > 300000: self.cache.clear()
            self.taste(board)
        return self.cache[k]

    def value(self, board, mv, watch=None, tag=""):
        # how good mv feels, plus what the mushroom body expects the reply to cost
        board.push(mv)
        k = self.key(board)
        if board.is_checkmate(): v, c = WIN, None          # game rules, not chess skill
        elif board.is_game_over(): v, c = 0.0, None
        elif k in self.cache and self.mb is None and watch is None: v, c = -self.cache[k], None
        else:
            if k in self.cache: f, c = self.cache[k], self.show(board, watch, tag)   # tasted by the workers already
            else: f, c = self.taste(board, watch, tag)
            v = -f + (self.mb.predict(c) if self.mb else 0.0)
        board.pop()
        return v, c

    def show(self, board, watch=None, tag=""):
        # rerun the first fly (live view in the gui, kenyon cells for the memory), feel comes from the cache
        f = self.flies[0]
        X = f.brain.run([board], on_frame=watch(board, tag) if watch else None)[0]
        c = np.zeros(f.brain.N, np.int16); c[f.brain.S] = X.sum(0)
        return c

    def worst_reply(self, board, mv, watch=None, tag=""):
        # imagine the opponent's captures, checks and promotions after mv, return the worst for us
        board.push(mv)
        forcing = [r for r in board.legal_moves if board.is_capture(r) or r.promotion or board.gives_check(r)]
        worst, left = None, []
        for r in forcing:
            board.push(r)
            if board.is_checkmate(): worst = -WIN
            elif board.is_game_over(): worst = 0.0 if worst is None else min(worst, 0.0)
            else: left.append(r)
            board.pop()
        if worst != -WIN:
            victim = lambda r: board.piece_type_at(r.to_square) or (1 if board.is_en_passant(r) else 0)
            for r in sorted(left, key=lambda r: (-victim(r), -bool(r.promotion)))[:MAX_REPLIES]:
                rs = board.san(r)
                board.push(r)
                x = self.taste(board, watch, f"{tag}  if {rs}")[0]   # our turn again
                board.pop()
                worst = x if worst is None else min(worst, x)
        board.pop()
        return worst

    @staticmethod
    def ordered(board):
        # captures first (biggest victim, cheapest attacker), then promotions and checks
        def key(m):
            v = board.piece_type_at(m.to_square)
            cap = VALUES[v] if v else (1 if board.is_en_passant(m) else 0)
            att = VALUES[board.piece_type_at(m.from_square)] or 10
            return (-(cap * 10 - att) if cap else 0, -bool(m.promotion), -board.gives_check(m))
        return sorted(board.legal_moves, key=key)

    def negamax(self, board, depth, alpha, beta):
        # value for the side to move, fly brain at the leaves
        if board.is_checkmate(): return -WIN
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3): return 0.0
        if depth == 0: return self.feel(board)
        best = -99.0
        for m in self.ordered(board):
            board.push(m); s = -self.negamax(board, depth - 1, -beta, -alpha); board.pop()
            if s > best:
                best = s
                if s > alpha: alpha = s
                if alpha >= beta: break
        return best

    def leaves(self, board, moves, depth):
        # every leaf of the full tree (to taste them in parallel before the alpha-beta)
        out = []
        def walk(d):
            if board.is_game_over(): return
            if d == 0:
                if self.key(board) not in self.cache: out.append(board.fen())
                return
            for m in board.legal_moves:
                board.push(m); walk(d - 1); board.pop()
        for m in moves:
            board.push(m); walk(depth - 1); board.pop()
        return list(dict.fromkeys(out))

    def prefetch(self, board, moves, depth):
        if not self.workers or depth > 2: return
        if self.pool is None:
            import multiprocessing as mp
            self.pool = mp.get_context("spawn").Pool(self.workers, _pool_init, (self.paths, self.seed))
        fens = self.leaves(board, moves, depth)
        for fen, v in zip(fens, self.pool.map(_pool_feel, fens, chunksize=4)):
            self.cache[self.key(chess.Board(fen))] = v
        self.evals += len(fens) * len(self.flies)

    def search(self, board, order, depth):
        # alpha-beta over the root moves (ordered by how they felt at depth 1)
        self.prefetch(board, order, depth)
        alpha, best, vals = -99.0, order[0], {}
        for m in order:
            board.push(m); s = -self.negamax(board, depth - 1, -99.0, -alpha); board.pop()
            vals[m] = s
            if s > alpha: alpha, best = s, m
        return best, vals

    def choose(self, board, verbose=False, watch=None):
        self.scores = []
        t0, e0 = time.time(), self.evals
        moves = list(board.legal_moves)
        self.prefetch(board, moves, 1)   # with workers: all flies taste all moves in parallel first
        cand = []
        for i, mv in enumerate(moves):
            san = board.san(mv)
            v, c = self.value(board, mv, watch, f"tasting {san}  ({i + 1}/{len(moves)})")
            cand.append([v, mv, c, san])
            self.scores.append((san, v))
        cand.sort(key=lambda s: -s[0])
        if self.depth > 1 and cand[0][0] < WIN:
            if self.note: self.note(f"imagining replies to its best {min(BEAM, len(cand))} moves ({self.depth} plies)")
            best, vals = self.search(board, [s[1] for s in cand[:BEAM]], self.depth)
            for s in cand: s[0] = vals.get(s[1], s[0])
            cand.sort(key=lambda s: (s[1] != best, -s[0]))
            self.scores = [(s[3], s[0]) for s in cand]
        elif self.lookahead:
            cand = cand[:TOP]      # only plays moves it has thought through
            for s in cand:
                w = self.worst_reply(board, s[1], watch, f"checking {s[3]}")
                if w is not None: s[0] = min(s[0], w)
            cand.sort(key=lambda s: -s[0])
            ahead = {s[3]: s[0] for s in cand}
            self.scores = [(san, ahead.get(san, v)) for san, v in self.scores]
        v, mv, c, san = cand[0]
        if self.mb is not None and c is None:   # mate or cached: taste it, so the memory gets the reward
            board.push(mv); c = self.taste(board)[1]; board.pop()
        if self.mb is not None and c is not None:
            before = material(board, board.turn)
            board.push(mv); after = material(board, not board.turn); board.pop()
            self.pending[board.turn] = (before, after, c)
        self.chosen = (mv, v, c, san)
        self.last = (time.time() - t0, self.evals - e0)
        if verbose: print(f"   🪰 {san}  (feels {v:+.2f}, {self.last[1]} brain runs, {self.last[0]:.0f}s)")
        return mv

    def learn(self, board, color):
        """after the opponent replied to color's last move: dopamine = outcome - prediction"""
        if self.mb is None or color not in self.pending: return None
        before, after, c = self.pending.pop(color)
        if board.is_checkmate(): out = -MATE if board.turn == color else MATE
        else: out = (material(board, color) - before) / 4.0
        return self.mb.learn(c, max(-MATE, min(MATE, out)), (after - before) / 4.0)

    def close(self):
        if self.pool is not None: self.pool.terminate(); self.pool = None


def adjudicate(b):
    # move cap: stockfish judges if installed, else material (3+ points = win)
    sf = shutil.which("stockfish") or ("/usr/games/stockfish" if Path("/usr/games/stockfish").exists() else None)
    if sf:
        with chess.engine.SimpleEngine.popen_uci(sf) as j:
            s = j.analyse(b, chess.engine.Limit(depth=15))["score"].white().score(mate_score=10000)
        return ("1-0" if s > 300 else "0-1" if s < -300 else "1/2-1/2"), f"adjudicated at move 60, eval {s:+}cp"
    m = material(b, chess.WHITE)
    return ("1-0" if m >= 3 else "0-1" if m <= -3 else "1/2-1/2"), f"adjudicated at move 60, material {m:+}"


def game(fly, opponent, fly_white, name):
    b, g = chess.Board(), chess.pgn.Game()
    g.headers.update(White="FlyGM 🪰" if fly_white else name, Black=name if fly_white else "FlyGM 🪰")
    node, t0 = g, time.time()
    while not b.is_game_over(claim_draw=True) and b.fullmove_number <= 60:
        mv = fly.choose(b, True) if (b.turn == chess.WHITE) == fly_white else opponent(b)
        node = node.add_variation(mv); b.push(mv)
    g.headers["Result"] = b.result(claim_draw=True)
    if g.headers["Result"] == "*":
        g.headers["Result"], g.headers["Termination"] = adjudicate(b)
    print(f"{g.headers['White']} vs {g.headers['Black']}: {g.headers['Result']} in {b.fullmove_number} moves, {time.time()-t0:.0f}s")
    return g


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        fit(sys.argv[2:]); sys.exit()
    fly = Fly(workers=max(1, (os.cpu_count() or 2) - 2))
    if cmd == "play":
        b = chess.Board(); you = input("white or black? [w/b] ").strip() != "b"
        while not b.is_game_over():
            print("\n", b, sep="")
            if b.turn == you:
                try: b.push_san(input("your move: ").strip())
                except ValueError: print("illegal")
            else: b.push(fly.choose(b, True))
        print("Result:", b.result()); fly.close(); sys.exit()
    white = "--black" not in sys.argv
    if cmd == "vs-random":
        g = game(fly, lambda b: random.choice(list(b.legal_moves)), white, "Random")
    else:
        eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/usr/games/stockfish")
        eng.configure({"Skill Level": 0})
        g = game(fly, lambda b: eng.play(b, chess.engine.Limit(time=0.05)).move, white, "Stockfish L0")
        eng.quit()
    fly.close()
    open(f"{cmd}.pgn", "w", encoding="utf-8").write(str(g) + "\n")   # utf-8 for the 🪰 on windows
