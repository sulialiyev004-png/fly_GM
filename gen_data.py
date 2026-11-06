# make training data: positions + stockfish eval + fly brain response
import sys, random, shutil, numpy as np, scipy.sparse as sp, chess, chess.engine
from reservoir import FlyReservoir

mode, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
random.seed(seed)
eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/usr/games/stockfish")
res = FlyReservoir(mode=mode, seed=seed)
rows, ys, fens = [], [], []
while len(ys) < n:
    b = chess.Board()
    # varied games: mix of engine moves and random moves
    eng.configure({"Skill Level": random.randint(0, 20)})
    for ply in range(random.randint(8, 120)):
        if b.is_game_over(): break
        if random.random() < 0.25: b.push(random.choice(list(b.legal_moves)))
        else: b.push(eng.play(b, chess.engine.Limit(depth=3)).move)
        if ply >= 6 and random.random() < 0.15 and not b.is_game_over():
            sc = eng.analyse(b, chess.engine.Limit(depth=10))["score"].pov(b.turn).score(mate_score=3000)
            rows.append(sp.csr_matrix(res.respond(b).astype(np.float32)))
            ys.append(sc); fens.append(b.fen())
            if len(ys) % 50 == 0: print(len(ys), flush=True)
            if len(ys) >= n: break
eng.quit()
np.savez_compressed(out, y=np.array(ys), fen=np.array(fens))
sp.save_npz(out + ".X.npz", sp.vstack(rows).tocsr())
print("done", out)
