# fly plays itself and learns with its mushroom body
#   python selfplay.py <minutes> <weights_file> <seed>
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # one core per worker, also saves ~600MB ram
import time, random
import numpy as np, chess
from fly_player import Fly, SWARM

if __name__ == "__main__":
    minutes, path, seed = float(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    random.seed(seed)
    # one fly (with senses), no search, old forcing-reply lookahead: fast enough to play many games
    fly = Fly(seed=seed, mb=True, lookahead=True, weights=path, readouts=SWARM[:1], depth=1)
    end = time.time() + minutes * 60
    print(f"start: {path}, n={fly.mb.n}", flush=True)
    games, err, base, seen = 0, [], [], []
    while time.time() < end:
        b = chess.Board()
        fly.pending.clear()
        for _ in range(2): b.push(random.choice(list(b.legal_moves)))   # random first moves for variety
        while not b.is_game_over(claim_draw=True) and b.fullmove_number <= 80 and time.time() < end:
            mover = b.turn
            b.push(fly.choose(b))
            # the side to move now just saw the reply to its last move
            for color in ([b.turn, mover] if b.is_game_over() else [b.turn]):
                c = fly.pending.get(color, (0, 0, None))[2]
                p = fly.mb.predict(c) if c is not None else 0.0
                d = fly.learn(b, color)
                if d is None: continue
                # base: error of just guessing the average so far (what a memory with no kc info gets)
                if seen: err.append(d * d); base.append((d + p - np.mean(seen)) ** 2)
                seen.append(d + p)
                fly.mb.save()
                if fly.mb.n % 10 == 0 and err:
                    print(f"n={fly.mb.n}  dopamine^2 last50 {np.mean(err[-50:]):.3f}  "
                          f"(average-guess {np.mean(base[-50:]):.3f})  ply {len(b.move_stack)}", flush=True)
        games += 1
        print(f"game {games}: {b.result(claim_draw=True)} after {len(b.move_stack)} plies", flush=True)
    fly.mb.save()
    print(f"done: n={fly.mb.n}, dopamine^2 {np.mean(err) if err else 0:.3f} vs average-guess {np.mean(base) if base else 0:.3f}", flush=True)
