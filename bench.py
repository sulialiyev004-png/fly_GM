# strength test vs a greedy bot (mates if it can, else takes the biggest piece, else random) or vs random moves
#   python bench.py <mode> <games> <seed> [greedy|random]
# modes: base       old fly (fly_readout.npz), tastes each move
#        ahead      base + checks forcing replies for its top 4 moves
#        ahead+mem  ahead + mushroom body memory (mb_weights.npz, frozen)
#        fly        new fly: 2-fly swarm with threat senses (w1s + w2s), tastes each move, no search
#        fly+1      new fly + 1 extra ply: alpha-beta over the opponent's replies (2 plies)
#        fly+2      new fly + 2 extra plies (3 plies)
#        fly8       all 8 flies, no search (the gui / fly_player default)
# the gui uses all 8 flies, the bench only 2 so that 20+20 games per mode fit in the time
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time, random
import numpy as np, chess
from fly_player import Fly, MODEL, SWARM
from mb import WEIGHTS, VALUES, material

MAX_MOVES = 40   # then adjudicate by material, 3+ points = win
BENCH = SWARM[1:3]
MODES = {"base": dict(readouts=[MODEL], depth=1), "ahead": dict(readouts=[MODEL], depth=1, lookahead=True),
         "ahead+mem": dict(readouts=[MODEL], depth=1, lookahead=True, mb=True),
         "fly": dict(readouts=BENCH, depth=1), "fly+1": dict(readouts=BENCH, depth=2), "fly+2": dict(readouts=BENCH, depth=3),
         "fly8": dict(readouts=SWARM, depth=1)}   # the default: all 8 flies, no search


def greedy(b):
    best, pick = -1, []
    for m in b.legal_moves:
        b.push(m); mate = b.is_checkmate(); b.pop()
        g = 1000 if mate else (VALUES[b.piece_type_at(m.to_square)] if b.piece_type_at(m.to_square) else 0)
        if b.is_en_passant(m): g = 1
        if m.promotion: g += VALUES[m.promotion] - 1
        if g > best: best, pick = g, [m]
        elif g == best: pick.append(m)
    return random.choice(pick)


def rand(b):
    return random.choice(list(b.legal_moves))


def play(fly, fly_white, opponent=greedy):
    b, swings, before = chess.Board(), [], None
    me = chess.WHITE if fly_white else chess.BLACK
    while not b.is_game_over(claim_draw=True) and b.fullmove_number <= MAX_MOVES:
        if b.turn == me:
            before = material(b, me)
            b.push(fly.choose(b))
        else:
            b.push(opponent(b))
            if before is not None:   # what fly's move + the reply did to material
                swings.append(material(b, me) - before)
    r = b.result(claim_draw=True)
    if r == "*":
        m = material(b, chess.WHITE)
        r = "1-0" if m >= 3 else "0-1" if m <= -3 else "1/2-1/2"
    score = 0.5 if r == "1/2-1/2" else float((r == "1-0") == fly_white)
    return score, r, b, swings


if __name__ == "__main__":
    mode, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    opp = sys.argv[4] if len(sys.argv) > 4 else "greedy"
    assert mode in MODES and opp in ("greedy", "random"), (mode, opp)
    random.seed(seed)
    if mode == "ahead+mem" and not WEIGHTS.exists(): print("warning: no mb_weights.npz, memory is empty")
    fly = Fly(seed=seed, **MODES[mode])
    if fly.mb: print(f"memory: n={fly.mb.n}", flush=True)
    scores, swings = [], []
    for k in range(n):
        t0, e0, fly_white = time.time(), fly.evals, (seed + k) % 2 == 0
        s, r, b, sw = play(fly, fly_white, greedy if opp == "greedy" else rand)
        scores.append(s); swings += sw
        blunders = sum(x <= -3 for x in sw)
        print(f"{mode} vs {opp} seed={seed} game {k + 1}: fly {'white' if fly_white else 'black'} {r} score {s} "
              f"plies {len(b.move_stack)} swing/move {np.mean(sw) if sw else 0:+.2f} blunders {blunders} "
              f"{time.time() - t0:.0f}s brain-runs/move {(fly.evals - e0) / max(1, len(sw)):.0f}", flush=True)
        print("   " + chess.Board().variation_san(b.move_stack), flush=True)
    w, d = scores.count(1.0), scores.count(0.5)
    print(f"RESULT {mode} vs {opp} seed={seed}: +{w} ={d} -{len(scores) - w - d}  score {np.mean(scores):.2f}  "
          f"swing/move {np.mean(swings):+.3f}  blunder rate {np.mean(np.array(swings) <= -3):.3f}  moves {len(swings)}", flush=True)
