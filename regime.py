import sys, numpy as np, chess
from reservoir import FlyReservoir
from train import load
rate, tms, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
_, cp, fens = load(["ds/hub_a"])
r = FlyReservoir(mode="hub", rate_hz=rate, t_ms=tms, seed=5)
X = np.stack([r.respond(chess.Board(f)) for f in fens[:n]])
np.savez_compressed(f"ds/regime_{int(rate)}_{int(tms)}", X=X, y=cp[:n])
print("done", rate, tms, X.sum(1).mean())
