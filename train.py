# compare readouts: raw board vs fly neurons
import sys, numpy as np, scipy.sparse as sp, chess
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from reservoir import board_features

def load(prefixes):
    Xs, ys, fens = [], [], []
    for p in prefixes:
        d = np.load(p + ".npz"); Xs.append(sp.load_npz(p + ".X.npz")); ys.append(d["y"]); fens += list(d["fen"])
    return sp.vstack(Xs).tocsr(), np.concatenate(ys), fens

def raw(fens):
    R = np.zeros((len(fens), 768), np.float32)
    for i, f in enumerate(fens): R[i, board_features(chess.Board(f))] = 1
    return R

def evaluate(name, X, y, folds=5):
    preds = np.zeros_like(y)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(X):
        m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
    r = np.corrcoef(preds, y)[0, 1]
    sign = np.mean(np.sign(preds) == np.sign(y))
    print(f"  {name:<28} corr={r:.3f}  who's-better acc={sign:.1%}")
    return r

if __name__ == "__main__":
    mode, prefixes = sys.argv[1], sys.argv[2:]
    from reservoir import FlyReservoir
    X, cp, fens = load(prefixes)
    y = np.tanh(cp / 400.0)
    print(f"{mode}: {len(y)} positions, {X.shape[1]} neurons, {len(np.unique(X.indices))} ever active")
    R = raw(fens)
    active = np.unique(X.indices)
    Xa = np.sqrt(X[:, active].toarray())
    # which active columns are stimulated input neurons?
    groups = np.load(f"ds/groups_{mode}.npy")
    is_in = np.isin(active, groups.ravel())
    evaluate("raw board (768)", R, y)
    evaluate(f"fly downstream only ({(~is_in).sum()})", Xa[:, ~is_in], y)
    evaluate("fly all neurons", Xa, y)
    evaluate("board + fly", np.hstack([R, Xa[:, ~is_in]]), y)
