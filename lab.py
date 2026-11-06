# stage lab: fixed dataset + teacher, cached brain responses, held-out tests, readouts for the player
#   python lab.py gen <seed> <n>            positions from varied games -> pos_<seed>.npz
#   python lab.py merge                     -> positions.npz (teacher, senses, train / held-out split by game)
#   python lab.py sim <cfg> <shard> <k>     cached brain responses (5 bins of 10ms). cfg: w0, w0s (senses), w0sa (senses + apl)
#   python lab.py eval <spec> ...           held-out corr + who's-better accuracy
#        spec: board | board+senses | w0:tot | w0s:bins | w0s:t20 | w0s+w1s:bins (swarm) | w0:kc (kenyon cells vs best capture)
#   python lab.py fit <cfg> <kind>          readout for the player -> readouts/<cfg>_<kind>.npz (train split only)
# brain caches are big (~150MB per cfg), set FLYGM_DS to keep them out of a synced folder
import os, sys
if sys.argv[1] in ("gen", "sim"): os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import re, glob, random, time
from pathlib import Path
import numpy as np, chess
from reservoir import FlyReservoir, board_features, threat_senses
from mb import material, VALUES

HERE = Path(__file__).parent
DS = Path(os.environ.get("FLYGM_DS", HERE / "runs" / "ds"))
READOUTS = HERE / "readouts"
MATE_Q = 40.0   # teacher score for mate, pawns
HELD = 0.27     # share of games kept for the held-out set


def qsearch(b, alpha=-99.0, beta=99.0, depth=8):
    # teacher: material (side to move, pawns) after captures only. in check: all evasions
    ms = list(b.legal_moves)
    check = b.is_check()
    if not ms: return -MATE_Q if check else 0.0
    stand = float(material(b, b.turn))
    if depth == 0: return stand
    if check: best = -MATE_Q
    else:
        best = stand
        if best >= beta: return best
        alpha = max(alpha, best)
        ms = [m for m in ms if b.is_capture(m) or m.promotion]
    victim = lambda m: VALUES.get(b.piece_type_at(m.to_square), 0) or (1 if b.is_en_passant(m) else 0)
    ms.sort(key=lambda m: (-victim(m), VALUES[b.piece_type_at(m.from_square)]))   # mvv-lva
    for m in ms:
        b.push(m); s = -qsearch(b, -beta, -alpha, depth - 1); b.pop()
        if s > best:
            best = s
            if s > alpha: alpha = s
            if alpha >= beta: break
    return best


def config(cfg):
    w, s, a = re.fullmatch(r"w(\d+)(s?)(a?)", cfg).groups()
    return dict(wiring=int(w), seed=int(w), senses=bool(s), apl=bool(a))


def gen(seed, n):
    # games mixing greedy bot, random and (old) fly moves; sometimes one extra random move,
    # since the fly mostly tastes positions after candidate moves, blunders included
    from fly_player import Fly
    from bench import greedy
    random.seed(seed); rnd = random.Random(seed)
    fly = Fly(seed=seed)
    fens, games, srcs, qs = [], [], [], []
    g = 0
    while len(fens) < n:
        b, g, took = chess.Board(), g + 1, 0
        for ply in range(rnd.randint(12, 150)):
            if b.is_game_over(): break
            r = rnd.random()
            if r < 0.05: mv, who = fly.choose(b), "fly"
            elif r < 0.65: mv, who = greedy(b), "greedy"
            else: mv, who = rnd.choice(list(b.legal_moves)), "random"
            b.push(mv)
            if ply >= 4 and took < 10 and rnd.random() < 0.12 and not b.is_game_over():
                extra = rnd.random() < 0.5
                if extra: b.push(rnd.choice(list(b.legal_moves))); who += "+cand"
                if not b.is_game_over():
                    fens.append(b.fen()); games.append(seed * 10000 + g); srcs.append(who); qs.append(qsearch(b)); took += 1
                if extra: b.pop()
        print(f"game {g}: {len(fens)} positions", flush=True)
    np.savez(DS / f"pos_{seed}.npz", fen=fens, game=games, src=srcs, q=qs)


def merge():
    d = [np.load(f) for f in sorted(glob.glob(str(DS / "pos_*.npz")))]
    fens = [f for x in d for f in x["fen"]]; games = [int(g) for x in d for g in x["game"]]
    srcs = [s for x in d for s in x["src"]]; qs = [float(q) for x in d for q in x["q"]]
    # real fly games from the last bench (fly vs greedy), every 3rd position
    k = 900000
    for f in sorted(glob.glob(str(HERE / "runs" / "bench_*.log"))):
        for line in open(f, encoding="utf-8"):
            if not line.startswith("   1."): continue
            b = chess.Board(); k += 1
            for i, tok in enumerate(t for t in line.split() if not t.endswith(".")):
                b.push_san(tok)
                if i % 3 == 2 and not b.is_game_over():
                    fens.append(b.fen()); games.append(k); srcs.append("bench"); qs.append(qsearch(b))
    held = np.array([random.Random(g).random() < HELD for g in games])
    S = np.array([threat_senses(chess.Board(f)) for f in fens])
    np.savez(DS / "positions.npz", fen=fens, game=games, src=srcs, q=qs, held=held, senses=S)
    q = np.array(qs)
    print(f"{len(fens)} positions, {len(set(games))} games, train {(~held).sum()}, held-out {held.sum()}, "
          f"teacher q mean {q.mean():+.2f} sd {q.std():.2f}, q==0 {np.mean(q == 0):.0%}")


def sim(cfg, shard, k):
    fens = np.load(DS / "positions.npz")["fen"]
    idx = np.arange(len(fens))[shard::k]
    brain = FlyReservoir(**config(cfg))
    X = np.zeros((len(idx), 5, brain.n), np.uint8)
    t = time.time()
    for j, i in enumerate(idx):
        X[j] = np.minimum(brain.run([chess.Board(fens[i])])[0], 255)
    on = X.any(axis=(0, 1))   # keep neurons that fired at least once
    np.savez_compressed(DS / f"{cfg}.{shard}.npz", idx=idx, X=X[:, :, on], act=brain.S[on])
    print(f"{cfg} shard {shard}: {len(idx)} boards, {(time.time() - t) / len(idx):.3f}s each", flush=True)


def load_brain(cfg, rows, only=None):
    # all shards -> (positions in `rows` order, bins, neurons that ever fired), global neuron ids
    files = glob.glob(str(DS / f"{cfg}.*.npz"))
    S = np.unique(np.concatenate([np.load(f)["act"] for f in files]))
    if only is not None: S = S[np.isin(S, only)]
    pos = np.full(100000, -1); pos[rows] = np.arange(len(rows))
    X = np.zeros((len(rows), 5, len(S)), np.uint8)
    for f in files:
        s = np.load(f); a = s["act"]; m = np.isin(a, S); r = pos[s["idx"]]; ok = r >= 0
        X[r[ok][:, None], :, np.searchsorted(S, a[m])[None, :]] = s["X"][ok][:, :, m].transpose(0, 2, 1)
    return X, S


def chunks(X, kind):
    # readout features a few thousand columns at a time: sqrt spike counts
    if kind == "bins":
        for j in range(0, X.shape[2], 800):
            yield np.sqrt(X[:, :, j:j + 800].reshape(len(X), -1).astype(np.float32))
        return
    nb = 5 if kind in ("tot", "kc", "in", "down") else int(kind[1:]) // 10
    for j in range(0, X.shape[2], 4000):
        yield np.sqrt(X[:, :nb, j:j + 4000].sum(1, dtype=np.uint16).astype(np.float32))


def ridge(make_chunks, y, tr, te, games, alphas=np.logspace(-3, 7, 21)):
    # kernel ridge (features don't fit in ram), alpha by 5-fold cv grouped by game on the train split
    # (leave-one-out picked alphas that memorize games: positions of one game are alike)
    ntr = tr.size
    Ktr, Kte, mus = np.zeros((ntr, ntr)), np.zeros((te.size, ntr)), []
    for Z in make_chunks():
        mu = Z[:ntr].mean(0); A = Z[:ntr] - mu
        Ktr += A @ A.T; Kte += (Z[ntr:] - mu) @ A.T; mus.append(mu)
    ym = y[tr].mean(); yt = y[tr] - ym
    fold = np.array([random.Random(int(g) * 7 + 1).randrange(5) for g in games[tr]])
    err = np.zeros(len(alphas))
    for f in range(5):
        a, b = fold != f, fold == f
        Kaa, Kba = Ktr[np.ix_(a, a)], Ktr[np.ix_(b, a)]
        m0, m1 = Kaa.mean(0), Kaa.mean()
        Kba = Kba - m0 - Kba.mean(1)[:, None] + m1; Kaa = Kaa - m0 - Kaa.mean(1)[:, None] + m1   # recenter on the fold
        e, Q = np.linalg.eigh(Kaa); Qy = Q.T @ (yt[a] - yt[a].mean())
        for i, al in enumerate(alphas):
            err[i] += np.sum((Kba @ (Q @ (Qy / (e + al))) + yt[a].mean() - yt[b]) ** 2)
    al = alphas[np.argmin(err)]
    e, Q = np.linalg.eigh(Ktr)
    c = Q @ ((Q.T @ yt) / (e + al))
    return Kte @ c + ym, c, ym, al, mus


def report(name, p, P, te, target="q"):
    q = P["q"][te] if target == "q" else P["gain"][te]
    r = np.corrcoef(p, np.tanh(q / 4.0))[0, 1]
    if target != "q":
        print(f"  {name:<40} corr with best capture {r:.3f}", flush=True); return r, None
    nz = q != 0
    tac = nz & (P["gain"][te] != 0)   # a capture changes who's better or by how much
    acc = np.mean(np.sign(p[nz]) == np.sign(q[nz]))
    acc_t = np.mean(np.sign(p[tac]) == np.sign(q[tac]))
    print(f"  {name:<40} corr {r:.3f}   who's-better {acc:.1%}   tactical {acc_t:.1%}", flush=True)
    return r, acc


def evaluate(specs, fit=None):
    P = dict(np.load(DS / "positions.npz"))
    P["gain"] = P["q"] - np.array([material(chess.Board(f), chess.Board(f).turn) for f in P["fen"]])
    held = P["held"]; tr, te = np.flatnonzero(~held), np.flatnonzero(held)
    rows = np.concatenate([tr, te])
    (DS / "pred").mkdir(exist_ok=True)
    out = {}
    for spec in specs:
        if spec in ("board", "board+senses"):
            R = np.zeros((len(held), 768), np.float32)
            for i, f in enumerate(P["fen"]): R[i, board_features(chess.Board(f))] = 1
            if spec == "board+senses": R = np.hstack([R, P["senses"]])
            p = ridge(lambda: iter([R[rows]]), np.tanh(P["q"] / 4.0), tr, te, P["game"])[0]
            out[spec] = report(spec, p, P, te); continue
        if spec == "old":   # the original fly_readout.npz (stockfish labels), same brain as w0
            d = np.load(HERE / "fly_readout.npz")
            X, S = load_brain("w0", te)
            loc = np.searchsorted(S, d["idx"]); ok = (loc < len(S)) & (S[np.minimum(loc, len(S) - 1)] == d["idx"])
            p = np.sqrt(X[:, :, loc[ok]].sum(1).astype(np.float32)) @ d["coef"][ok] + float(d["b"])
            out[spec] = report("old readout (fly_readout.npz)", p, P, te); continue
        cfgs, kind = spec.split(":")
        ps = []
        for cfg in cfgs.split("+"):
            f = DS / "pred" / f"{cfg}_{kind}.npy"
            if f.exists() and fit is None: ps.append(np.load(f)); continue
            only = np.load(HERE / "mb_cells.npz")["kc"] if kind == "kc" else None
            if kind in ("in", "down"):   # diagnostics: only the input (board/sense) neurons, or everything but them
                br = FlyReservoir(**config(cfg))
                inp = np.concatenate([br.groups.ravel()] + ([br.sense.ravel()] if br.sense is not None else []))
                only = inp if kind == "in" else np.setdiff1d(br.S, inp)
            X, S = load_brain(cfg, rows, only)
            target = P["gain"] if kind == "kc" else P["q"]
            mk = lambda: chunks(X, kind)
            p, c, ym, a, mus = ridge(mk, np.tanh(target / 4.0), tr, te, P["game"])
            np.save(f, p); ps.append(p)
            if len(cfgs.split("+")) == 1:
                out[spec] = report(f"{cfg}:{kind} ({len(S)} neurons, alpha {a:.0e})", p, P, te, "gain" if kind == "kc" else "q")
            if fit is not None:
                # primal weights for the player: w = A^T c, back to (bins, neurons)
                ws = [(Z[:tr.size] - mu).T @ c for Z, mu in zip(mk(), mus)]
                b = float(ym - sum(mu @ wc for mu, wc in zip(mus, ws)))
                w = np.concatenate([wc.reshape(5, -1) if kind == "bins" else wc[None] for wc in ws], axis=1).astype(np.float32)
                nb = int(kind[1:]) // 10 if kind[0] == "t" and kind != "tot" else 5
                READOUTS.mkdir(exist_ok=True)
                np.savez(READOUTS / f"{cfg}_{kind}.npz", idx=S.astype(np.int32), coef=w, b=b, kind=kind, nbins=nb,
                         **{k: v for k, v in config(cfg).items()})
                print(f"  saved readouts/{cfg}_{kind}.npz", flush=True)
            del X
        if len(ps) > 1: out[spec] = report(f"swarm of {len(ps)} {kind}: {cfgs}", np.mean(ps, 0), P, te)
    return out


if __name__ == "__main__":
    DS.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "gen": gen(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "merge": merge()
    elif cmd == "sim": sim(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "eval": evaluate(sys.argv[2:])
    elif cmd == "fit": evaluate([f"{sys.argv[2]}:{sys.argv[3]}"], fit=True)
