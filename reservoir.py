# fly brain as a reservoir: board (+ threat senses) -> input neurons -> run 50ms -> spike counts
import os, zlib, random
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import chess

HERE = Path(__file__).parent
DATA = HERE / "data"
CACHE = DATA / "cache"
N = 127400
V0, VRST, VTH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, KICK = 0.275, 0.275 * 250
BIN = 10.0         # ms per spike count bin
MARGIN = 3.0       # live set: neurons that got this close to threshold (mV) on sample boards
KC_ACTIVE = 0.05   # apl: share of kenyon cells allowed to fire per board
PER_SENSE = 4      # bristle neurons per sense channel
SENSES = ["pain P", "pain N", "pain B", "pain R", "pain Q",            # my piece attacked, undefended or by a cheaper piece
          "ache P", "ache N", "ache B", "ache R", "ache Q",            # my piece attacked but defended
          "hunger P", "hunger N", "hunger B", "hunger R", "hunger Q",  # their piece I can win
          "king zone", "check"]
VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}


def build_cache():
    # parquet -> plain .npy files, built once (pandas keeps ~2GB per process otherwise)
    import pandas as pd
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    con = pd.read_parquet(DATA / "2023_03_23_connectivity_630_final.parquet", columns=cols)
    pre, post = con.Presynaptic_Index.values, con.Postsynaptic_Index.values
    w = con["Excitatory x Connectivity"].values.astype(np.float32) * W_SYN
    W = sp.csc_matrix((w, (post, pre)), shape=(N, N), dtype=np.float32)
    # input neurons: purely excitatory, ranked by total outgoing weight
    #   "hub":      strongest excitatory neurons anywhere, 1 per feature
    #   "afferent": low-input (sensory-like) neurons, 3 per feature
    indeg = np.bincount(post, minlength=N)
    out_pos = np.asarray(W.maximum(0).sum(axis=0)).ravel()
    out_neg = np.asarray(W.minimum(0).sum(axis=0)).ravel()
    arrays = {}
    for mode, k in (("hub", 1), ("afferent", 3)):
        ok = (out_pos > 0) & (out_neg == 0)
        if mode == "afferent": ok &= indeg <= 50
        cand = np.flatnonzero(ok)
        cand = cand[np.argsort(-out_pos[cand])]
        arrays[f"groups_{mode}"] = cand[:768 * k].reshape(768, k)
        if mode == "hub": arrays["hub_pool"] = cand[:768 * 4]   # for rewired flies
    arrays.update(w_data=W.data, w_indices=W.indices, w_indptr=W.indptr)
    CACHE.mkdir(exist_ok=True)
    for name, a in arrays.items(): save(CACHE / f"{name}.npy", a)


def save(path, a):
    # write then rename, safe if two processes build at once
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, a)
    os.replace(tmp, path)


def load_w():
    # memory-mapped, so parallel workers share one copy of the connectome
    if not (CACHE / "w_indptr.npy").exists(): build_cache()
    d, i, p = (np.load(CACHE / f"w_{k}.npy", mmap_mode="r") for k in ("data", "indices", "indptr"))
    return sp.csc_matrix((d, i, p), shape=(N, N), copy=False)


def hub_pool():
    f = CACHE / "hub_pool.npy"
    if not f.exists():   # older caches didn't save it
        W = load_w(); d, p = np.asarray(W.data), np.asarray(W.indptr)
        col = np.repeat(np.arange(N), np.diff(p))
        pos = np.bincount(col, np.maximum(d, 0), N); neg = np.bincount(col, np.minimum(d, 0), N)
        cand = np.flatnonzero((pos > 0) & (neg == 0))
        save(f, cand[np.argsort(-pos[cand])][:768 * 4])
    return np.load(f)


def board_groups(mode="hub", wiring=0):
    if not (CACHE / f"groups_{mode}.npy").exists(): build_cache()   # first run
    g = np.load(CACHE / f"groups_{mode}.npy")
    if wiring == 0: return g
    # another fly: same kind of neurons, different random wiring
    return np.random.default_rng(1000 + wiring).choice(hub_pool(), g.size, replace=False).reshape(g.shape)


def sense_groups(wiring, taken):
    # real mechanosensory bristle neurons (flywire annotations), strongest output first
    pool = np.load(HERE / "mb_cells.npz")["bristle"]
    pool = pool[~np.isin(pool, taken)]
    n = len(SENSES) * PER_SENSE
    pick = pool[:n] if wiring == 0 else np.random.default_rng(2000 + wiring).choice(pool[:300], n, replace=False)
    return pick.reshape(len(SENSES), PER_SENSE)


def board_features(board: chess.Board) -> np.ndarray:
    """Indices (0..767) of active piece-square features, side to move = 'us'."""
    b = board if board.turn == chess.WHITE else board.mirror()
    idx = []
    for sq, pc in b.piece_map().items():
        own = 0 if pc.color == chess.WHITE else 1
        idx.append((own * 6 + pc.piece_type - 1) * 64 + sq)
    return np.array(idx, dtype=np.int64)


def threat_senses(b: chess.Board) -> np.ndarray:
    # nociceptor-like senses for the side to move (raw counts, see SENSES)
    me, x = b.turn, np.zeros(len(SENSES), np.float32)
    for sq, pc in b.piece_map().items():
        if pc.piece_type == chess.KING: continue
        att = b.attackers(not pc.color, sq)
        if not att: continue
        weak = not b.attackers(pc.color, sq) or min(VAL[b.piece_type_at(a)] for a in att) < VAL[pc.piece_type]
        t = pc.piece_type - 1
        if pc.color == me: x[t if weak else 5 + t] += 1
        elif weak: x[10 + t] += 1
    k = b.king(me)
    if k is not None: x[15] = sum(b.is_attacked_by(not me, s) for s in chess.SquareSet(chess.BB_KING_ATTACKS[k]))
    x[16] = b.is_check()
    return x


def sense_rates(x):
    # sense -> firing rate of its bristle neurons: 100Hz per unit, max 300Hz
    lvl = x.copy(); lvl[15] /= 2; lvl[16] *= 3
    return np.minimum(lvl, 3) * 100.0


def ranges(starts, lens):
    # concatenated aranges [s, s + l)
    end = np.cumsum(lens)
    return np.arange(end[-1]) + np.repeat(starts - end + lens, lens)


def sample_boards(n, seed=0):
    # varied positions (greedy-ish and random moves) for finding the live set
    rnd, out = random.Random(seed), []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rnd.randint(0, 90)):
            if b.is_game_over(): break
            ms = list(b.legal_moves); caps = [m for m in ms if b.is_capture(m)]
            b.push(rnd.choice(caps if caps and rnd.random() < 0.5 else ms))
        if not b.is_game_over(): out.append(b)
    return out


class FlyReservoir:
    def __init__(self, mode="hub", dt=0.1, t_ms=50.0, rate_hz=200.0, seed=0, wiring=0, senses=False, apl=False, live=True):
        self.N, self.dt, self.t_ms, self.rate, self.seed, self.apl = N, dt, t_ms, rate_hz, seed or 0, apl
        self.groups = board_groups(mode, wiring)    # feature -> neurons
        self.sense = sense_groups(wiring, self.groups) if senses else None
        self.key = f"{mode}_w{wiring}" + ("_s" if senses else "") + ("_apl" if apl else "")
        self.kc = np.load(HERE / "mb_cells.npz")["kc"]
        if live and not (CACHE / f"live_{self.key}_indptr.npy").exists(): self.make_live()
        if live:
            # only neurons that can fire for some board, the rest never spike (checked in make_live)
            self.S = np.load(CACHE / f"live_{self.key}.npy")
            d, i, p = (np.load(CACHE / f"live_{self.key}_{k}.npy", mmap_mode="r") for k in ("data", "indices", "indptr"))
            self.W = sp.csc_matrix((d, i, p), shape=(len(self.S),) * 2, copy=False)
        else:
            self.S, self.W = np.arange(N), load_w()
        self.n = len(self.S)
        self.loc = np.full(N, -1, np.int64); self.loc[self.S] = np.arange(self.n)
        self.kc_loc = self.loc[self.kc][self.loc[self.kc] >= 0]

    def inputs(self, board):
        # input neurons (global ids) and their poisson rates
        ids = self.groups[board_features(board)].ravel()
        rate = np.full(ids.size, self.rate)
        if self.sense is not None:
            r = sense_rates(threat_senses(board)); on = r > 0
            ids = np.concatenate([ids, self.sense[on].ravel()])
            rate = np.concatenate([rate, np.repeat(r[on], PER_SENSE)])
        return ids, rate

    def respond(self, board: chess.Board, on_frame=None, every=100) -> np.ndarray:
        """Spike counts of all neurons for this position.
        on_frame(recent_counts, t_ms) is called every `every` steps (for live visualisation)."""
        c = np.zeros(N, np.int16)
        c[self.S] = self.run([board], on_frame, every)[0].sum(0)
        return c

    def run(self, boards, on_frame=None, every=100, vmax=None):
        """(boards, bins of 10ms, live neurons) spike counts"""
        B, n, dt = len(boards), self.n, self.dt
        T, per = int(round(self.t_ms / dt)), int(round(BIN / dt))
        # input spikes, seeded by the position so the same board always feels the same
        ts, fs, stim = [], [], []
        for b, board in enumerate(boards):
            ids, rate = self.inputs(board)
            ids = self.loc[ids] + b * n
            key = (board.board_fen() + " " + "wb"[board.turn == chess.BLACK]).encode()
            rng = np.random.default_rng([self.seed, zlib.crc32(key)])
            t, j = np.nonzero(rng.random((T, ids.size)) < rate * (dt / 1000.0))
            ts.append(t); fs.append(ids[j]); stim.append(ids)
        t = np.concatenate(ts); o = np.argsort(t, kind="stable")
        fire, cut = np.concatenate(fs)[o], np.searchsorted(t[o], np.arange(T + 1))
        is_stim = np.zeros(B * n, bool); is_stim[np.concatenate(stim)] = True
        if self.apl:
            is_kc = np.zeros(B * n, bool); is_kc[(self.kc_loc + n * np.arange(B)[:, None]).ravel()] = True
            won, nwon = np.zeros(B * n, bool), np.zeros(B, np.int64)
        d, ix, p = self.W.data, self.W.indices, self.W.indptr
        v = np.full(B * n, V0, np.float32); g = np.zeros(B * n, np.float32); tmp = np.empty_like(v)
        counts = np.zeros((-(-T // per), B * n), np.int16)
        a_v, a_g = np.float32(dt / T_MBR), np.float32(np.exp(-dt / TAU))
        delay, rfc = int(round(T_DLY / dt)), int(round(T_RFC / dt))
        empty = np.empty(0, np.int64)
        buf, ref, last = [empty] * delay, [empty] * rfc, 0
        for step in range(T):
            arr = buf[step % delay]
            if arr.size:   # spikes from T_DLY ago arrive
                pre = arr % n
                s = p[pre]; l = p[pre + 1] - s
                k = ranges(s, l)
                np.add.at(g, ix[k] if B == 1 else ix[k] + np.repeat(arr - pre, l), d[k])
            r = np.concatenate(ref)   # refractory: voltage held at reset, g doesn't decay
            gr = g[r]
            np.subtract(V0, v, out=tmp); tmp += g; tmp *= a_v; v += tmp
            g *= a_g
            v[r] = VRST; g[r] = gr
            v[fire[cut[step]:cut[step + 1]]] += KICK
            if vmax is not None: np.maximum(vmax, v, out=vmax)
            spk = np.flatnonzero(v > VTH)
            if self.apl and spk.size: spk = self.inhibit(spk, v, is_kc, won, nwon)
            v[spk] = VRST; g[spk] = 0.0
            counts[step // per, spk] += 1
            buf[step % delay] = spk
            ref[step % rfc] = spk[~is_stim[spk]]   # input neurons have no refractory period
            if on_frame is not None and (step + 1) % every == 0:
                c = np.zeros(N, np.int16); c[self.S] = counts[:, :n].sum(0)
                on_frame(c - last, (step + 1) * dt); last = c
        return counts.reshape(-1, B, n).transpose(1, 0, 2)

    def inhibit(self, spk, v, is_kc, won, nwon):
        # apl: after the first KC_ACTIVE share of kenyon cells fired, the rest are shut down
        c = spk[is_kc[spk] & ~won[spk]]
        if not c.size: return spk
        b = c // self.n
        o = np.lexsort((-v[c], b)); c, b = c[o], b[o]   # most depolarized win first
        ok = np.arange(c.size) - np.searchsorted(b, b) < int(KC_ACTIVE * len(self.kc)) - nwon[b]
        won[c[ok]] = True; np.add.at(nwon, b[ok], 1)
        v[c[~ok]] = VRST
        return np.setdiff1d(spk, c[~ok], assume_unique=True)

    def make_live(self, n_boards=200, batch=10):
        # full brain on sample boards: keep every neuron that came near threshold
        full = FlyReservoir.__new__(FlyReservoir)
        full.__dict__.update(self.__dict__)
        full.S, full.W, full.n = np.arange(N), load_w(), N
        full.loc = np.arange(N); full.kc_loc = self.kc
        keep = np.zeros(N, bool)
        boards = sample_boards(n_boards)
        for i in range(0, n_boards, batch):
            vmax = np.full(batch * N, V0, np.float32)
            full.run(boards[i:i + batch], vmax=vmax)
            keep |= (vmax.reshape(batch, N) > VTH - MARGIN).any(0)
        keep[self.groups.ravel()] = True
        if self.sense is not None: keep[self.sense.ravel()] = True
        S = np.flatnonzero(keep)
        W = full.W[:, S].tocsr()[S, :].tocsc()
        save(CACHE / f"live_{self.key}.npy", S)
        for k in ("data", "indices", "indptr"): save(CACHE / f"live_{self.key}_{k}.npy", getattr(W, k))
