# mushroom body: kenyon cells -> mbon weights, learned by dopamine
# dopamine = outcome - prediction. PAM fires on reward (>0), PPL1 on pain (<0)
# outcome = material change after the opponent's reply (or mate), in pawns/4
#   python mb.py cells                      -> mb_cells.npz (needs flywire annotations)
#   python mb.py merge out.npz a.npz b.npz  -> average w, sum n
import os, sys
from pathlib import Path
import numpy as np
import chess

HERE = Path(__file__).parent
CELLS = HERE / "mb_cells.npz"
WEIGHTS = HERE / "mb_weights.npz"
LR = 0.05
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATE = 3.0   # mate outcome, pawns/4 units (a queen is 2.25)


def material(board, color):
    # piece values, color minus the other side
    return sum(VALUES[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


class MB:
    def __init__(self, path=WEIGHTS):
        cells = np.load(CELLS)
        self.kc, self.pam, self.ppl1, self.mbon = (cells[k] for k in ("kc", "pam", "ppl1", "mbon"))
        self.path = Path(path)
        self.w, self.n = np.zeros(len(self.kc), np.float32), 0
        if self.path.exists():
            d = np.load(self.path)
            self.w, self.n = d["w"].astype(np.float32), int(d["n"])

    def code(self, counts):
        # kc activity for this position
        return np.sqrt(counts[self.kc].astype(np.float32))

    def predict(self, counts):
        # mbon output = what the fly expects the opponent's reply to do to it
        return float(self.w @ self.code(counts))

    def learn(self, counts, outcome, eaten=0.0):
        # eaten = what the fly's own move already captured, it knew that part
        x = self.code(counts)
        d = float(outcome - eaten - self.w @ x)
        self.w += np.float32(LR * d) * x / (x @ x + 1e-6)
        self.n += 1
        return d

    def save(self, path=None):
        p = Path(path or self.path)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, w=self.w, n=self.n)
        os.replace(tmp, p)


def build_cells():
    # kenyon cells, mbons and dopamine neurons from the flywire annotations (783 ids that match 630)
    import pandas as pd
    a = pd.read_csv(HERE / "data/flywire_annotations/neuron_annotations.tsv", sep="\t", dtype=str)
    ids = pd.read_csv(HERE / "data/2023_03_23_completeness_630_final.csv", index_col=0).index.astype(str)
    id2i = {f: i for i, f in enumerate(ids)}
    a = a[a.root_id.isin(id2i)]
    i = a.root_id.map(id2i).to_numpy()
    ct = a.cell_type.fillna("")
    dan = a.cell_class == "DAN"
    sel = dict(kc=a.cell_class == "Kenyon_Cell", mbon=a.cell_class == "MBON",
               pam=dan & ct.str.startswith("PAM"), ppl1=dan & ct.str.startswith("PPL1"))
    cells = {k: np.sort(i[m.to_numpy(bool)]).astype(np.int32) for k, m in sel.items()}
    # head bristle mechanosensory neurons (BM_*), for the threat senses, strongest output first
    from reservoir import load_w
    W = load_w()
    bm = i[(a.cell_class == "mechanosensory").to_numpy() & ct.str.startswith("BM").to_numpy()]
    out = np.array([np.maximum(W.data[W.indptr[j]:W.indptr[j + 1]], 0).sum() for j in bm])
    cells["bristle"] = bm[np.argsort(-out)].astype(np.int32)
    np.savez(CELLS, **cells)
    print("saved", CELLS.name, {k: len(v) for k, v in cells.items()})


def merge(out, files):
    import glob   # windows shells don't expand runs/*.npz
    files = [g for f in files for g in sorted(glob.glob(f))]
    ds = [np.load(f) for f in files]
    w = np.mean([d["w"] for d in ds], axis=0).astype(np.float32)
    n = int(sum(int(d["n"]) for d in ds))
    np.savez(out, w=w, n=n)
    print(f"merged {len(ds)} files -> {out}: n={n}, |w| mean {np.abs(w).mean():.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "cells": build_cells()
    elif sys.argv[1] == "merge": merge(sys.argv[2], sys.argv[3:])
