# fly brain LIF model (same equations/constants as Shiu et al 2024), numpy instead of brian2
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).parent / "data"

# Sugar-sensing gustatory neurons (right hemisphere) and MN9, the motor neuron
# that drives proboscis extension (= "I want to eat this"). IDs from Shiu et al.
SUGAR_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_ID = 720575940660219265

# constants (mV, ms)
V0, VRST, VTH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, F_POI = 0.275, 250


class FlyBrain:
    def __init__(self, dt=0.1, seed=None):
        comp = pd.read_csv(DATA / "2023_03_23_completeness_630_final.csv", index_col=0)
        con = pd.read_parquet(DATA / "2023_03_23_connectivity_630_final.parquet")
        self.ids = comp.index.values
        self.id2i = {f: i for i, f in enumerate(self.ids)}
        self.N = len(self.ids)
        w = con["Excitatory x Connectivity"].values.astype(np.float32) * W_SYN
        # column j = outgoing weights of neuron j
        self.W = sp.csc_matrix(
            (w, (con["Postsynaptic_Index"].values, con["Presynaptic_Index"].values)),
            shape=(self.N, self.N), dtype=np.float32)
        self.n_syn = len(con)
        self.dt = dt
        self.delay = int(round(T_DLY / dt))
        self.rfc_steps = int(round(T_RFC / dt))
        self.rng = np.random.default_rng(seed)
        self.sugar = np.array([self.id2i[f] for f in SUGAR_IDS])
        self.mn9 = self.id2i[MN9_ID]

    def run(self, stim_idx, rate_hz, t_ms=100.0, record=None):
        """Poisson-stimulate `stim_idx` at `rate_hz` for `t_ms`.
        Returns spike counts for neurons in `record` (default: all)."""
        N, dt = self.N, self.dt
        steps = int(round(t_ms / dt))
        v = np.full(N, V0, np.float32)
        g = np.zeros(N, np.float32)
        rfc_left = np.zeros(N, np.int32)
        counts = np.zeros(N, np.int32)
        buf = [np.empty(0, np.int64)] * self.delay   # spike ring buffer
        a_v, a_g = dt / T_MBR, np.float32(np.exp(-dt / TAU))
        p_kick = rate_hz * dt / 1000.0
        kick = W_SYN * F_POI
        stim_idx = np.asarray(stim_idx)
        for t in range(steps):
            arriving = buf[t % self.delay]
            if arriving.size:
                g += np.asarray(self.W[:, arriving].sum(axis=1)).ravel()
            free = rfc_left == 0
            v[free] += a_v * (V0 - v[free] + g[free])
            g[free] *= a_g
            rfc_left[~free] -= 1
            if p_kick > 0 and stim_idx.size:
                hit = stim_idx[self.rng.random(stim_idx.size) < p_kick]
                v[hit] += kick
            spk = np.flatnonzero(v > VTH)
            if spk.size:
                v[spk] = VRST
                g[spk] = 0.0
                rfc_left[spk] = self.rfc_steps
                rfc_left[np.intersect1d(spk, stim_idx)] = 0  # no refractory for Poisson targets
                counts[spk] += 1
            buf[t % self.delay] = spk
        return counts if record is None else counts[record]

    def taste(self, sugar_hz, t_ms=100.0):
        """Feed the fly sugar at `sugar_hz`; return MN9 spikes (proboscis drive)."""
        return int(self.run(self.sugar, sugar_hz, t_ms, record=[self.mn9])[0])
