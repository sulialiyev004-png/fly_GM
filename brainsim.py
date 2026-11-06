# whole fly brain for sensory experiments (mind reader + dreams)
# same LIF model as the chess brain, with two fixes so senses don't all blur together:
#  - sensory neurons only fire from their stimulus (brain inputs onto their axons are modulatory)
#  - dopamine / serotonin / octopamine are slow modulators, so no fast synapses from them
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

HERE = Path(__file__).parent
V0, VRST, VTH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, KICK = 0.275, 0.275 * 250


class Brain:
    def __init__(self, dt=0.1, seed=None):
        con = pd.read_parquet(HERE / "data/2023_03_23_connectivity_630_final.parquet",
                              columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        self.N = N = 127400
        pre, post = con.Presynaptic_Index.values, con.Postsynaptic_Index.values
        w = con["Excitatory x Connectivity"].values.astype(np.float32) * W_SYN
        sens = np.zeros(N, bool); sens[np.load(HERE / "sensory_neurons.npy")] = True
        mod = np.zeros(N, bool); mod[np.load(HERE / "modulatory_neurons.npy")] = True
        keep = ~sens[post] & ~mod[pre]
        self.W = sp.csc_matrix((w[keep], (post[keep], pre[keep])), shape=(N, N))
        self.groups = dict(np.load(HERE / "neuron_groups.npz"))
        self.dt = dt
        self.delay = int(round(T_DLY / dt))
        self.rfc = int(round(T_RFC / dt))
        self.rng = np.random.default_rng(seed)

    def run(self, stim, rate_hz, t_ms, on_frame=None, every=100, restim=None):
        """poke `stim` neurons with poisson spikes for t_ms. returns spike counts.
        on_frame(recent_counts, t_ms) every `every` steps. restim(t_ms) may return new neurons to poke."""
        W, N, dt = self.W, self.N, self.dt
        stim = np.asarray(stim, np.int64)
        v = np.full(N, V0, np.float32)
        g = np.zeros(N, np.float32)
        rl = np.zeros(N, np.int16)
        counts = np.zeros(N, np.int32)
        last = counts.copy()
        buf = [np.empty(0, np.int64)] * self.delay
        a_v, a_g = np.float32(dt / T_MBR), np.float32(np.exp(-dt / TAU))
        for t in range(int(round(t_ms / dt))):
            if restim is not None:
                new = restim(t * dt)
                if new is not None: stim, rate_hz = np.asarray(new[0], np.int64), new[1]
            arr = buf[t % self.delay]
            if arr.size:   # synapses of neurons that spiked 1.8 ms ago
                st, ln = W.indptr[arr], W.indptr[arr + 1] - W.indptr[arr]
                tot = ln.sum()
                if tot:
                    k = np.repeat(st - np.concatenate(([0], np.cumsum(ln)[:-1])), ln) + np.arange(tot)
                    g += np.bincount(W.indices[k], weights=W.data[k], minlength=N).astype(np.float32)
            free = rl == 0
            v += np.where(free, a_v * (V0 - v + g), 0)
            g = np.where(free, g * a_g, g)
            rl -= (~free).astype(np.int16)
            if stim.size and rate_hz > 0:
                v[stim[self.rng.random(stim.size) < rate_hz * dt / 1000.0]] += KICK
            spk = np.flatnonzero(v > VTH)
            if spk.size:
                v[spk] = VRST; g[spk] = 0.0; rl[spk] = self.rfc
                rl[np.intersect1d(spk, stim)] = 0
                counts[spk] += 1
            buf[t % self.delay] = spk
            if on_frame is not None and (t + 1) % every == 0:
                on_frame(counts - last, (t + 1) * dt); last = counts.copy()
        return counts
