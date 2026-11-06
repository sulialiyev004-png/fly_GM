# brain picture for fly_mind.py (front view, flywire coords)
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def _blur(a, r):
    """Fast box blur; two passes ~ gaussian."""
    k = 2 * r + 1
    for _ in range(2):
        c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0))), axis=0); a = (c[k:] - c[:-k]) / k
        c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r))), axis=1); a = (c[:, k:] - c[:, :-k]) / k
    return a


# fire palette: black -> purple -> red -> orange -> yellow -> white
_STOPS = np.array([[0, 0, 0], [70, 20, 110], [200, 30, 70], [255, 120, 20], [255, 220, 60], [255, 255, 235]], np.float32)


def _fire(t):
    t = np.clip(t, 0, 1) * (len(_STOPS) - 1)
    i = np.minimum(t.astype(int), len(_STOPS) - 2)
    f = (t - i)[..., None]
    return _STOPS[i] * (1 - f) + _STOPS[i + 1] * f


class BrainView:
    def __init__(self, width=560):
        d = np.load(HERE / "brain_positions.npz")
        self.W = width
        self.H = int(width / float(d["aspect"])) + 8
        pad = 6
        self.px = np.clip((d["xy"][:, 0] * (self.W - 2 * pad) + pad).astype(int), 0, self.W - 1)
        self.py = np.clip((d["xy"][:, 1] * (self.H - 2 * pad) + pad).astype(int), 0, self.H - 1)
        self.valid = d["valid"]
        # background: density of all neurons = brain silhouette
        dens = np.zeros((self.H, self.W), np.float32)
        np.add.at(dens, (self.py[self.valid], self.px[self.valid]), 1)
        dens = np.log1p(_blur(dens, 1))
        dens /= dens.max()
        self.bg = (np.stack([dens * 60, dens * 85, dens * 130], -1) + np.array([10, 10, 14], np.float32))

    def render(self, counts=None, inputs=None, gain=1.0):
        """counts: spikes per neuron (None = brain at rest). inputs: indices of input neurons."""
        img = self.bg.copy()
        if counts is not None:
            act = np.zeros((self.H, self.W), np.float32)
            nz = np.flatnonzero(counts)
            nz = nz[self.valid[nz]]
            np.add.at(act, (self.py[nz], self.px[nz]), counts[nz].astype(np.float32))
            halo = _blur(act, 2) * 0.7 + _blur(act, 6) * 2.5
            heat = np.log1p(halo * gain) / np.log1p(60.0)
            img = img * 0.6 + _fire(heat)
            if inputs is not None:   # stimulated sensory neurons: blue dots
                ii = inputs[self.valid[inputs]]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        img[np.clip(self.py[ii] + dy, 0, self.H - 1), np.clip(self.px[ii] + dx, 0, self.W - 1)] = [90, 220, 255]
        img = np.clip(img, 0, 255).astype(np.uint8)
        return b"P6 %d %d 255\n" % (self.W, self.H) + img.tobytes()
