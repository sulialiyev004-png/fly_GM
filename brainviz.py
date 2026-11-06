# draws the brain and the neurons that fired
#   BrainView:   flat front view (brain_positions.npz)
#   BrainView3D: rotatable 3d view (brain3d.npz), pure numpy: rotate, perspective, splat, blur
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

    def render(self, counts=None, inputs=None, gain=1.0, flash=None):
        """counts: spikes per neuron (None = brain at rest). inputs: indices of input neurons.
        flash: (neuron indices, rgb) drawn as big dots, e.g. dopamine neurons."""
        img = self.bg.copy()
        if counts is not None:
            act = np.zeros((self.H, self.W), np.float32)
            nz = np.flatnonzero(counts)
            nz = nz[self.valid[nz]]
            np.add.at(act, (self.py[nz], self.px[nz]), counts[nz].astype(np.float32))
            glow = _blur(act, 2) * 0.7 + _blur(act, 6) * 2.5
            heat = np.log1p(glow * gain) / np.log1p(60.0)
            img = img * 0.6 + _fire(heat)
            if inputs is not None:   # input neurons that receive the board: blue dots
                ii = inputs[self.valid[inputs]]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        img[np.clip(self.py[ii] + dy, 0, self.H - 1), np.clip(self.px[ii] + dx, 0, self.W - 1)] = [90, 220, 255]
        if flash is not None:
            ii, rgb = flash
            ii = ii[self.valid[ii]]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    img[np.clip(self.py[ii] + dy, 0, self.H - 1), np.clip(self.px[ii] + dx, 0, self.W - 1)] = rgb
        img = np.clip(img, 0, 255).astype(np.uint8)
        return b"P6 %d %d 255\n" % (self.W, self.H) + img.tobytes()


_LUT = _fire(np.linspace(0, 1, 1024)).astype(np.float32)   # fire palette as a lookup table


def _soften(a):
    # cheap 3x3 binomial blur
    a = a.copy()
    a[1:-1] = a[1:-1] * 0.5 + (a[:-2] + a[2:]) * 0.25
    a[:, 1:-1] = a[:, 1:-1] * 0.5 + (a[:, :-2] + a[:, 2:]) * 0.25
    return a


class BrainView3D:
    """rotatable 3d brain (flywire positions), same render() as BrainView.
    yaw / pitch (radians) and zoom can be changed between frames."""
    DIST = 3.0       # camera distance in brain radii (perspective strength)
    FAR_DIM = 0.35   # farthest neurons are this much dimmer

    def __init__(self, width=700, height=460):
        d = np.load(HERE / "brain3d.npz")
        p = d["q"].astype(np.float32) / 65535 * 2 - 1
        p[:, 1] *= -1                      # flywire y points ventral, flip so dorsal is up
        self.valid = d["valid"]
        p -= (p[self.valid].max(0) + p[self.valid].min(0)) / 2
        self.p = p
        allp = p[self.valid]
        self.bgp = np.ascontiguousarray(allp[::2])   # every 2nd neuron is enough for the silhouette
        self.r = float(np.sqrt((allp ** 2).sum(1)).max())
        self.W, self.H = width // 2 * 2, height // 2 * 2
        self.scale = 0.47 * self.W         # brain x spans -1..1
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0
        # silhouette density -> colour in one lookup, log curve baked in, fixed so rotating doesn't flicker
        self.dmax = float(_soften(self._splat(self.bgp, None, 1)).max())
        v = np.log1p(np.linspace(0, self.dmax, 4096)) / np.log1p(self.dmax)
        self.bg_lut = (v[:, None] * [60, 85, 130] + [10, 10, 14]).astype(np.float32)

    def _project(self, pts):
        # rotate (yaw around the vertical axis, then pitch) + perspective -> pixel x, y, inside mask, depth dim
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        R = np.array([[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]], np.float32)
        q = pts @ R.T
        z = q[:, 2] / self.r
        f = (self.scale * self.zoom * self.DIST) / (self.DIST + z)
        x = (self.W / 2 + q[:, 0] * f).astype(np.int32)
        y = (self.H / 2 - q[:, 1] * f).astype(np.int32)
        ok = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        return x, y, ok, 1 - self.FAR_DIM * (z + 1) / 2

    def _splat(self, pts, w, s):
        # sum of weights (times depth dim) per pixel, at 1/s resolution
        x, y, ok, dim = self._project(pts)
        w = dim if w is None else w * dim
        W, H = self.W // s, self.H // s
        return np.bincount((y[ok] // s) * W + x[ok] // s, w[ok], W * H).astype(np.float32).reshape(H, W)

    def _dots(self, img, ii, rgb, r):
        # solid (2r+1)^2 dots, dimmer when far
        x, y, ok, dim = self._project(self.p[ii])
        pix, dim = y[ok] * self.W + x[ok], dim[ok]
        flat = img.reshape(-1, 3)
        col = np.asarray(rgb, np.float32) * dim[:, None]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                flat[np.clip(pix + dy * self.W + dx, 0, len(flat) - 1)] = col

    def _add_half(self, img, half):
        # add a half resolution rgb layer (glows are blurry anyway, 4x cheaper)
        img.reshape(self.H // 2, 2, self.W // 2, 2, 3)[...] += half[:, None, :, None, :]

    def render(self, counts=None, inputs=None, gain=1.0, flash=None):
        """counts: spikes (or fading activity) per neuron, None = at rest. inputs: input neuron indices.
        flash: (neuron indices, rgb), drawn as glowing dots, e.g. dopamine neurons."""
        dens = _soften(self._splat(self.bgp, None, 1)) * (self.zoom ** 2 * 4095 / self.dmax)
        img = self.bg_lut[np.minimum(dens, 4095).astype(np.int32)]
        if counts is not None:
            nz = np.flatnonzero(counts > 0.05)
            nz = nz[self.valid[nz]]
            if nz.size:
                a = self._splat(self.p[nz], counts[nz].astype(np.float32) * (self.zoom / 4), 2)
                glow = _blur(a, 1) * 0.7 + _blur(a, 3) * 2.5
                heat = np.log1p(glow * gain) * (1023 / np.log1p(60.0))
                # background dims as activity rises (smooth, no jump when it fades out)
                img *= 0.6 + 0.4 * np.exp(-float(counts[nz].sum()) / 2000.0)
                self._add_half(img, _LUT[np.minimum(heat, 1023).astype(np.int32)])
            if inputs is not None:   # input neurons that receive the board: blue dots
                self._dots(img, inputs[self.valid[inputs]], (90, 220, 255), 1)
        if flash is not None:
            ii, rgb = flash
            ii = ii[self.valid[ii]]
            halo = _blur(self._splat(self.p[ii], None, 2), 3)
            halo = np.minimum(halo / max(float(halo.max()), 1e-6), 1)
            self._add_half(img, halo[..., None] * (np.asarray(rgb, np.float32) * 0.8))
            self._dots(img, ii, rgb, 2)
        np.minimum(img, 255, out=img)
        return b"P6 %d %d 255\n" % (self.W, self.H) + img.astype(np.uint8).tobytes()
