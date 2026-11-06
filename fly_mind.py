# fly mind: poke its senses and read what the brain does, or let it dream
# run: python fly_mind.py      (dreams are saved to the dreams/ folder: gif + wav + poster)
import threading, time, sys, tkinter as tk
from pathlib import Path
import numpy as np
from brainsim import Brain
from mindviz import BrainView
import mind

HERE = Path(__file__).parent
BG, PANEL, TXT, DIM, ACC = "#161512", "#262421", "#bababa", "#8b8987", "#3692e7"
UI = "Segoe UI"
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None


class App:
    def __init__(self, root):
        self.root = root
        root.title("Fly Mind - read a fruit fly's brain")
        root.configure(bg=BG)
        left = tk.Frame(root, bg=BG); left.grid(row=0, column=0, sticky="n", padx=(14, 6), pady=14)
        tk.Label(left, text="Give it a sense", fg=TXT, bg=BG, font=(UI, 12, "bold")).pack(anchor="w", pady=(0, 6))
        mk = dict(bg="#302e2c", fg=TXT, activebackground=ACC, activeforeground="#fff", relief="flat",
                  font=(UI, 10), anchor="w", padx=10, pady=5, bd=0, cursor="hand2", width=24)
        self.buttons = []
        for key, (label, _, _) in mind.STIMULI.items():
            b = tk.Button(left, text=label, command=lambda k=key: self.start(self.stimulus, k), **mk)
            b.pack(fill="x", pady=2); self.buttons.append(b)
        tk.Label(left, text="Or let it sleep", fg=TXT, bg=BG, font=(UI, 12, "bold")).pack(anchor="w", pady=(14, 6))
        b = tk.Button(left, text="Dream (3 sec of brain)", command=lambda: self.start(self.dream), **mk)
        b.pack(fill="x", pady=2); self.buttons.append(b)
        self.play_btn = tk.Button(left, text="Play last dream music", command=self.play, state="disabled", **mk)
        self.play_btn.pack(fill="x", pady=2)

        right = tk.Frame(root, bg=PANEL); right.grid(row=0, column=1, sticky="n", padx=(6, 14), pady=14)
        self.view = BrainView(640)
        self.img = tk.PhotoImage(data=self.view.render(), format="PPM")
        self.lbl = tk.Label(right, image=self.img, bg=PANEL, bd=0); self.lbl.pack(padx=12, pady=(12, 4))
        self.cap = tk.Label(right, text="loading the brain...", fg=DIM, bg=PANEL, font=(UI, 10)); self.cap.pack(anchor="w", padx=12)
        self.text = tk.Text(right, width=78, height=12, bg=PANEL, fg=TXT, relief="flat", wrap="word",
                            font=(UI, 10), highlightthickness=0, padx=4)
        self.text.pack(padx=12, pady=(8, 12))
        self.text.tag_configure("head", font=(UI, 11, "bold"), foreground="#ffffff")
        self.text.tag_configure("fly", font=(UI, 11, "italic"), foreground="#f6d743")

        self.brain, self.busy, self.frame, self.shown, self.done = None, True, None, None, None
        self.last_wav = None
        threading.Thread(target=self.load, daemon=True).start()
        self.poll()

    def load(self):
        self.brain = Brain(seed=int(time.time()))
        self.busy = False
        self.frame = (self.view.render(), "ready. pick a sense on the left")

    def start(self, fn, *args):
        if self.busy or self.brain is None: return
        self.busy = True
        for b in self.buttons: b.config(state="disabled")
        threading.Thread(target=fn, args=args, daemon=True).start()

    def finish(self, head, lines, extra=None):
        self.done = (head, lines, extra)

    # senses
    def stimulus(self, key):
        label, group, rate = mind.STIMULI[key]
        stim = self.brain.groups[group]
        t_ms = 200
        self.done = None
        self.set_text(f"{label}: poking {len(stim)} sensory neurons at {rate} Hz for {t_ms} ms...", [])

        def on_frame(c, t):
            self.frame = (self.view.render(c, stim, gain=6.0), f"{label} | {t:.0f} ms")
        counts = self.brain.run(stim, rate, t_ms, on_frame=on_frame, every=50)
        self.frame = (self.view.render(counts, stim), f"{label} | all {t_ms} ms together")
        r = mind.read(self.brain, counts, group, t_ms)
        head, lines = mind.narrate(r)
        self.finish(head, lines, mind.llm_monologue(label, head, lines))

    # dreams
    def dream(self, seconds=3.0):
        seed = int(time.time()) % 100000
        plan = mind.dream_plan(seconds, seed)
        restim = mind.make_restim(self.brain, plan, seed)
        names = {g: lab for _, (lab, g, _) in mind.STIMULI.items()}
        G = self.brain.groups
        frames, groups, total = [], [], np.zeros(self.brain.N, np.int64)
        gif_view = BrainView(420) if Image is not None else None
        self.done = None
        self.set_text(f"dream #{seed}: {len(plan)} scenes of random senses, {seconds:g} seconds of brain time...", [])

        def scene_at(t):
            s = [p for p in plan if p[0] <= t][-1]
            return names.get(s[1], "silence") if s[1] else "silence"

        def on_frame(c, t):
            total[:] += c
            groups.append({g: int(c[G[g]].sum()) for g in mind.MUSIC})
            self.frame = (self.view.render(c, None, gain=8.0), f"dream #{seed} | {t:.0f} ms | {scene_at(t)}")
            if gif_view is not None:   # small frames for the gif, kept compressed
                p = gif_view.render(c, None, gain=8.0)
                frames.append(Image.frombytes("RGB", (gif_view.W, gif_view.H), p[p.index(b"255\n") + 4:]).quantize(96))
        self.brain.run(np.empty(0, int), 0, seconds * 1000, on_frame=on_frame, every=100, restim=restim)

        out = HERE / "dreams"; out.mkdir(exist_ok=True)
        wav = out / f"dream_{seed}.wav"
        mind.save_wav(wav, mind.music(groups))
        self.last_wav = wav
        saved = [wav.name]
        if Image is not None:
            frames[0].save(out / f"dream_{seed}.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
            saved.append(f"dream_{seed}.gif")
            self.poster(out / f"dream_{seed}.png", total, seed, plan, names, seconds)
            saved.append(f"dream_{seed}.png")
        self.frame = (self.view.render(total, None, gain=0.5), f"dream #{seed} | long exposure of all {seconds:g} seconds")
        scenes = ", ".join(names.get(g, "silence") if g else "silence" for _, g, _, _ in plan)
        head = f"Dream #{seed}: {int((total > 0).sum()):,} neurons fired, {int(total.sum()):,} spikes."
        self.finish(head, [f"Scenes: {scenes}.", f"Saved to dreams/: {', '.join(saved)}"])

    def poster(self, path, total, seed, plan, names, seconds):
        v = BrainView(1600)
        ppm = v.render(total, None, gain=0.5)
        brain = Image.frombytes("RGB", (v.W, v.H), ppm[ppm.index(b"255\n") + 4:])
        W, H = v.W + 160, v.H + 360
        img = Image.new("RGB", (W, H), (10, 10, 14))
        img.paste(brain, (80, 120))
        d = ImageDraw.Draw(img)
        def font(sz):
            for f in ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]:
                try: return ImageFont.truetype(f, sz)
                except OSError: pass
            return ImageFont.load_default()
        d.text((80, 40), f"DROSOPHILA DREAM #{seed}", fill=(255, 220, 120), font=font(56))
        scenes = ", ".join(dict.fromkeys(names.get(g, "silence") if g else "silence" for _, g, _, _ in plan))
        d.text((80, v.H + 170), f"{seconds:g} seconds of a simulated fruit fly brain  |  {int((total > 0).sum()):,} neurons  |  {int(total.sum()):,} spikes",
               fill=(186, 186, 186), font=font(30))
        d.text((80, v.H + 220), f"dreamed of: {scenes}", fill=(139, 137, 135), font=font(26))
        d.text((80, v.H + 270), "FlyWire connectome, 127,400 neurons", fill=(90, 90, 90), font=font(22))
        img.save(path)

    def play(self):
        if self.last_wav is None: return
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(self.last_wav), winsound.SND_FILENAME | winsound.SND_ASYNC)

    # ui loop
    def set_text(self, head, lines, extra=None):
        def do():
            self.text.delete("1.0", "end")
            self.text.insert("end", head + "\n\n", "head")
            for l in lines: self.text.insert("end", "- " + l + "\n")
            if extra: self.text.insert("end", "\n" + extra + "\n", "fly")
        self.root.after(0, do)

    def poll(self):
        fr = self.frame
        if fr is not None and fr is not self.shown:
            self.shown = fr
            self.img = tk.PhotoImage(data=fr[0], format="PPM")
            self.lbl.config(image=self.img); self.cap.config(text=fr[1])
        if self.done is not None:
            head, lines, extra = self.done; self.done = None
            self.set_text(head, lines, extra)
            self.busy = False
            for b in self.buttons: b.config(state="normal")
            if self.last_wav is not None and sys.platform == "win32": self.play_btn.config(state="normal")
        self.root.after(50, self.poll)


if __name__ == "__main__":
    root = tk.Tk(); root.resizable(False, False)
    App(root); root.mainloop()
