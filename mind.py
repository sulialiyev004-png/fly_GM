# reading the fly's mind + making its dreams
import os, json, wave, random, urllib.request
import numpy as np

STIMULI = {   # button label, neuron group, poisson rate
    "sugar": ("Sugar", "in_sugar", 100),
    "bitter": ("Bitter taste", "in_bitter", 100),
    "water": ("Water", "in_water", 100),
    "vinegar": ("Smell: vinegar (fruit)", "in_vinegar", 40),
    "mold": ("Smell: mold (geosmin)", "in_mold", 60),
    "co2": ("Smell: CO2", "in_co2", 40),
    "looming": ("Something flying at her", "in_looming", 100),
    "sound": ("Sound (wing song)", "in_sound", 100),
    "heat": ("Heat", "in_heat", 40),
    "cold": ("Cold", "in_cold", 40),
    "pheromone": ("Pheromone", "in_pheromone", 40),
}

# what each group means, for the read-out
READ = ["feeding", "escape", "backward", "steering", "reward", "punish", "memory", "memory_out",
        "innate_smell", "smell", "compass", "central_complex", "clock", "hormones", "motor", "descending"]


def read(brain, counts, stim_group, t_ms):
    """per group: fraction of neurons that fired and mean rate (Hz), the stimulus neurons themselves left out"""
    G, out = brain.groups, {}
    skip = G[stim_group] if stim_group else np.empty(0, int)
    for k in READ:
        idx = np.setdiff1d(G[k], skip)
        if len(idx) == 0: continue
        c = counts[idx]
        out[k] = dict(frac=float((c > 0).mean()), hz=float(c.sum() / len(idx) / (t_ms / 1000)), spikes=int(c.sum()))
    out["_total"] = dict(neurons=int((counts > 0).sum()), spikes=int(counts.sum()))
    return out


def narrate(r):
    """turn the numbers into plain sentences. only says things the neurons actually did"""
    f = lambda k: r.get(k, {}).get("frac", 0.0)
    lines, approach, avoid = [], 0, 0
    if r.get("feeding", {}).get("spikes", 0) >= 2:
        lines.append("MN9, the motor neuron that pushes out its proboscis, is firing. It wants to EAT this."); approach += 2
    if r.get("escape", {}).get("spikes", 0) > 0:
        lines.append("The giant fiber fired. That's the escape reflex: it would jump and take off right now."); avoid += 2
    if r.get("backward", {}).get("spikes", 0) > 0:
        lines.append("Moonwalker neurons (MDN) are on, it wants to back away."); avoid += 1
    storm = f("smell") > 0.5 and f("memory") > 0.3
    if storm:
        lines.append(f"Smell storm: {f('smell'):.0%} of smell neurons and {f('memory'):.0%} of the mushroom body "
                     "lit up at once. (this simple model is missing some of the brakes a real fly has)")
    else:
        if f("smell") > 0.05: lines.append(f"Smell centers are working ({f('smell'):.0%} of antennal lobe projection neurons).")
        if f("innate_smell") > 0.05: lines.append("Lateral horn is active: an inborn, hard-wired reaction to this smell.")
        if f("memory") > 0.05: lines.append(f"Mushroom body is active ({f('memory'):.0%} of Kenyon cells): it's checking this against memories.")
    if f("reward") > 0.1: lines.append(f"Reward dopamine (PAM neurons, {f('reward'):.0%}) is firing, this feels good."); approach += 1
    if f("punish") > 0.3: lines.append(f"Punishment dopamine (PPL1, {f('punish'):.0%}) is firing, this feels bad."); avoid += 1
    if f("steering") > 0.1: lines.append("Steering neurons are active, it's turning.")
    if f("compass") > 0.1: lines.append("Compass neurons (EPG) are updating its sense of direction.")
    if f("clock") > 0.1: lines.append("Even its internal clock neurons twitched.")
    if f("hormones") > 0.1: lines.append("Hormone-releasing neurons responded.")
    if f("motor") > 0.1: lines.append(f"{f('motor'):.0%} of head motor neurons are moving (mouth, neck).")
    if f("descending") > 0.02: lines.append(f"{f('descending'):.0%} of descending neurons are sending commands down to the body.")
    if not lines:
        lines.append("Almost nothing reached the brain's deeper centers. It barely noticed.")
    mood = "wants it" if approach > avoid else "wants to get away" if avoid > approach else "neutral / curious"
    t = r["_total"]
    head = f"{t['neurons']:,} neurons fired, {t['spikes']:,} spikes. Overall: {mood}."
    return head, lines


def llm_monologue(stim_label, head, lines):
    """optional: if FLY_LLM is set (an ollama model name), let a local LLM speak as the fly, using only these facts"""
    model = os.environ.get("FLY_LLM")
    if not model: return None
    prompt = (f"You are a fruit fly. Stimulus: {stim_label}. What your neurons did:\n- " + "\n- ".join([head] + lines) +
              "\nSay what you feel in 2 short first-person sentences. Use only these facts, no new ones.")
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate",
                                     data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["response"].strip()
    except Exception:
        return None


# dreams

DREAM_SCENES = ["in_sugar", "in_bitter", "in_water", "in_vinegar", "in_mold", "in_co2",
                "in_looming", "in_sound", "in_heat", "in_cold", "in_pheromone", None]
MUSIC = {   # group -> note (hz), a minor pentatonic
    "punish": 110.0, "descending": 164.8, "innate_smell": 392.0, "steering": 293.7, "smell": 523.3,
    "memory": 659.3, "feeding": 587.3, "escape": 82.4, "reward": 1046.5, "compass": 880.0,
}


def dream_plan(seconds, seed):
    """random 'scenes': short bursts of random senses, sometimes silence"""
    rng = random.Random(seed)
    t, plan = 0.0, []
    while t < seconds * 1000:
        g = rng.choice(DREAM_SCENES)
        dur = rng.uniform(120, 320)
        plan.append((t, g, rng.uniform(0.3, 0.8), rng.choice([20, 40, 60, 100])))
        t += dur
    return plan


def make_restim(brain, plan, seed):
    rng = np.random.default_rng(seed)
    state = {"i": -1}

    def restim(t_ms):
        i = state["i"]
        while i + 1 < len(plan) and plan[i + 1][0] <= t_ms: i += 1
        if i == state["i"]: return None
        state["i"] = i
        _, g, part, rate = plan[i]
        if g is None: return (np.empty(0, int), 0)
        n = brain.groups[g]
        return (rng.choice(n, max(1, int(len(n) * part)), replace=False), rate)
    return restim


def music(frames_groups, ms_per_frame=40, sr=44100):
    """frames_groups: list of {group: spikes} per frame -> stereo-free 16-bit audio array"""
    n = int(len(frames_groups) * ms_per_frame / 1000 * sr) + sr * 2
    out = np.zeros(n, np.float32)
    env_t = np.arange(int(sr * 1.2)) / sr
    for i, fg in enumerate(frames_groups):
        at = int(i * ms_per_frame / 1000 * sr)
        for g, hz in MUSIC.items():
            s = fg.get(g, 0)
            if s <= 0: continue
            amp = min(1.0, np.log1p(s) / 8)
            tone = (np.sin(2 * np.pi * hz * env_t) + 0.3 * np.sin(4 * np.pi * hz * env_t)) * np.exp(-env_t * 4) * amp
            out[at:at + len(tone)] += tone[:len(out) - at]
    d = int(0.23 * sr)   # little echo
    for _ in range(3): out[d:] += 0.35 * out[:-d]
    peak = np.abs(out).max()
    if peak > 0: out = out / peak * 0.9
    return (out * 32767).astype(np.int16)


def save_wav(path, audio, sr=44100):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())
