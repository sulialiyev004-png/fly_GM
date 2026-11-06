# FlyGM launcher: double-click to play.
# Put this file next to fly_gui.py. First run installs libs + downloads brain data (~90 MB).

import os
import re
import sys
import subprocess
import importlib.util
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

GAME = "fly_gui.py"
DATA_DIR = os.path.join(HERE, "data")
DATA_URL = "https://github.com/philshiu/Drosophila_brain_model/raw/main/"
DATA_FILES = [
    "2023_03_23_completeness_630_final.csv",
    "2023_03_23_connectivity_630_final.parquet",
]

# import name -> pip name (when they differ)
PIP_NAMES = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
}
ALWAYS = ["chess", "scipy", "pandas", "pyarrow", "sklearn", "numpy"]


def fail(msg):
    print("\n[!] " + msg)
    input("\nPress Enter to close...")
    sys.exit(1)


def find_imports():
    """Collect top-level imports from every .py in the folder."""
    local = {f[:-3] for f in os.listdir(HERE) if f.endswith(".py")}
    local |= {d for d in os.listdir(HERE) if os.path.isdir(d)}
    names = set(ALWAYS)
    pat = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", re.M)
    for f in os.listdir(HERE):
        if f.endswith(".py") and f != os.path.basename(__file__):
            try:
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    names |= set(pat.findall(fh.read()))
            except OSError:
                pass
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    return sorted(n for n in names if n not in local and n not in stdlib and n != "__future__")


def install_missing():
    missing = [n for n in find_imports() if importlib.util.find_spec(n) is None]
    if not missing:
        return
    pkgs = [PIP_NAMES.get(n, n) for n in missing]
    print("Installing libraries: " + ", ".join(pkgs))
    r = subprocess.call([sys.executable, "-m", "pip", "install", *pkgs])
    if r != 0:
        fail("pip install failed. Check your internet and try again.")
    importlib.invalidate_caches()


def download(name):
    path = os.path.join(DATA_DIR, name)
    tmp = path + ".part"
    print("Downloading " + name)

    def hook(blocks, bs, total):
        if total > 0:
            done = min(blocks * bs, total)
            pct = done * 100 // total
            bar = "#" * (pct // 4)
            print(f"\r  [{bar:<25}] {pct:3d}%  {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(DATA_URL + name, tmp, hook)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        fail(f"Could not download {name}: {e}")
    os.replace(tmp, path)
    print()


def get_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in DATA_FILES:
        p = os.path.join(DATA_DIR, name)
        if not os.path.exists(p) or os.path.getsize(p) < 1000:
            download(name)


def main():
    print("=== FlyGM ===\n")
    if not os.path.exists(GAME):
        fail(f"{GAME} not found. Put start_flygm.py in the same folder as {GAME}.")
    install_missing()
    get_data()
    print("\nLoading the fly brain, give it ~10 sec...\n")
    r = subprocess.call([sys.executable, GAME])
    if r != 0:
        fail(f"Game crashed (code {r}). Screenshot the error above and send it.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
