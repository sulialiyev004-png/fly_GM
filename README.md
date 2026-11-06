# FlyGM

chess played by a simulated fruit fly brain (drosophila, full connectome from FlyWire, 127k neurons / 14.7M connections). brain model is the LIF model from Shiu et al 2024, rewritten in numpy so it runs faster than brian2.

## how it works

the board gets encoded into 768 input neurons (piece x square). on top of that the fly has threat senses, like nociceptors, that go into real mechanosensory bristle neurons from the flywire annotations. the whole brain runs for 50ms, then a linear readout on the spike counts says how good the position is. 8 flies with different input wiring taste every position and their feelings are averaged (a swarm). for every legal move the swarm tastes the position after it and plays the best one. stockfish isn't used anywhere in play.

it's still weak, and the brain itself is the weak part: reading the board directly (no fly) predicts who's better more accurately than the fly brain does (see round 2). what made the fly stronger is a better teacher, the senses and averaging many flies. searching the opponent's replies made it worse (see bench), so it's off by default (`DEPTH` in fly_player.py).

there's also `flygm.py`, a hybrid mode where stockfish gives the fly its top 5 moves as sugar (better move = sweeter) and the fly picks whichever one makes its proboscis motor neuron (MN9) fire the most. that one plays strong but obviously that's stockfish.

### round 2: making it stronger (measured on held-out positions)

stockfish isn't installed here, so the teacher is material after a capture-only search (quiescence with python-chess: up to 8 plies of captures, all evasions when in check, mate = 40 pawns). dataset (`lab.py`): 6197 positions from 823 games of greedy bot / random / old fly moves (half of them one random move further, like the candidate moves the fly tastes), plus every 3rd position of the 18 fly-vs-greedy games of the last bench. split by game: 4528 train, 1669 held-out. readouts are ridge regression on sqrt spike counts (target tanh(material/4)), alpha picked by 5-fold cv grouped by game inside the train split (leave-one-out picked alphas that memorize games and looked better than they were). brain responses are cached once per brain setup, readout changes don't re-simulate.

- corr: held-out prediction vs teacher
- who's better: sign right, on the 1493 held-out positions where the teacher isn't equal
- tactical: same, only the 559 of those where a capture changes the material count

| stage | setup | corr | who's better | tactical | kept |
|---|---|---|---|---|---|
| no fly | raw board (768 piece-square) | 0.843 | 91.4% | 88.7% | baseline |
| no fly | board + threat senses | 0.858 | 93.1% | 91.6% | baseline |
| 0 | old readout (fly_readout.npz, stockfish labels) | 0.513 | 71.3% | 64.9% | |
| 0 | same brain, readout trained on the teacher | 0.664 | 78.0% | 75.3% | yes |
| 1 | + threat senses into bristle neurons | 0.721 | 81.7% | 80.7% | yes |
| 2 | spike timing: 5 bins of 10ms instead of 50ms total | 0.643 | 76.2% | 74.2% | no |
| 2 | shorter runs: 10 / 20 / 30 / 40 ms | 0.604 / 0.653 / 0.680 / 0.697 | | | no |
| 2 | other single flies (rewired, w1..w7) | 0.722 - 0.746 | 81-83% | 79-84% | |
| 2 | swarm of 2 | 0.765 | 83.7% | 83.9% | |
| 2 | swarm of 4 | 0.786 | 84.7% | 84.1% | yes |
| 2 | swarm of 8 | **0.799** | **86.6%** | **85.7%** | yes, best |
| 3 | APL inhibition (kenyon cells 69% -> 4.9% active) | 0.706 | 80.8% | 80.7% | no |
| check | only the fly's input neurons (board + bristle) | 0.752 | 82.3% | 81.0% | |
| check | every neuron except the input ones | 0.657 | 78.0% | 75.7% | |

stage 1 answer: **no, the fly does not beat "board + senses only"** (0.721 vs 0.858 for one fly, 0.799 for 8). the senses help the fly a lot (+0.06), but a linear readout of the brain loses information: the input neurons alone read better than the whole brain, and the rest of the brain alone is worse. poisson spiking makes the counts noisy and the downstream neurons mix that noise in. averaging flies with different wiring cancels some of it, that's why the swarm works.

what didn't help:
- spike timing (5 bins): more features, more noise, worse. shorter runs are worse too, 50ms is best of what was tried.
- APL: k-winners-take-all on the kenyon cells (the first 5% to fire win, the APL shuts the rest) does make them sparse like a real fly (69% -> 4.9% active). but the eval got a bit worse (0.721 -> 0.706), and kenyon cells still barely know what the opponent can capture: corr with the best capture 0.113 without APL, 0.129 with. not a clear improvement, so no new dopamine self-play this round.
- the old mushroom body memory (mb_weights_selfplay.npz, 1643 lessons) learned an average, see last round. the gui memory now starts from zero.

### search (stage 4)

negamax alpha-beta, captures first (biggest victim, cheapest attacker), then promotions and checks. the fly tastes every move first, only its best 8 get the alpha-beta (beam). mate / stalemate are game rules, every other leaf is the swarm's feel. with worker processes (the gui) all leaves of that tree are tasted in parallel first, then the alpha-beta runs on the cache; same move as the sequential search, just more brain runs. a position always feels the same (input spikes are seeded by the position), so nothing is tasted twice.

| search | flies | brain runs per move | time per move |
|---|---|---|---|
| none (1 ply) | 1 | 17 | 3s, one core |
| 2 plies | 1 | 92 | 8s, one core |
| 3 plies | 1 | 452 | 37s, one core |
| 2 plies | 4 | 764 | 6s, 18 workers |
| 2 plies | 8 | 1523 | 12s, 18 workers |

(sample middlegame positions with ~20 legal moves, real games have more.) 3 plies with 8 flies would be minutes per move. with an exact material eval instead of the brain the same search does what it should (depth 1 grabs a pawn defended by a pawn with the queen, depth 2 and 3 don't), so the code is fine, the problem is the eval (see bench).

### bench (stage 5)

20 games vs the greedy bot (mates if it can, else takes the biggest piece) and 20 vs random moves per mode, colors alternate, 40 move cap then material decides (3+ points = win). to fit in the time the search modes use 2 flies (w1s + w2s, held-out corr 0.765), not 8. material per move = what the fly's move + the reply did to material, blunder = -3 or worse. logs in `runs/bench2/`.

| mode | flies | search | vs greedy +/=/- | score | material/move | blunders | vs random +/=/- | score | material/move | blunders | brain runs/move |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (old fly) | 1, no senses | none | 2/2/16 | 0.15 | -0.83 | 19.9% | 17/1/2 | 0.88 | +0.19 | 5.4% | 19-32 |
| fly | 2 | none | 8/2/10 | **0.45** | +0.02 | **10.2%** | 18/2/0 | 0.95 | +0.41 | 1.7% | 54-67 |
| fly+1 | 2 | 2 plies | 6/2/12 | 0.35 | -0.32 | 13.8% | 18/2/0 | 0.95 | +0.45 | 2.7% | 235-256 |
| fly+2 | 2 | 3 plies | not run | | | | | | | | ~1200 |
| **fly8 (default)** | 8 | none | **15/1/4** | **0.78** | **+0.21** | **5.6%** | not run | | | | 245 |

- the default (8 flies, no search) is the strongest: vs greedy score 0.78 vs 0.45 for 2 flies (p=0.04) and 0.15 for the old fly (p<0.001). it was run vs greedy only, at the end, since vs random every new mode already scores ~0.95.
- new fly vs old fly: clearly better vs greedy (score +0.30, permutation p=0.04; material per move +0.85, p<0.001) and keeps more material vs random (+0.23 per move, p=0.004, score 0.95 vs 0.88 not significant).
- search: **2 plies played worse than no search** vs greedy (material per move -0.34, p=0.013; score 0.35 vs 0.45, p=0.6) and the same vs random, at ~4x the brain runs. the fly's feeling is noisy (corr ~0.77), and taking the worst of ~30 noisy replies mostly finds noise: every move looks bad for a random reason, so the real threats get lost.
- 3 plies: one 40 move game would take ~2h of a core (~1200 brain runs per move), the two games started didn't finish, so it isn't benched.

### learning (mushroom body)

the fly also has a mushroom body (`mb.py`). the real kenyon cells (5095 of them, from the flywire annotations) fire while it tastes a move, and a learned kenyon cell -> mbon weight vector says what it expects the opponent's reply to cost. after the reply it gets dopamine = outcome - prediction (outcome = material change, or mate). PAM neurons = reward (better than expected), PPL1 = pain (worse). the gui keeps learning from your games into `mb_weights.npz` (starts empty, the self-play memory from last round is in `mb_weights_selfplay.npz`). the memory's guess is added to how each move feels.

honest note: the kenyon cells barely encode who can capture what (see APL above), so the memory mostly learns an average. after 2h of self-play on 19 cores (1643 lessons) it expected to lose ~half a pawn after every move. bench vs greedy last round (6 games each, 60 move cap): base 0/1/5, ahead 1/1/4, ahead+mem 0/0/6, none of it significant.

### speed

only 21-27k of the 127k neurons ever get near threshold (within 3mV, on 200 sample boards), so the sim only runs those ("live set", cached in `data/cache/live_*`). same spikes as the full brain (99.95% identical counts, 14 of 300k spikes missed on test boards), 3.4x faster: one 50ms run is ~0.08s on one core.

## run

in the flygm folder (tested on python 3.11):

> **Note:** the `data/` folder is **not included in this GitHub repo** (the connectome is ~115 MB and the `data/cache/` it builds is ~300 MB, both too large to commit). It is excluded via `.gitignore`. Step 1 below downloads the connectome; `data/cache/*.npy` is rebuilt automatically on the first run of `fly_gui.py`. The annotation file for `mb.py cells` comes from github.com/flyconnectome/flywire_annotations (see below). Or just run `python start_flygm.py`, which downloads the brain data for you.

1. get the 2 brain files (~90MB, from github.com/philshiu/Drosophila_brain_model) into `data/`
```
curl -L --create-dirs -o data/2023_03_23_completeness_630_final.csv https://github.com/philshiu/Drosophila_brain_model/raw/main/2023_03_23_completeness_630_final.csv
curl -L --create-dirs -o data/2023_03_23_connectivity_630_final.parquet https://github.com/philshiu/Drosophila_brain_model/raw/main/2023_03_23_connectivity_630_final.parquet
```
in windows powershell type `curl.exe` instead of `curl` (plain `curl` is something else there), cmd / mac / linux are fine.

2. install
```
pip install chess scipy pandas pyarrow scikit-learn pillow
```

3. run
```
python fly_gui.py     # play chess vs the fly
python fly_mind.py    # poke its senses or let it dream (gif + wav + poster go to dreams/)
```

first start of fly_gui takes ~10 min: it turns the parquet into `data/cache/*.npy` (memory-mapped, so many processes share one copy, ~65MB ram each instead of ~2GB) and builds a live set for each of the 8 flies. after that it loads in seconds. `mb_cells.npz` (kenyon cells, mbons, PAM, PPL1, bristle neurons) is already built, to rebuild it put `Supplemental_file1_neuron_annotations.tsv` from github.com/flyconnectome/flywire_annotations at `data/flywire_annotations/neuron_annotations.tsv` and run `python mb.py cells`.

board on the left, brain on the right: all 127,400 neurons at their real 3d positions (`brain3d.npz`), slowly spinning. drag to rotate, scroll to zoom, double-click to reset, it starts spinning again a few seconds after you let go. the "2D / 3D" button switches back to the old flat front view. all 8 flies taste the moves on all cores but one, then you watch the first fly's spikes for each move (they fade out smoothly), green / red glow = dopamine / pain neurons. ~5-20 sec per move depending on what else the pc is doing. the window is ~1540x890 (96px squares, 700px brain) for recording on a 1080p screen, it turns on windows dpi awareness so it stays sharp at 125% scaling. the 90px pieces in `pieces/big` are upscaled from the 68px set.

other stuff:
```
python fly_player.py play      # terminal version
python fly_player.py vs-sf0    # fly vs stockfish lvl 0 (needs stockfish in PATH)
python flygm.py --play         # hybrid mode (needs stockfish)
python bench.py fly 20 1 random                # base | ahead | ahead+mem | fly | fly+1 | fly+2 | fly8, vs greedy | random
python selfplay.py 120 runs/w1.npz 1           # fly plays itself for 120 min and learns
python mb.py merge mb_weights.npz runs/w*.npz  # average w, sum n
```

round 2 pipeline (set `FLYGM_DS` to a folder outside onedrive/dropbox, the brain caches are ~150MB per setup):
```
python lab.py gen 1 345          # positions from varied games (run one per core, seeds 1..16)
python lab.py merge              # teacher + senses + train / held-out split
python lab.py sim w3s 0 6        # brain responses, shard 0 of 6. w<k> = wiring k, s = senses, a = apl
python lab.py eval board board+senses w0:tot w0s:tot w0s:bins w0s+w1s+w2s+w3s:tot w0sa:kc
python lab.py fit w3s tot        # readout for the player -> readouts/w3s_tot.npz
```

for parallel runs start one process per core with different seeds, then merge.

## credits

- connectome: FlyWire (Dorkenwald et al, Schlegel et al, Nature 2024)
- brain model: Shiu et al, Nature 2024 - github.com/philshiu/Drosophila_brain_model
- neuron positions + cell types (kenyon cells, mbons, PAM, PPL1, bristle neurons): github.com/flyconnectome/flywire_annotations
- pieces: cburnett set by Colin M.L. Burnett (GPLv2+)
