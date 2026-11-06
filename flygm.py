# hybrid mode: stockfish gives top moves, each move = sugar (better = sweeter),
# fly picks the one that makes MN9 (proboscis motor neuron) fire most
#   python flygm.py [--play] [--pure] [--sf-level N]
import argparse, math, random, shutil, time
import chess, chess.engine, chess.pgn
from flybrain import FlyBrain

STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"


def score_cp(info, pov):
    s = info["score"].pov(pov)
    return s.score(mate_score=10000)


class FlyGM:
    def __init__(self, engine, k=5, depth=14, taste_ms=100, max_hz=150, softness=80, seed=None):
        self.brain = FlyBrain(seed=seed)
        self.engine, self.k, self.depth = engine, k, depth
        self.taste_ms, self.max_hz, self.softness = taste_ms, max_hz, softness
        self.log = []   # (rank_chosen, cp_loss)

    def candidates(self, board):
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=self.k)
        return [(i["pv"][0], score_cp(i, board.turn)) for i in infos if "pv" in i]

    def choose(self, board, verbose=True):
        cands = self.candidates(board)
        best = cands[0][1]
        tastes = []
        for rank, (mv, cp) in enumerate(cands):
            loss = max(0, best - cp)
            hz = self.max_hz * math.exp(-loss / self.softness)   # sweeter = better move
            mn9 = self.brain.taste(hz, self.taste_ms)
            tastes.append((mn9, -rank, mv, cp, hz))
        mn9, neg_rank, mv, cp, hz = max(tastes, key=lambda x: (x[0], random.random()))
        self.log.append((-neg_rank, best - cp))
        if verbose:
            menu = "  ".join(f"{board.san(m)}:{h:.0f}Hz->{s}" for s, _, m, _, h in tastes)
            print(f"   🪰 tastes  {menu}   -> eats {board.san(mv)}")
        return mv


class PureFly:
    """No engine. Every legal move = same sugar. The fly picks by raw neural noise."""
    def __init__(self, n_taste=6, seed=None):
        self.brain, self.n = FlyBrain(seed=seed), n_taste
        self.log = []

    def choose(self, board, verbose=True):
        moves = random.sample(list(board.legal_moves), min(self.n, board.legal_moves.count()))
        tastes = [(self.brain.taste(100, 100), random.random(), m) for m in moves]
        mv = max(tastes)[2]
        if verbose:
            print(f"   🪰 (blind) tastes {len(moves)} moves -> {board.san(mv)}")
        return mv


def play_vs_engine(fly, engine, sf_level, fly_white=True, max_moves=200):
    engine.configure({"Skill Level": sf_level})
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update(White="FlyGM 🪰" if fly_white else f"Stockfish L{sf_level}",
                        Black=f"Stockfish L{sf_level}" if fly_white else "FlyGM 🪰",
                        Event="Fly brain vs engine")
    node = game
    t0 = time.time()
    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        fly_turn = (board.turn == chess.WHITE) == fly_white
        if fly_turn:
            mv = fly.choose(board)
        else:
            mv = engine.play(board, chess.engine.Limit(time=0.05)).move
            print(f"   Stockfish: {board.san(mv)}")
        node = node.add_variation(mv)
        board.push(mv)
    game.headers["Result"] = board.result(claim_draw=True)
    print(f"\nResult: {game.headers['Result']}  ({time.time()-t0:.0f}s)")
    return game


def play_human(fly):
    board = chess.Board()
    side = input("Play white or black? [w/b] ").strip().lower() != "b"
    while not board.is_game_over():
        print("\n" + str(board if side else board.transform(chess.flip_vertical).transform(chess.flip_horizontal)))
        if board.turn == side:
            txt = input("Your move (e.g. e4, Nf3): ").strip()
            try:
                board.push_san(txt)
            except ValueError:
                print("Illegal move, try again")
        else:
            print("Fly is tasting...")
            board.push(fly.choose(board))
    print("Result:", board.result())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--pure", action="store_true")
    ap.add_argument("--sf-level", type=int, default=10, help="opponent Stockfish skill 0-20")
    ap.add_argument("--black", action="store_true", help="fly plays black")
    ap.add_argument("--pgn", default="game.pgn")
    a = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)      # opponent
    eyes = chess.engine.SimpleEngine.popen_uci(STOCKFISH)        # fly's full-strength "eyes"
    fly = PureFly() if a.pure else FlyGM(eyes)
    try:
        if a.play:
            play_human(fly)
        else:
            game = play_vs_engine(fly, engine, a.sf_level, fly_white=not a.black)
            open(a.pgn, "w", encoding="utf-8").write(str(game) + "\n")   # utf-8 for the 🪰 on windows
            if fly.log and not a.pure:
                n = len(fly.log)
                top1 = sum(r == 0 for r, _ in fly.log) / n
                acpl = sum(min(l, 1000) for _, l in fly.log) / n
                print(f"Fly picked engine's #1 move {top1:.0%} of the time, avg loss {acpl:.0f} cp/move")
            print(f"PGN saved to {a.pgn}")
    finally:
        engine.quit(); eyes.quit()
