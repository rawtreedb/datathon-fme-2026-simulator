"""Fallback/baseline rules (python -m pytest simulator/test_baseline.py, or python simulator/test_baseline.py)."""
import os

import numpy as np

os.environ.setdefault("SEED_A", "1")  # dummy seeds: simulator.py reads them at import; no secrets needed
os.environ.setdefault("SEED_B", "2")
import simulator as S  # noqa: E402
import world as W


class FakeWorld:
    def __init__(self, pids, sales):
        self.pid, self.n = list(pids), len(pids)
        self.pidx = {p: i for i, p in enumerate(self.pid)}
        self.sales = np.r_[sales, np.zeros(W.MAXP - len(sales))]

    bestseller_ranking = W.World.bestseller_ranking
    available_top = staticmethod(W.World.available_top)


def starter_order(pids, sales):
    """The public starter's best_sellers(), on the same counts."""
    return [p for _, p in sorted(((int(n), p) for p, n in zip(pids, sales) if n > 0), reverse=True)]


PIDS = [f"p_{i:06x}" for i in range(14)]
SALES = [50, 40, 40, 30, 30, 30, 20, 10, 10, 10, 10, 5, 0, 0]  # ties inside the top 10 and across slot 10


def test_ranking_matches_starter_including_ties():
    w = FakeWorld(PIDS, SALES)
    assert [w.pid[i] for i in w.bestseller_ranking(w.sales)][:12] == starter_order(PIDS, SALES)


def test_first_ten_in_stock_and_restock():
    w = FakeWorld(PIDS, SALES)
    rank = w.bestseller_ranking(w.sales)
    active = np.ones(w.n, bool)
    full = list(w.available_top(rank, active))
    assert len(full) == 10 and full == list(rank[:10])
    active[rank[0]] = active[rank[3]] = False  # two out of stock: everything moves up, 11th and 12th enter
    out = list(w.available_top(rank, active))
    assert out == [i for i in rank if active[i]][:10] and rank[0] not in out and rank[3] not in out
    assert all(active[i] for i in out) and len(set(out)) == 10
    active[:] = True  # restocked: back to the original list
    assert list(w.available_top(rank, active)) == full


def test_ranking_ignores_products_added_later():
    w = FakeWorld(PIDS, SALES)
    rank = w.bestseller_ranking(w.sales)
    assert len(w.available_top(rank, np.ones(5, bool), k=10)) == 5  # snapshot shorter than the ranking: no IndexError


def test_validate_uses_request_snapshot():
    sim = S.Sim.__new__(S.Sim)
    sim.w = FakeWorld(PIDS, SALES)
    items = PIDS[:10]
    active = np.ones(len(PIDS), bool)
    assert sim.validate(items, active) == list(range(10))
    active[3] = False
    assert sim.validate(items, active) is None                    # out of stock at request time
    assert sim.validate(PIDS[:9] + [PIDS[0]], np.ones(14, bool)) is None  # duplicate
    assert sim.validate(PIDS[:9], np.ones(14, bool)) is None              # 9 items
    assert sim.validate(PIDS[:9] + ["p_nope"], np.ones(14, bool)) is None  # unknown product
    assert sim.validate(PIDS[:10], np.ones(9, bool)) is None             # product added after the request


def test_fallback_is_always_valid_for_teams():
    sim = S.Sim.__new__(S.Sim)
    sim.w = FakeWorld(PIDS, SALES)
    rank = sim.w.bestseller_ranking(sim.w.sales)
    rng = np.random.default_rng(0)
    for _ in range(200):
        active = rng.random(len(PIDS)) > 0.2
        if active.sum() < 10:
            continue
        fb = W.World.available_top(rank, active)
        assert sim.validate([PIDS[i] for i in fb], active) == list(fb)


def test_stock_out_grace():
    """A product sold out less than GRACE s before the request still makes a valid answer: its catalog_updates row
    lands up to ~2 s late. apply_actions keeps out_at; the request snapshot (`active`) is never changed."""
    from datetime import datetime, timedelta, timezone
    sim = S.Sim.__new__(S.Sim)
    sim.w = FakeWorld(PIDS, SALES)
    sim.w.active, sim.w.seed, sim.w._pop_cache = np.ones(len(PIDS), bool), 1, None
    t0 = datetime(2026, 11, 14, 11, 50, tzinfo=timezone.utc)
    sim.sales, sim.seq, sim.out_at = np.array(SALES, float), 1, {}
    sim.heap, sim.updates_written = [(t0, 0, "stock_out", 3)], t0 + timedelta(days=1)  # replayed: no catalog_updates rows
    sim.apply_actions(t0)
    out = sorted(sim.out_at)
    assert len(out) == 3 and all(sim.out_at[i] == t0 for i in out) and not sim.w.active[out].any()
    active = sim.w.active.copy()                                   # the request snapshot
    items = out + [i for i in range(len(PIDS)) if i not in out][:7]
    answer = [PIDS[i] for i in items]
    for dt, ok in ((0.0, True), (1.8, True), (4.9, True), (5.0, False), (60.0, False)):
        got = sim.validate(answer, sim.valid_at(t0 + timedelta(seconds=dt), active))
        assert (got == items) is ok, (dt, got)
    assert sim.validate(answer, active) is None and not active[out].any()  # snapshot untouched: fallback and scoring use it
    sim.apply_actions(t0 + timedelta(hours=1))                     # restock
    assert sim.out_at == {} and sim.w.active.all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
