"""Generate catalog + 60 days of history as JSONL.gz files.

    python generate.py --out data/ [--events 10000000 --impressions 2000000]

Uses WORLD_SEED from the environment. Upload afterwards with `python admin.py load data/`.
"""
import argparse
import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import world as W

CHUNK = 2000            # sessions per vectorised batch
FILE_ROWS = 250_000     # rows per output file


class Writer:
    def __init__(self, out, name):
        self.out, self.name, self.k, self.rows, self.f = out, name, 0, 0, None

    def write(self, row):
        if self.f is None or self.rows >= FILE_ROWS:
            self.close()
            self.f = gzip.open(self.out / f"{self.name}-{self.k:04d}.jsonl.gz", "wt", compresslevel=3)
            self.k, self.rows = self.k + 1, 0
        self.f.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.rows += 1

    def close(self):
        if self.f:
            self.f.close()


def timestamps(rng, n, end, days=60):
    day = rng.integers(0, days, n)
    hour = rng.choice(24, n, p=W.HOUR_W / W.HOUR_W.sum())
    sec = rng.random(n) * 3600
    start = end - timedelta(days=days)
    return [start + timedelta(days=int(d), hours=int(h), seconds=float(s)) for d, h, s in zip(day, hour, sec)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--events", type=int, default=10_000_000, help="approximate history_events rows")
    ap.add_argument("--impressions", type=int, default=2_000_000, help="history_sessions rows")
    ap.add_argument("--end", default="2026-11-14T07:00:00Z", help="history ends here (event start)")
    a = ap.parse_args()
    seed = int(os.environ["WORLD_SEED"])
    end = datetime.fromisoformat(a.end.replace("Z", "+00:00"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w = W.World(seed, created_before=end)
    U = w.users
    act = U["activity"] / U["activity"].sum()
    rng = W.rng_for("history", seed)

    with gzip.open(out / "catalog-0000.jsonl.gz", "wt") as f:
        for i in range(w.n):
            f.write(json.dumps(w.product_row(i)) + "\n")

    # organic events: view -> add_to_cart -> purchase
    ev = Writer(out, "history_events")
    sales = np.zeros(w.n)
    n_sessions, done = a.events // 4, 0
    while done < n_sessions:
        m = min(CHUNK, n_sessions - done)
        us = rng.choice(len(act), m, p=act)
        nv = np.minimum(1 + rng.geometric(0.3, m), 9)
        V0, ss = w.sessions(rng, U["P"][us], U["beta"][us], nv)
        for r, (u, t, views) in enumerate(zip(us, timestamps(rng, m, end), ss)):
            base = {"user_id": U["id"][u], "session_id": W.hexid(rng, "s_", 12), "device": W.DEVICES[U["device"][u]], "region": W.REGIONS[U["region"][u]]}
            st = w.session_terms(views[:1])
            for v in views:
                t += timedelta(seconds=float(rng.exponential(40)))
                row = {**base, "product_id": w.pid[v], "price_eur": float(w.price[v])}
                ev.write({"ts": W.iso(t), "event_type": "view", **row})
                if rng.random() < W.sigmoid(V0[r, v] + st[v] - W.TAU_CART):
                    t += timedelta(seconds=float(rng.exponential(20)))
                    ev.write({"ts": W.iso(t), "event_type": "add_to_cart", **row})
                    if rng.random() < 0.5:
                        t += timedelta(seconds=float(rng.exponential(60)))
                        ev.write({"ts": W.iso(t), "event_type": "purchase", **row})
                        sales[v] += 1
        done += m
        if done % 100_000 == 0 or done == n_sessions:
            print(f"events: {done}/{n_sessions} sessions", flush=True)
    ev.close()

    # carousel impressions from the old policy: Plackett-Luce over the category, weights = (sales + 1)^2
    hs = Writer(out, "history_sessions")
    cat_of = W.SUB_CAT[w.sub]
    cat_idx = [np.flatnonzero(cat_of == c) for c in range(len(W.CATEGORIES))]
    logw = 2 * np.log1p(sales)
    cat_p = np.array([np.exp(w.b[i]).sum() for i in cat_idx])
    cat_p /= cat_p.sum()
    done = 0
    while done < a.impressions:
        m = min(CHUNK, a.impressions - done)
        us = rng.choice(len(act), m, p=act)
        nv = rng.choice(7, m, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.06, 0.04])
        V0, ss = w.sessions(rng, U["P"][us], U["beta"][us], nv)
        for r, (u, t, views) in enumerate(zip(us, timestamps(rng, m, end), ss)):
            c = cat_of[views[-1]] if views else rng.choice(len(cat_p), p=cat_p)
            idx = cat_idx[c]
            shown = idx[np.argsort(-(logw[idx] + rng.gumbel(size=len(idx))))[:10]]
            wts = np.exp(logw[shown])
            prop = wts / (np.exp(logw[idx]).sum() - np.r_[0, np.cumsum(wts)[:-1]])
            V = V0[r] + w.session_terms(views)
            dr = {"exam": rng.random(10), "g": np.zeros(W.MAXP), "g0": rng.gumbel(), "buy": np.zeros(W.MAXP)}
            dr["g"][shown], dr["buy"][shown] = rng.gumbel(size=10), rng.random(10)
            clicked, bought = w.choose(V, shown, dr)
            vts = [t - timedelta(seconds=float(s)) for s in np.sort(rng.exponential(45, len(views)).cumsum())[::-1]]
            hs.write({
                "request_id": W.hexid(rng, "r_", 16), "ts": W.iso(t), "user_id": U["id"][u],
                "context": {"device": W.DEVICES[U["device"][u]], "region": W.REGIONS[U["region"][u]],
                            "page": "home" if not views else ("product" if rng.random() < 0.7 else "category")},
                "recent_views": [{"product_id": w.pid[v], "ts": W.iso(x)} for v, x in zip(views, vts)],
                "shown": [w.pid[i] for i in shown], "propensity": [round(float(p), 6) for p in prop],
                "clicked": w.pid[clicked] if clicked >= 0 else None, "purchased": w.pid[clicked] if bought else None,
                "revenue_eur": float(w.price[clicked]) if bought else 0.0,
            })
        done += m
        if done % 100_000 == 0 or done == a.impressions:
            print(f"impressions: {done}/{a.impressions}", flush=True)
    hs.close()


if __name__ == "__main__":
    main()
