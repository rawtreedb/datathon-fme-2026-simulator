"""Live shopper simulator.

    python simulator.py

Env: RAWTREE_API_KEY (admin key), WORLD_SEED, SEED_A (practice), SEED_B (final),
     PRACTICE_START, FINAL_START, FINAL_END, HISTORY_END (ISO UTC), DEADLINE_S=3, RATE=3, EVENTS_RATE=50.

Everything is a function of the seeds and the clock, so it can be restarted at any time:
catalog changes are replayed, satiation state is reloaded from RawTree.
"""
import heapq
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

import world as W
from rawtree import RawTree

E = os.environ
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
PRACTICE_START = ts(E.get("PRACTICE_START", "2026-11-14T09:00:00Z"))
FINAL_START = ts(E.get("FINAL_START", "2026-11-15T09:00:00Z"))   # 10:00 CET
FINAL_END = ts(E.get("FINAL_END", "2026-11-15T11:00:00Z"))       # 12:00 CET
HISTORY_END = ts(E.get("HISTORY_END", "2026-11-14T07:00:00Z"))
SEEDS = {"practice": int(E["SEED_A"]), "final": int(E["SEED_B"])}
DEADLINE, POLL, POLL_CHUNK = float(E.get("DEADLINE_S", 3)), 0.5, 10
RATE, SESSIONS_PER_S = float(E.get("RATE", 3)), float(E.get("EVENTS_RATE", 50)) / 3.6
NEW_USER_SHARE = 0.3
PSEUDO = ("baseline", "reference", "oracle")
GRACE = 5  # s: a product sold out less than this before a request still makes a valid answer (its catalog_updates row lands 0.3-1.8 s late)


def now():
    return datetime.now(timezone.utc)


def round_of(t):
    return "final" if t >= FINAL_START else "practice"


def build_schedule(w):
    """Catalog actions (heap of (t, seq, kind, payload)) and trend windows [(start, end, subcategory)]."""
    acts, trends, H = [], [], timedelta(hours=1)
    rng = W.rng_for("schedule", SEEDS["practice"])
    t = PRACTICE_START + H
    while t < FINAL_START:
        acts.append((t, "new_products", (SEEDS["practice"], 100)))
        t += 4 * H
    t = PRACTICE_START + 0.5 * H
    while t < FINAL_START:
        trends.append((t, t + 0.75 * H, int(rng.integers(len(W.SUBCATS)))))
        t += 2 * H
    t = PRACTICE_START + 1.5 * H
    while t + 2 * H < FINAL_START:
        c = int(rng.integers(len(W.CATEGORIES)))
        acts += [(t, "price_drop", c), (t + 2 * H, "price_restore", c)]
        t += 3 * H
    rng = W.rng_for("schedule", SEEDS["final"])
    acts.append((FINAL_START, "new_products", (SEEDS["final"], 300)))
    s = FINAL_START + timedelta(minutes=float(rng.uniform(0, 75)))
    trends.append((s, s + 0.75 * H, int(rng.integers(len(W.SUBCATS)))))
    acts.append((FINAL_START + timedelta(minutes=float(rng.uniform(10, 60))), "price_drop", int(rng.integers(len(W.CATEGORIES)))))
    t = PRACTICE_START  # first stock-out at the start, so shop.catalog_updates is never empty during the event
    while t < FINAL_END:
        acts.append((t, "stock_out", 3))
        t += timedelta(minutes=5)
    heap = [(t, i, k, p) for i, (t, k, p) in enumerate(sorted(acts, key=lambda a: a[0]))]
    heapq.heapify(heap)
    return heap, trends


class Sim:
    def __init__(self):
        self.rt = RawTree()
        self.w = W.World(int(E["WORLD_SEED"]), created_before=HISTORY_END)
        w, U = self.w, self.w.users
        self.hist_p = U["activity"] / U["activity"].sum()
        self.pools = {r: w.make_users(W.rng_for("newusers", s), 20_000, "u_") for r, s in SEEDS.items()}
        self.p_hat = U["P"] + W.rng_for("phat", w.seed).normal(0, 0.5 * w.pscale, U["P"].shape)
        self.beta_hat = U["beta"].mean()
        self.uidx = {u: i for i, u in enumerate(U["id"])}
        self.heap, self.trends = build_schedule(w)
        self.seq = len(self.heap)
        self.out_at = {}  # product idx -> when it went out of stock, while it stays out (GRACE)
        self.pending = {}
        self.teams = []
        self.bought = defaultdict(lambda: defaultdict(set))  # team -> user -> {product idx}; "" = shared organic
        self.buf = defaultdict(list)                          # (db, table) -> rows
        self.io = ThreadPoolExecutor(64)                      # inserts take ~0.5 s: ~45/s with 40 teams
        self.poller = ThreadPoolExecutor(1)                   # never queued behind inserts
        self.poll_pool = ThreadPoolExecutor(16)               # chunks of one poll
        self.poll_future, self.last_poll, self.last_flush, self.errors = None, 0.0, 0.0, 0
        self.load_state()

    # ---------- startup ----------
    def q(self, sql):
        """Query that returns [] while a live table is still empty (its columns do not exist yet)."""
        try:
            return self.rt.query(sql)
        except RuntimeError as e:
            print(f"(empty?) {str(e)[:160]}")
            return []

    def load_state(self):
        rt, w = self.rt, self.w
        sales = np.zeros(W.MAXP)
        for r in rt.query("SELECT product_id, count() AS n FROM shop.history_events WHERE event_type = 'purchase' GROUP BY product_id"):
            if r["product_id"] in w.pidx:
                sales[w.pidx[r["product_id"]]] = r["n"]
        self.sales, self.ranking = sales, w.bestseller_ranking(sales)  # frozen order; each request takes its first 10 in stock
        for table in ("history_events", "events"):
            for r in self.q(f"SELECT user_id, product_id FROM shop.{table} WHERE event_type = 'purchase'"):
                if r["product_id"] in w.pidx:
                    self.bought[""][r["user_id"]].add(w.pidx[r["product_id"]])
        for r in self.q("SELECT team_id, user_id, purchased FROM organizer.outcomes WHERE purchased != ''"):
            if r["purchased"] in w.pidx:
                self.bought[r["team_id"]][r["user_id"]].add(w.pidx[r["purchased"]])
        last = self.q("SELECT toString(max(ts::DateTime64(3))) AS m FROM shop.catalog_updates")
        m = last[0]["m"] if last else ""
        self.updates_written = ts(m.replace(" ", "T") + ("" if m.endswith("Z") else "Z")) if m and not m.startswith("1970") else PRACTICE_START - timedelta(days=1)
        self.apply_actions(now())  # replay; only rows newer than updates_written get written
        self.refresh_teams()

    def refresh_teams(self):
        # never shrink: a replica still catching up lists fewer databases. After remove-team during the event, `make deploy`.
        found = {d for d in self.rt.databases() if re.fullmatch(r"team_\d+", d)}
        self.teams = sorted(found | set(getattr(self, "teams", [])))

    # ---------- catalog ----------
    def apply_actions(self, t):
        w, changed = self.w, False
        while self.heap and self.heap[0][0] <= t:
            at, _, kind, p = heapq.heappop(self.heap)
            if kind == "new_products":
                seed, n = p
                idx, change = w.add_products(W.rng_for("batch", seed, at.isoformat()), n, [W.iso(at)] * n, new=True), "new_product"
            elif kind == "price_drop":
                idx, change = np.flatnonzero((W.SUB_CAT[w.sub] == p) & w.active), "price_change"
                w.price[idx] = np.floor(w.base_price[idx] * 0.8) + 0.99
            elif kind == "price_restore":
                idx, change = np.flatnonzero(W.SUB_CAT[w.sub] == p), "price_change"
                w.price[idx] = w.base_price[idx]
            elif kind == "stock_out":
                rng = W.rng_for("stock", w.seed, at.isoformat())
                act = np.flatnonzero(w.active)
                pw = (self.sales[act] + 1) / (self.sales[act] + 1).sum()
                idx, change = rng.choice(act, p, replace=False, p=pw), "out_of_stock"
                w.active[idx] = False
                self.out_at.update({int(i): at for i in idx})
                heapq.heappush(self.heap, (at + timedelta(hours=1), self.seq, "restock", [int(i) for i in idx]))
                self.seq += 1
            elif kind == "restock":
                idx, change = np.array(p), "back_in_stock"
                w.active[idx] = True
                for i in p:
                    self.out_at.pop(i, None)
            w._pop_cache, changed = None, True
            if at > self.updates_written:
                self.buf[("shop", "catalog_updates")] += [{**w.product_row(i), "ts": W.iso(at), "change": change} for i in idx]

    def trend_at(self, t):
        return next((s for a, b, s in self.trends if a <= t < b), None)

    # ---------- generation ----------
    def user(self, rng, t):
        if rng.random() < NEW_USER_SHARE:
            pool, new = self.pools[round_of(t)], True
            u = int(rng.integers(len(pool["id"])))
        else:
            pool, new = self.w.users, False
            u = int(rng.choice(len(self.hist_p), p=self.hist_p))
        return pool, u, new

    def make_request(self, k, t):
        w, seed = self.w, SEEDS[round_of(t)]
        rng = W.rng_for("req", seed, k)
        pool, u, new = self.user(rng, t)
        nv = int(rng.choice(7, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.06, 0.04]))
        V0, (views,) = w.sessions(rng, pool["P"][u][None], pool["beta"][u][None], [nv], w.trend_vector(self.trend_at(t)))
        rid = W.hexid(rng, "r_", 16)
        offs = np.sort(rng.exponential(45, nv).cumsum())[::-1]
        row = {"request_id": rid, "ts": W.iso(t), "user_id": pool["id"][u], "is_new_user": new,
               "context": {"device": W.DEVICES[pool["device"][u]], "region": W.REGIONS[pool["region"][u]],
                           "page": "home" if not views else ("product" if rng.random() < 0.7 else "category")},
               "recent_views": [{"product_id": w.pid[v], "ts": W.iso(t - timedelta(seconds=float(o)))} for v, o in zip(views, offs)]}
        active = w.active.copy()  # stock at request time: used for the fallback, validation and scoring alike
        p_hat = self.p_hat[u] if not new else self.p_hat.mean(0)
        V = V0[0] + w.session_terms(views)
        Vo = V.copy()
        oseen = self.bought["oracle"][row["user_id"]] | self.bought[""][row["user_id"]]
        if oseen:
            Vo[list(oseen)] -= W.DELTA
        self.pending[rid] = {"t": t, "row": row, "user": row["user_id"], "seed": seed, "views": views, "answers": {}, "active": active,
                             "valid": self.valid_at(t, active),
                             "V": V, "cold": new or bool(w.is_new[views].any()),
                             "fallback": W.World.available_top(self.ranking, active),
                             "reference": w.reference(p_hat, self.beta_hat, views, self.bought["reference"][row["user_id"]] | self.bought[""][row["user_id"]]),
                             "oracle": np.argsort(-np.where(w.active[: len(Vo)], Vo, -np.inf))[:10]}
        self.io.submit(self.safe, self.rt.insert, "shop", "requests", [row])  # right away and in parallel: an insert takes ~0.5 s

    def make_events(self, s, t):
        w = self.w
        rng = W.rng_for("ev", SEEDS[round_of(t)], s)
        m = int(SESSIONS_PER_S) + (rng.random() < SESSIONS_PER_S % 1)
        users = [self.user(rng, t) for _ in range(m)]
        P = np.array([p["P"][u] for p, u, _ in users])
        beta = np.array([p["beta"][u] for p, u, _ in users])
        V0, ss = w.sessions(rng, P, beta, np.minimum(1 + rng.geometric(0.3, m), 9), w.trend_vector(self.trend_at(t)))
        rows = []
        for r, ((pool, u, _), views) in enumerate(zip(users, ss)):
            base = {"user_id": pool["id"][u], "session_id": W.hexid(rng, "s_", 12), "device": W.DEVICES[pool["device"][u]], "region": W.REGIONS[pool["region"][u]]}
            st = w.session_terms(views[:1])
            for j, v in enumerate(views):
                e = {**base, "product_id": w.pid[v], "price_eur": float(w.price[v])}
                at = t + timedelta(milliseconds=int(1000 * (r + j / 10) / max(m, 1)))
                rows.append({"ts": W.iso(at), "event_type": "view", **e})
                if rng.random() < W.sigmoid(V0[r, v] + st[v] - W.TAU_CART):
                    rows.append({"ts": W.iso(at), "event_type": "add_to_cart", **e})
                    if rng.random() < 0.5:
                        rows.append({"ts": W.iso(at), "event_type": "purchase", **e})
                        self.bought[""][base["user_id"]].add(v)
        self.buf[("shop", "events")] += rows

    # ---------- answers ----------
    def poll(self, teams, ids, sent):
        """Answers of every team for these requests. One UNION ALL per 10 teams, in parallel: a single
        40-team UNION ALL takes 0.5-1 s and would stretch the 0.5 s poll cadence (measured 2026-11-09)."""
        q = "', '".join(ids)
        chunks = [teams[i:i + POLL_CHUNK] for i in range(0, len(teams), POLL_CHUNK)]
        return sent, [r for rows in self.poll_pool.map(lambda c: self.poll_chunk(c, q), chunks) for r in rows]

    def poll_chunk(self, teams, q):
        sel = lambda t: f"SELECT '{t}' AS team, request_id::String AS request_id, items, model_version::String AS model_version FROM {t}.submissions WHERE request_id::String IN ('{q}')"  # noqa: E731
        try:
            return self.rt.query(" UNION ALL ".join(map(sel, teams)))
        except Exception:  # one broken team table must not hide the others
            rows = []
            for t in teams:
                try:
                    rows += self.rt.query(sel(t))
                except Exception as e:
                    print(f"poll {t}: {str(e)[:200]}")
            return rows

    def valid_at(self, t, active):
        """Stock that answers are judged against: `active` plus products that went out of stock less than GRACE s
        before t, whose catalog_updates row teams may not see yet. Those still cannot be bought (resolve)."""
        valid = active.copy()
        valid[[i for i, at in self.out_at.items() if (t - at).total_seconds() < GRACE]] = True
        return valid

    def validate(self, items, active):
        """Product indices of a valid answer (10 distinct products in stock when the request was made, see valid_at), else None."""
        try:
            idx = [self.w.pidx.get(str(x)) for x in items]
        except TypeError:
            return None
        if len(idx) != 10 or len(set(idx)) != 10 or any(i is None or i >= len(active) or not active[i] for i in idx):
            return None
        return idx

    def record(self, sent, rows):
        for r in rows:
            p = self.pending.get(r["request_id"])
            if p is None or (r["team"] in p["answers"] and p["answers"][r["team"]]["items"] is not None):
                continue
            items = r["items"] if isinstance(r["items"], list) else None
            p["answers"][r["team"]] = {"seen": sent, "items": self.validate(items, p["valid"]) if items is not None else None, "mv": str(r.get("model_version", ""))[:64]}

    # ---------- outcomes ----------
    def resolve(self, rid):
        p, w = self.pending.pop(rid), self.w
        dr = w.draws(p["seed"], rid)
        rnd, user = round_of(p["t"]), p["user"]
        for team in self.teams + list(PSEUDO):
            a = p["answers"].get(team)
            latency = int(1000 * (a["seen"] - p["t"]).total_seconds()) if a else None
            if team in PSEUDO:
                items, served, reason = p["fallback" if team == "baseline" else team], "team", ""
            elif a and a["items"] is not None and latency <= 1000 * (DEADLINE + POLL):
                items, served, reason = np.array(a["items"]), "team", ""
            else:
                items, served = p["fallback"], "fallback"
                reason = "missing" if not a else ("invalid" if a["items"] is None else "late")
            V = np.r_[p["V"], np.full(w.n - len(p["V"]), p["V"].min() - 10)]  # products added after the request: unattractive
            seen = self.bought[team][user] | self.bought[""][user]
            if seen:
                V[list(seen)] -= W.DELTA
            if served == "team" and team not in PSEUDO:  # accepted within GRACE: shown, but a sold-out product cannot be bought
                V[items[~p["active"][items]]] = p["V"].min() - 10
            clicked, bought = w.choose(V, items, dr)
            p_buy, exp_revenue = W.expected(V, items, w.price)  # scored instead of `bought` (no luck); never shown to teams
            if bought:
                self.bought[team][user].add(clicked)
            out = {"request_id": rid, "shown": [w.pid[i] for i in items], "served_by": served, "fallback_reason": reason, "latency_ms": latency,
                   "clicked": w.pid[clicked] if clicked >= 0 else "", "purchased": w.pid[clicked] if bought else "",
                   "revenue_eur": float(w.price[clicked]) if bought else 0.0}
            self.buf[("organizer", "outcomes")].append({**out, "t": p["t"].timestamp(), "ts": p["row"]["ts"], "round": rnd, "team_id": team, "user_id": user,
                                                        "cold": p["cold"], "bought": int(bought), "p_buy": p_buy, "exp_revenue_eur": exp_revenue,
                                                        "model_version": a["mv"] if a else ""})
            if team not in PSEUDO:
                self.buf[(team, "feedback")].append({**out, "ts": W.iso(now()), "model_version": a["mv"] if a else ""})

    # ---------- io ----------
    def safe(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            self.errors += 1
            print(f"io error: {str(e)[:300]}")

    def flush(self):
        for (db, table), rows in self.buf.items():
            for i in range(0, len(rows), 20_000):
                self.io.submit(self.safe, self.rt.insert, db, table, rows[i:i + 20_000])
        self.buf = defaultdict(list)

    # ---------- main loop ----------
    def run(self):
        k = max(0, int((now() - PRACTICE_START).total_seconds() * RATE))
        s = max(0, int((now() - PRACTICE_START).total_seconds()))
        last_status, last_teams = time.time(), time.time()
        print(f"teams={self.teams} next request #{k}")
        while True:
            t = now()
            live = PRACTICE_START <= t < FINAL_END
            self.apply_actions(t)
            while live and PRACTICE_START + timedelta(seconds=k / RATE) <= t:
                self.make_request(k, t)
                k += 1
            while live and PRACTICE_START + timedelta(seconds=s) <= t:
                self.make_events(s, t)
                s += 1
            if self.poll_future and self.poll_future.done():
                self.record(*self.poll_future.result())
                self.poll_future = None
            if self.poll_future is None and time.time() - self.last_poll >= POLL:
                ids = [rid for rid, p in self.pending.items() if (t - p["t"]).total_seconds() <= DEADLINE + POLL]
                self.last_poll = time.time()
                if ids:
                    self.poll_future = self.poller.submit(self.poll, list(self.teams), ids, t)
            for rid in [rid for rid, p in self.pending.items() if (t - p["t"]).total_seconds() > DEADLINE + POLL + 0.7]:
                self.resolve(rid)
            if time.time() - self.last_flush >= 1:
                self.flush()
                self.last_flush = time.time()
            if time.time() - last_teams >= 30:
                self.io.submit(self.safe, self.refresh_teams)
                last_teams = time.time()
            if time.time() - last_status >= 10:
                print(f"{W.iso(t)} {round_of(t)} req#{k} pending={len(self.pending)} teams={len(self.teams)} trend={W.SUBCATS[self.trend_at(t)] if self.trend_at(t) is not None else None} io_errors={self.errors}", flush=True)
                last_status = time.time()
            if t >= FINAL_END and not self.pending:
                self.flush()
                self.io.shutdown(wait=True)
                print("final round over, stream stopped")
                return
            time.sleep(0.05)


if __name__ == "__main__":
    Sim().run()
