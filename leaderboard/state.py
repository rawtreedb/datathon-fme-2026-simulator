"""GET /api/state: leaderboard data. Vercel Python function, stdlib only. ?by=model adds per model_version rows (Beta view of /leaderboard).

Env: RAWTREE_API_KEY (read-only key: SELECT on organizer.outcomes), PRACTICE_START, FINAL_START, FINAL_END (ISO UTC),
     PRACTICE_REFRESH (seconds, default 600; lower it for load tests),
     PRACTICE_HIDE_SCORE (truthy: the practice ranking is shown without the 0-100 score column).
The time gating happens here: final-round numbers are never computed before FINAL_END.
Responses are cached by the Vercel CDN (s-maxage), never across a phase boundary.
"""
import json
import os
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

E = os.environ
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()  # noqa: E731
PRACTICE_START = ts(E.get("PRACTICE_START", "2026-11-14T09:00:00Z"))
FINAL_START = ts(E.get("FINAL_START", "2026-11-15T09:00:00Z"))   # 10:00 CET
FINAL_END = ts(E.get("FINAL_END", "2026-11-15T11:00:00Z"))       # 12:00 CET
REVEAL = FINAL_END + 60                                          # let the last requests resolve
PRACTICE_WINDOW = 1 * 3600
STEP, MIN_WINDOW, MAX_WINDOW = 300, 900, 3 * 3600  # ?from=&to= practice windows (Beta): 5-min steps, 15 min to 3 h
REFRESH = {"waiting": 30, "practice": int(E.get("PRACTICE_REFRESH", 600)), "final_running": 30, "final": 60}  # PRACTICE_REFRESH: load tests
HIDE_SCORE = E.get("PRACTICE_HIDE_SCORE", "").lower() in ("1", "true", "yes")
API = E.get("RAWTREE_API_URL", "https://api.rawtree.com")
# ?view=progress (/progress): anonymised final-round rank every 5 min. It computes final numbers during the final, so it only
# exists where FINAL_PROGRESS is set (an organisers' preview, never production); PROGRESS_SALT keys the aliases.
FINAL_PROGRESS = E.get("FINAL_PROGRESS", "").lower() in ("1", "true", "yes")
# Progressive reveal during the final: at each 5-min checkpoint from REVEAL_FROM, a random batch (order keyed by PROGRESS_SALT)
# of the teams ranked below REVEAL_BUFFER shows its team_NN; at REVEAL_LAST everyone but the top 3; the top 3 at the final ranking.
REVEAL_FROM, REVEAL_LAST, REVEAL_BUFFER = FINAL_START + 90 * 60, FINAL_END - 5 * 60, 5   # 11:30 and 11:55 CET


def query(sql):
    req = urllib.request.Request(f"{API}/v1/query", data=json.dumps({"sql": sql}).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {E['RAWTREE_API_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["data"]


# outcomes columns are schemaless (Dynamic): cast once so aggregates, argMax and countIf work
# p_buy / exp_revenue_eur: the shopper's purchase probability and expected revenue under the hidden model for the list
# actually shown (computed by the simulator). Scoring them instead of the realised purchase removes luck from the ranking.
OUTCOMES = """(SELECT team_id::String AS team_id, t::Float64 AS t, p_buy::Float64 AS p_buy, cold::Bool AS cold,
                served_by::String AS served_by, exp_revenue_eur::Float64 AS revenue_eur, model_version::String AS model_version
         FROM organizer.outcomes)"""


def norm(x, base, ideal):
    return max(0.0, min(100.0, 100 * (x - base) / (ideal - base))) if ideal > base else 0.0


def ranking(start, end):
    rows = query(f"""
        SELECT team_id, count() AS requests, 100 * avg(p_buy) AS conversion,
               100 * sumIf(p_buy, cold) / greatest(countIf(cold), 1) AS cold_conversion,
               100 * countIf(served_by = 'team') / count() AS on_time,
               sum(revenue_eur) / count() AS revenue_per_request,
               argMaxIf(model_version, t, model_version != '') AS model_version
        FROM {OUTCOMES} WHERE t >= {start} AND t < {end} GROUP BY team_id""")
    by = {r["team_id"]: r for r in rows}
    base, ref = by.pop("baseline", None), by.pop("reference", None)
    ideal = by.pop("oracle", None) or ref  # pre-oracle data: fall back to the old reference anchor
    if not base or not ideal:
        return [], None, None
    for r in by.values():
        r["conversion_score"] = norm(r["conversion"], base["conversion"], ideal["conversion"])
        r["cold_score"] = norm(r["cold_conversion"], base["cold_conversion"], ideal["cold_conversion"])
        # the 70 stream points (50 conversion + 10 cold start + 10 reliability) normalised to 0-100
        r["score"] = (50 * r["conversion_score"] + 10 * r["cold_score"] + 10 * r["on_time"]) / 70
    return sorted(by.values(), key=lambda r: (-r["score"], -r["revenue_per_request"])), base, ref


def progress(start, end):
    """Hourly conversion score per team, so teams can see themselves improve."""
    rows = query(f"""
        SELECT team_id, intDiv(toUInt32(t), 3600) AS h, 100 * avg(p_buy) AS conversion
        FROM {OUTCOMES} WHERE t >= {start} AND t < {end} GROUP BY team_id, h ORDER BY h""")
    hours = {}
    for r in rows:
        hours.setdefault(int(r["h"]), {})[r["team_id"]] = r["conversion"]
    out = {}
    for h, teams in sorted(hours.items()):
        b, f = teams.get("baseline"), teams.get("oracle") or teams.get("reference")
        if b is not None and f is not None:
            for team, conv in teams.items():
                if team not in ("baseline", "reference", "oracle"):
                    out.setdefault(team, []).append([h * 3600, round(norm(conv, b, f), 4)])
    return out


MIN_SHOWN, MIN_REQUESTS, MAX_VERSIONS = 100, 300, 10  # listed from 100 requests in the window; 300 (~2 min of full traffic) to be a team's best


def versions(start, end, teams):
    """Per model_version metrics for the Beta view of /leaderboard (?by=model). Requests the team did not answer have no version,
    so they count for the team (reliability) but for no model. Late or invalid answers count for their version.
    Versions with fewer than MIN_SHOWN requests are not listed (still counted in the team row)."""
    rows = query(f"""
        SELECT team_id, model_version, count() AS requests, 100 * avg(p_buy) AS conversion, countIf(cold) AS cold_requests,
               100 * sumIf(p_buy, cold) / greatest(countIf(cold), 1) AS cold_conversion,
               sum(revenue_eur) / count() AS revenue_per_request, min(t) AS first, max(t) AS last
        FROM {OUTCOMES} WHERE t >= {start} AND t < {end} AND model_version != ''
        GROUP BY team_id, model_version HAVING requests >= {MIN_SHOWN} ORDER BY requests DESC""")
    by = {}
    for r in rows:
        by.setdefault(r.pop("team_id"), []).append(r)
    for t in teams:
        vs = by.get(t["team_id"], [])
        t["more_versions"] = max(0, len(vs) - MAX_VERSIONS)  # free text from teams: cap the rows, keep the busiest
        t["versions"] = sorted(vs[:MAX_VERSIONS], key=lambda v: (v["requests"] < MIN_REQUESTS, -v["conversion"]))
        t["best"] = next((v["model_version"] for v in t["versions"] if v["requests"] >= MIN_REQUESTS), None)


def lights(now):
    rows = query(f"""
        SELECT team_id, 100 * countIf(served_by = 'team') / count() AS on_time
        FROM {OUTCOMES} WHERE t >= {now - 305} AND t < {now - 5} AND team_id NOT IN ('baseline', 'reference', 'oracle')
        GROUP BY team_id ORDER BY team_id""")
    return [{"team_id": r["team_id"], "green": r["on_time"] > 90} for r in rows]


def aliases(teams):
    """Stable two-letter aliases ("Team KX"), keyed by PROGRESS_SALT so they cannot be recomputed from team_NN."""
    import hashlib
    import hmac
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    codes, out, used = [a + b for a in letters for b in letters], {}, set()
    for t in sorted(teams):
        i = int(hmac.new(E.get("PROGRESS_SALT", "").encode(), t.encode(), hashlib.sha256).hexdigest(), 16) % len(codes)
        while codes[i] in used:  # collision: next free code, deterministic for a given salt and team set
            i = (i + 1) % len(codes)
        used.add(codes[i])
        out[t] = "Team " + codes[i]
    return out


def final_progress(now):
    """Cumulative final-round rank of every team at each completed 5-min mark: the ranking if the final ended there.
    Same score as ranking(); one query bucketed by 5 min. Only ranks and aliases leave this function."""
    end = min(now, FINAL_END)
    marks = int((end - FINAL_START) // STEP)
    if marks < 1:
        return {"checkpoints": [], "teams": []}
    rows = query(f"""
        SELECT team_id, intDiv(toUInt32(t) - {int(FINAL_START)}, {STEP}) AS k, count() AS n, sum(p_buy) AS p,
               countIf(cold) AS nc, sumIf(p_buy, cold) AS pc, countIf(served_by = 'team') AS ok, sum(revenue_eur) AS rev
        FROM {OUTCOMES} WHERE t >= {FINAL_START} AND t < {FINAL_START + marks * STEP} GROUP BY team_id, k""")
    acc = {}  # team -> per-mark sums
    for r in rows:
        acc.setdefault(r["team_id"], [[0.0] * 6 for _ in range(marks)])[int(r["k"])] = [r["n"], r["p"], r["nc"], r["pc"], r["ok"], r["rev"]]
    ranks = {t: [] for t in acc if t not in ("baseline", "reference", "oracle")}
    cum, score = {t: [0.0] * 6 for t in acc}, {}
    for k in range(marks):
        m = {}
        for t, buckets in acc.items():
            cum[t] = [a + b for a, b in zip(cum[t], buckets[k])]
            n, p, nc, pc, ok, rev = cum[t]
            m[t] = (100 * p / max(n, 1), 100 * pc / max(nc, 1), 100 * ok / max(n, 1), rev / max(n, 1))
        base, ideal = m.get("baseline"), m.get("oracle") or m.get("reference")
        if not base or not ideal:
            continue
        score = {t: (50 * norm(c, base[0], ideal[0]) + 10 * norm(cc, base[1], ideal[1]) + 10 * ot) / 70 for t, (c, cc, ot, _) in m.items() if t in ranks}
        for i, t in enumerate(sorted(score, key=lambda t: (-score[t], -m[t][3])), 1):
            ranks[t].append(i)
    names = aliases(ranks)
    checkpoints = [FINAL_START + (k + 1) * STEP for k in range(marks)]
    revealed = reveals(ranks, checkpoints)
    order = sorted(ranks, key=lambda t: ranks[t][-1] if ranks[t] else 999)
    # status light like the final-round board (last 5 min), and score gaps at the last checkpoint: relative, never absolute
    green = {r["team_id"]: r["green"] for r in lights(now)} if now < FINAL_END + 60 else {}
    lead = score.get(order[0], 0.0) if order else 0.0
    teams, above = [], None
    for i, t in enumerate(order):
        sc, top10 = score.get(t, 0.0), i < 10  # gaps only for the top 10
        teams.append({"alias": names[t], "ranks": ranks[t], "green": green.get(t),
                      "gap_leader": round(lead - sc, 2) if top10 else None,
                      "gap_above": round(above - sc, 2) if top10 and above is not None else None,
                      "team": t if t in revealed else None, "revealed_at": revealed.get(t)})  # team_NN only once revealed
        above = sc
    nxt = next((c for c in [REVEAL_FROM + i * STEP for i in range(int((REVEAL_LAST - REVEAL_FROM) // STEP) + 1)] if c > (checkpoints[-1] if checkpoints else 0)), None)
    return {"checkpoints": checkpoints, "teams": teams,
            "reveal": {"revealed": len(revealed), "to_reveal": max(0, len(ranks) - 3), "next": nxt, "from": REVEAL_FROM, "last": REVEAL_LAST}}


def reveals(ranks, checkpoints):
    """team -> checkpoint at which it was revealed. Recomputed from the rank history on every call (no stored state), so it
    never changes once shown. Steps REVEAL_FROM..REVEAL_LAST-5min reveal ceil(eligible / steps left) teams ranked below
    REVEAL_BUFFER, in a random order nobody can predict (keyed by PROGRESS_SALT); REVEAL_LAST reveals all but the top 3."""
    import hashlib
    import hmac
    import math
    key = E.get("PROGRESS_SALT", "").encode()
    order = sorted(ranks, key=lambda t: hmac.new(key, b"reveal:" + t.encode(), hashlib.sha256).hexdigest())
    steps = [c for c in checkpoints if REVEAL_FROM <= c < REVEAL_LAST]
    total_steps = int((REVEAL_LAST - REVEAL_FROM) // STEP)  # 11:30..11:50: 5
    out = {}
    for k, c in enumerate(checkpoints):
        if c < REVEAL_FROM:
            continue
        now_rank = {t: ranks[t][k] for t in ranks if len(ranks[t]) > k}
        if c >= REVEAL_LAST:
            batch = [t for t in order if t not in out and now_rank.get(t, 999) > 3]
        else:
            eligible = [t for t in order if t not in out and now_rank.get(t, 999) > REVEAL_BUFFER]
            left = total_steps - steps.index(c)
            batch = eligible[:math.ceil(len(eligible) / max(left, 1))]
        for t in batch:
            out[t] = c
    return out


def practice_window(now, q):
    """The last hour, sliding, unless ?from= (epoch s) and optional ?to= (default now) ask for another window:
    rounded down to 5 min, 15 min to 3 h long and inside the practice round, so never final-round data.
    Few distinct windows (5-min grid, 3 h max) keep the CDN cache useful and the cluster load bounded."""
    lo, hi = PRACTICE_START, min(now, FINAL_START)
    try:
        f = int(float(q["from"][0])) // STEP * STEP
        t = min(hi, max(lo + MIN_WINDOW, int(float(q["to"][0])) // STEP * STEP)) if "to" in q else hi
    except (KeyError, ValueError, OverflowError):
        return [max(lo, now - PRACTICE_WINDOW), now], False
    return [max(lo, t - MAX_WINDOW, min(f, t - MIN_WINDOW)), t], True


def state(now, by_model=False, q=None):
    if now < PRACTICE_START:
        mode, boundary = "waiting", PRACTICE_START
    elif now < FINAL_START:
        mode, boundary = "practice", FINAL_START
    elif now < REVEAL:
        mode, boundary = "final_running", REVEAL
    else:
        mode, boundary = "final", float("inf")
    ttl = int(max(1, min(REFRESH[mode], boundary - now)))
    s = {"mode": mode, "updated_at": now, "next_update": now + ttl,
         "practice_start": PRACTICE_START, "final_start": FINAL_START, "final_end": FINAL_END}
    if mode == "practice":
        s["window"], s["custom"] = practice_window(now, q or {})
        s["limits"] = {"earliest": PRACTICE_START, "step": STEP, "min": MIN_WINDOW, "max": MAX_WINDOW}
        s["teams"], s["baseline"], s["reference"] = ranking(*s["window"])
        s["progress"] = progress(PRACTICE_START, now)
        if HIDE_SCORE:  # keep the ranking, drop the 0-100 score from the JSON: it is what the final round scores
            s["hidden"] = True
            for r in s["teams"]:
                for k in ("score", "conversion_score", "cold_score"):
                    r.pop(k)
    elif mode == "final_running":
        s["lights"] = lights(now)
        s["progress"] = progress(PRACTICE_START, FINAL_START)
    elif mode == "final":
        s["window"] = [FINAL_START, FINAL_END]
        s["teams"], s["baseline"], s["reference"] = ranking(*s["window"])
        s["progress"] = progress(PRACTICE_START, FINAL_END)
    if by_model and s.get("teams"):  # never during final_running: no teams there
        versions(*s["window"], s["teams"])
        s["min_requests"] = MIN_REQUESTS
    return s, ttl


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            q = parse_qs(urlparse(self.path).query)
            if q.get("view") == ["progress"]:  # organisers' preview only: 404 wherever FINAL_PROGRESS is not set
                now = time.time()
                if not FINAL_PROGRESS or now < FINAL_START:
                    body, status, cache = b'{"error":"not available"}', 404, "no-store"
                else:
                    body, status, cache = json.dumps({"now": now, "final_start": FINAL_START, "final_end": FINAL_END, **final_progress(now)}).encode(), 200, "public, max-age=0, s-maxage=60"
            else:
                s, ttl = state(time.time(), by_model=q.get("by") == ["model"], q=q)
                body, status, cache = json.dumps(s).encode(), 200, f"public, max-age=0, s-maxage={ttl}"
        except Exception as e:  # the page keeps the last data and retries
            print("state error:", str(e)[:300])
            body, status, cache = b'{"error":"unavailable"}', 503, "no-store"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)
