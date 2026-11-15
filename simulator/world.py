"""Hidden shopper model: the "truth" behind every click and purchase of the challenge.

Everything is a pure function of a seed, so the simulator can restart without saving state.
Run `python world.py` to calibrate: it prints baseline / reference / oracle conversion.
"""
import hashlib
from datetime import datetime, timedelta, timezone

import numpy as np

D = 16
MAXP = 10_000  # capacity for product indices; CRN draws are sized to it so indices stay stable

TAXONOMY = {
    "Electronics": ["Smartphones", "Phone Cases", "Chargers", "Headphones", "Laptops", "Laptop Sleeves", "Tablets", "Smartwatches", "Cameras"],
    "Computers & Gaming": ["Monitors", "Keyboards", "Mice", "Game Consoles", "Video Games", "Gaming Headsets", "Webcams", "External Drives", "Routers"],
    "Home & Kitchen": ["Coffee Machines", "Coffee Capsules", "Blenders", "Cookware", "Knives", "Cutting Boards", "Kettles", "Food Storage", "Toasters"],
    "Fashion": ["Sneakers", "Socks", "Jeans", "T-Shirts", "Jackets", "Backpacks", "Sunglasses", "Watches", "Belts"],
    "Beauty & Health": ["Shampoo", "Conditioner", "Face Cream", "Sunscreen", "Perfume", "Electric Shavers", "Shaver Blades", "Toothbrushes", "Vitamins"],
    "Sports & Outdoors": ["Running Shoes", "Fitness Trackers", "Yoga Mats", "Dumbbells", "Bicycles", "Bike Helmets", "Tents", "Sleeping Bags", "Water Bottles"],
    "Toys & Baby": ["Building Blocks", "Dolls", "Board Games", "Puzzles", "RC Cars", "Toy Batteries", "Strollers", "Baby Bottles", "Picture Books"],
    "Home & Garden": ["Sofas", "Cushions", "Lamps", "Light Bulbs", "Plants", "Plant Pots", "Garden Tools", "Bedding", "Pillows"],
    "Pets": ["Dog Food", "Cat Food", "Dog Toys", "Cat Litter", "Pet Beds", "Leashes", "Aquariums", "Fish Food", "Pet Shampoo"],
    "Books & Media": ["Novels", "Comics", "Cookbooks", "Travel Guides", "Vinyl Records", "Turntables", "E-Readers", "E-Reader Covers", "Notebooks"],
}
CATEGORIES = list(TAXONOMY)
SUBCATS = [s for subs in TAXONOMY.values() for s in subs]
SUB_CAT = np.array([CATEGORIES.index(c) for c, subs in TAXONOMY.items() for _ in subs])
COMPLEMENTS = [  # viewing the first makes the second more attractive
    ("Smartphones", "Phone Cases"), ("Smartphones", "Chargers"), ("Smartphones", "Headphones"), ("Tablets", "Chargers"),
    ("Laptops", "Laptop Sleeves"), ("Laptops", "Mice"), ("Monitors", "Keyboards"), ("Game Consoles", "Video Games"),
    ("Game Consoles", "Gaming Headsets"), ("Coffee Machines", "Coffee Capsules"), ("Knives", "Cutting Boards"),
    ("Sneakers", "Socks"), ("Running Shoes", "Socks"), ("Jeans", "Belts"), ("Electric Shavers", "Shaver Blades"),
    ("Shampoo", "Conditioner"), ("Bicycles", "Bike Helmets"), ("Tents", "Sleeping Bags"), ("RC Cars", "Toy Batteries"),
    ("Strollers", "Baby Bottles"), ("Lamps", "Light Bulbs"), ("Plants", "Plant Pots"), ("Bedding", "Pillows"),
    ("Dog Food", "Dog Toys"), ("Cat Food", "Cat Litter"), ("Aquariums", "Fish Food"), ("Turntables", "Vinyl Records"),
    ("E-Readers", "E-Reader Covers"), ("Cookbooks", "Cookware"), ("Running Shoes", "Water Bottles"),
]
COMP = np.zeros((len(SUBCATS), len(SUBCATS)), bool)
for a, b in COMPLEMENTS:
    COMP[SUBCATS.index(a), SUBCATS.index(b)] = True

# (word, phrase, price multiplier). Words in descriptions carry latent taste: the cold-start signal.
ATTRS = [
    ("premium", "premium materials", 1.6), ("budget", "great value for money", 0.6), ("eco", "made from recycled materials", 1.1),
    ("compact", "compact and lightweight", 1.0), ("pro", "built for professionals", 1.5), ("kids", "designed for kids", 0.8),
    ("vintage", "a vintage look", 1.1), ("sport", "a sporty style", 1.0), ("wireless", "wireless connectivity", 1.2),
    ("smart", "smart app control", 1.3), ("handmade", "a handmade finish", 1.3), ("minimal", "a minimalist design", 1.0),
    ("family", "a family-size format", 1.1), ("travel", "a travel-ready build", 1.0), ("luxury", "a luxury feel", 2.2),
    ("durable", "extra durability", 1.1), ("colorful", "bold colours", 0.9), ("organic", "organic ingredients", 1.2),
    ("waterproof", "a waterproof shell", 1.2), ("quiet", "whisper-quiet operation", 1.2),
]
TEMPLATES = [
    "{brand} {sub} with {a1} and {a2}.",
    "{sub} by {brand}, featuring {a1}. Comes with {a2}.",
    "From {brand} ({sub}): {a1}, {a2}.",
    "{brand} {sub} line, known for {a1} and {a2}.",
]
REGIONS = ["Catalunya", "Madrid", "Andalucía", "Comunitat Valenciana", "País Vasco", "Galicia", "Aragón", "Murcia", "Canarias", "Castilla y León"]
DEVICES = ["web", "ios", "android"]
PAGES = ["home", "product", "category"]
PRICE_SENSITIVE = {"Andalucía", "Murcia", "Canarias"}  # android shoppers here are twice as price sensitive
# segment: (category affinities, beta, device mix web/ios/android)
SEGMENTS = [
    ({"Electronics": 1, "Computers & Gaming": 1}, 0.6, (0.3, 0.45, 0.25)),
    ({"Home & Kitchen": 1, "Home & Garden": 1, "Toys & Baby": 0.7}, 0.9, (0.45, 0.25, 0.30)),
    ({"Fashion": 1, "Beauty & Health": 1}, 0.8, (0.2, 0.5, 0.3)),
    ({"Sports & Outdoors": 1, "Fashion": 0.5}, 0.7, (0.3, 0.35, 0.35)),
    ({"Pets": 1, "Home & Garden": 0.6}, 1.0, (0.35, 0.25, 0.4)),
    ({"Books & Media": 1, "Computers & Gaming": 0.4}, 0.5, (0.5, 0.3, 0.2)),
    ({c: 0.4 for c in CATEGORIES}, 2.0, (0.25, 0.15, 0.6)),  # bargain hunters
]
SEG_W = np.array([0.18, 0.17, 0.17, 0.13, 0.1, 0.1, 0.15])

# Model constants (tuned with `python world.py`).
ATTR_SD = 1.0     # within-subcategory taste spread (attributes)
USER_SD = 1.0     # personal taste spread around the segment mean
TASTE = 1.6       # weight of p_u . q_i
POP = 0.7         # weight of product appeal b_i
GAMMA = 0.35      # session intent: similarity to products viewed in this session
KAPPA = 1.2       # complements
DELTA = 4.0       # satiation: already bought
TREND = 1.6       # boost of the trending subcategory
U0 = 10.0         # outside option utility ("buy nothing")
TAU_BUY = 8.2     # click -> purchase threshold
TAU_CART = 6.5    # organic view -> add_to_cart threshold
EXAM = 1 / np.log2(np.arange(10) + 2)  # position bias: slot k examined with prob 1/log2(k+1)
HOUR_W = np.array([2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 6, 7, 8, 9, 10, 9, 6, 4], float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rng_for(*parts):
    h = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def hexid(rng, prefix, n=10):
    return prefix + "".join(rng.choice(list("0123456789abcdef"), n))


class World:
    def __init__(self, seed, n_products=5000, n_users=60_000, created_before=datetime(2026, 11, 14, tzinfo=timezone.utc)):
        rng = rng_for("world", seed)
        self.seed = seed
        self.cat_vec = rng.normal(0, 1.0, (len(CATEGORIES), D))
        self.sub_vec = self.cat_vec[SUB_CAT] + rng.normal(0, 0.7, (len(SUBCATS), D))
        self.attr_vec = rng.normal(0, ATTR_SD, (len(ATTRS), D))
        self.sub_logp = rng.normal(3.3, 1.0, len(SUBCATS))  # mean log price per subcategory
        self.brands, self.brand_cat = [], []
        syl = ["ka", "lo", "mi", "ra", "ve", "no", "ti", "sa", "do", "re", "lu", "xa", "zo", "pe", "ni", "ma", "ko", "ri", "ta", "vo", "be", "ga", "fi", "ne"]
        for c in range(len(CATEGORIES)):
            while sum(1 for x in self.brand_cat if x == c) < 8:
                name = "".join(rng.choice(syl, rng.integers(2, 4))).capitalize()
                if name not in self.brands:
                    self.brands.append(name)
                    self.brand_cat.append(c)
        self.brand_cat = np.array(self.brand_cat)
        self.brand_b = rng.normal(0, 0.4, len(self.brands))
        self.sub_pop = rng.dirichlet(np.ones(len(SUBCATS)) * 2)  # how many products per subcategory

        # products (arrays grow when new products arrive)
        self.pid, self.desc, self.created = [], [], []
        self.sub = np.zeros(0, int); self.brand = np.zeros(0, int); self.attrs = np.zeros((0, 2), int)
        self.q = np.zeros((0, D)); self.b = np.zeros(0); self.base_price = np.zeros(0)
        self.price = np.zeros(0); self.active = np.zeros(0, bool); self.is_new = np.zeros(0, bool)
        self.add_products(rng, n_products, [iso(created_before - timedelta(days=int(d))) for d in rng.integers(60, 900, n_products)])

        # normalise the latent space so dot products have unit-ish scale
        qq = self.q @ self.q[rng.choice(n_products, 500)].T
        self.qscale = 1 / qq.std()
        self.q *= np.sqrt(self.qscale)
        self.users = self.make_users(rng_for("users", seed), n_users, "u_")
        pq = self.users["P"][:500] @ self.q.T
        self.pscale = 1 / pq.std()
        self.pscale *= TASTE
        self.users["P"] *= self.pscale

    # ---------- products ----------
    def add_products(self, rng, n, created, new=False):
        sub = rng.choice(len(SUBCATS), n, p=self.sub_pop)
        brand = np.array([rng.choice(np.flatnonzero(self.brand_cat == SUB_CAT[s])) for s in sub])
        attrs = np.array([rng.choice(len(ATTRS), 2, replace=False) for _ in range(n)])
        q = self.sub_vec[sub] + self.attr_vec[attrs].sum(1) + rng.normal(0, 0.3, (n, D))
        if hasattr(self, "qscale"):
            q *= np.sqrt(self.qscale)
        mult = np.array([ATTRS[a][2] * ATTRS[c][2] for a, c in attrs])
        price = np.maximum(np.floor(np.exp(self.sub_logp[sub] + rng.normal(0, 0.35, n)) * mult) + 0.99, 1.99)
        b = POP * (self.brand_b[brand] + rng.normal(0, 0.65, n))
        taken = set(self.pid)
        for i in range(n):
            while (p := hexid(rng, "p_", 6)) in taken:
                pass
            taken.add(p)
            self.pid.append(p)
            self.desc.append(rng.choice(TEMPLATES).format(brand=self.brands[brand[i]], sub=SUBCATS[sub[i]],
                                                          a1=ATTRS[attrs[i][0]][1], a2=ATTRS[attrs[i][1]][1]))
        self.created += list(created)
        self.sub = np.r_[self.sub, sub]; self.brand = np.r_[self.brand, brand]; self.attrs = np.r_[self.attrs, attrs]
        self.q = np.r_[self.q, q]; self.b = np.r_[self.b, b]; self.base_price = np.r_[self.base_price, price]
        self.price = np.r_[self.price, price]; self.active = np.r_[self.active, np.ones(n, bool)]; self.is_new = np.r_[self.is_new, np.full(n, new)]
        self.pidx = {p: i for i, p in enumerate(self.pid)}
        self._pop_cache = None
        return np.arange(len(self.pid) - n, len(self.pid))

    def product_row(self, i):
        return {"product_id": self.pid[i], "category": CATEGORIES[SUB_CAT[self.sub[i]]], "subcategory": SUBCATS[self.sub[i]],
                "brand": self.brands[self.brand[i]], "price_eur": float(self.price[i]), "description": self.desc[i],
                "created_at": self.created[i], "is_active": bool(self.active[i])}

    @property
    def n(self):
        return len(self.pid)

    def relprice(self):
        return np.log(self.price) - self.sub_logp[self.sub]

    # ---------- users ----------
    def make_users(self, rng, n, prefix):
        seg = rng.choice(len(SEGMENTS), n, p=SEG_W)
        means = np.array([sum(w * self.cat_vec[CATEGORIES.index(c)] for c, w in aff.items()) / len(aff) ** 0.5 for aff, _, _ in SEGMENTS])
        P = means[seg] + rng.normal(0, USER_SD, (n, D))
        if hasattr(self, "pscale"):
            P *= self.pscale
        device = np.array([rng.choice(3, p=SEGMENTS[s][2]) for s in seg])
        region = rng.choice(len(REGIONS), n, p=np.array([16, 14, 18, 11, 5, 6, 3, 3, 4, 5]) / 85)
        beta = np.array([SEGMENTS[s][1] for s in seg]) * rng.lognormal(0, 0.2, n)
        sensitive = np.isin(region, [REGIONS.index(r) for r in PRICE_SENSITIVE]) & (device == 2)
        beta[sensitive] *= 2
        ids = [hexid(rng, prefix, 8) for _ in range(n)]
        return {"id": ids, "P": P, "beta": beta, "device": device, "region": region, "seg": seg, "activity": rng.lognormal(0, 1.0, n)}

    # ---------- utility ----------
    def base_values(self, P, beta, trend=None):
        """(m, n) utility without session terms, satiation or noise."""
        V = self.b + P @ self.q.T - beta[:, None] * self.relprice()
        if trend is not None:
            V = V + trend
        return V

    def session_terms(self, views):
        """(n,) intent + complements given product indices viewed in this session."""
        if len(views) == 0:
            return 0.0
        intent = (self.q[views] @ self.q.T).mean(0)
        comp = COMP[self.sub[views]].any(0)[self.sub]
        return GAMMA * intent + KAPPA * comp

    def trend_vector(self, sub):
        return None if sub is None else TREND * (self.sub == sub)

    # ---------- sessions (shared by history, organic stream and live requests) ----------
    def popularity_sampler(self):
        if self._pop_cache is None:
            w = np.exp(self.b) * self.active
            self._pop_cache = [(idx := np.flatnonzero(self.sub == s), np.cumsum(w[idx]) / max(w[idx].sum(), 1e-9)) for s in range(len(SUBCATS))]
        return self._pop_cache

    def sessions(self, rng, P, beta, n_views, trend=None):
        """Return (V0 rows (m,n), list of view-index lists). Views follow taste, then subcategory / complement browsing."""
        V0 = self.base_values(P, beta, trend)
        noisy = np.where(self.active, V0, -np.inf) + rng.gumbel(size=V0.shape)
        top = np.argpartition(-noisy, 4, axis=1)[:, :4]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(noisy, top, 1), 1), 1)
        pop = self.popularity_sampler()
        out = []
        for r in range(len(P)):
            views, alt = [], list(top[r])
            for k in range(n_views[r]):
                if k == 0:
                    v = alt.pop(0)
                else:
                    x = rng.random()
                    s = self.sub[views[-1]]
                    comps = np.flatnonzero(COMP[self.sub[views[0]]])
                    if x < 0.6 or (x < 0.75 and len(comps) == 0) or not alt:
                        pass
                    elif x < 0.75:
                        s = rng.choice(comps)
                    else:
                        views.append(alt.pop(0))
                        continue
                    idx, cum = pop[s]
                    if len(idx) == 0:
                        continue
                    v = idx[min(np.searchsorted(cum, rng.random()), len(idx) - 1)]
                views.append(int(v))
            out.append(views)
        return V0, out

    # ---------- carousel outcome with common random numbers ----------
    def draws(self, seed, request_id):
        rng = rng_for("crn", seed, request_id)
        return {"exam": rng.random(10), "g": rng.gumbel(size=MAXP), "g0": rng.gumbel(), "buy": rng.random(MAXP)}

    def choose(self, V, items, dr):
        """V: (n,) full utility for this shopper (incl. satiation). items: 10 product indices. -> (clicked idx or -1, purchased)."""
        items = np.asarray(items)
        seen = items[dr["exam"] < EXAM]
        if len(seen) == 0:
            return -1, False
        u = V[seen] + dr["g"][seen]
        k = int(np.argmax(u))
        if u[k] <= U0 + dr["g0"]:
            return -1, False
        i = int(seen[k])
        return i, bool(dr["buy"][i] < sigmoid(V[i] - TAU_BUY))

    # ---------- reference policies ----------
    def bestseller_ranking(self, sales):
        """Frozen fallback/baseline order: purchase count desc, then product_id desc.
        Same order as the public starter (`sorted((n, product_id), reverse=True)`); no hidden tie-break."""
        return np.array(sorted(range(self.n), key=lambda i: (sales[i], self.pid[i]), reverse=True), dtype=int)

    @staticmethod
    def available_top(ranking, active, k=10):
        """Fallback/baseline for one request: the first k products of the frozen ranking in stock right now."""
        ranking = ranking[ranking < len(active)]
        return ranking[active[ranking]][:k]

    def reference(self, p_hat, beta_hat, views, bought=()):
        """Strong organiser recommender: noisy true model, no trend knowledge."""
        if getattr(self, "_qn", np.zeros((0, D))).shape[0] != self.n:
            self._qn = self.q + rng_for("refq", self.seed).normal(0, 0.35, (MAXP, D))[: self.n] * np.sqrt(self.qscale)
        qn = self._qn
        V = self.b + qn @ p_hat - beta_hat * self.relprice()
        if views:
            V = V + GAMMA * (qn[views] @ qn.T).mean(0) + KAPPA * COMP[self.sub[views]].any(0)[self.sub]
        V[list(bought)] -= DELTA
        V[~self.active] = -np.inf
        return np.argsort(-V)[:10]


def calibrate(seed=1, n=6000):
    w = World(seed)
    rng = np.random.default_rng(7)
    # crude sales proxy from organic sessions, like the history would give
    us = rng.choice(len(w.users["id"]), 20_000, p=w.users["activity"] / w.users["activity"].sum())
    V0, ss = w.sessions(rng, w.users["P"][us], w.users["beta"][us], rng.integers(1, 6, len(us)))
    sales = np.zeros(MAXP)
    for r, views in enumerate(ss):
        for v in views:
            if rng.random() < sigmoid(V0[r, v] - TAU_CART) * 0.5:
                sales[v] += 1
    print(f"organic view->purchase rate {sales.sum() / sum(map(len, ss)):.3f}")
    best = w.available_top(w.bestseller_ranking(sales), w.active)
    mean_beta = w.users["beta"].mean()
    new = w.make_users(rng_for("calnew"), n, "n_")
    conv = {"baseline": 0, "reference": 0, "oracle": 0, "random": 0}
    clicks = dict.fromkeys(conv, 0)
    for k in range(n):
        if rng.random() < 0.3:
            P, beta, p_hat = new["P"][k], new["beta"][k], new["P"].mean(0)
        else:
            u = rng.choice(len(w.users["id"]), p=w.users["activity"] / w.users["activity"].sum())
            P, beta = w.users["P"][u], w.users["beta"][u]
            p_hat = P + rng.normal(0, 0.5 * w.pscale, D)
        nv = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.1]))
        V0, (views,) = w.sessions(rng, P[None], np.array([beta]), [nv])
        V = V0[0] + w.session_terms(views)
        dr = w.draws(seed, f"cal{k}")
        policies = {"baseline": best, "reference": w.reference(p_hat, mean_beta, views),
                    "oracle": np.argsort(-np.where(w.active, V, -np.inf))[:10], "random": rng.choice(w.n, 10, replace=False)}
        for name, items in policies.items():
            c, p = w.choose(V, items, dr)
            clicks[name] += c >= 0
            conv[name] += p
    for name in conv:
        print(f"{name:10s} click {100 * clicks[name] / n:5.2f}%  conversion {100 * conv[name] / n:5.2f}%")


# every subset of the 10 slots a shopper may examine, and its probability (slots are examined independently)
SUBSETS = ((np.arange(1 << 10)[:, None] >> np.arange(10)) & 1).astype(bool)
P_SUBSET = np.prod(np.where(SUBSETS, EXAM, 1 - EXAM), axis=1)


def expected(V, items, price):
    """Exact expected (purchases, revenue) of showing `items`: choose() averaged over all its randomness.
    The leaderboard scores this instead of the realised coin flip, so luck does not move the ranking."""
    v = V[np.asarray(items)]
    m = max(v.max(), U0)
    ev = np.exp(v - m)
    p_click = P_SUBSET @ (SUBSETS * ev / (np.exp(U0 - m) + SUBSETS @ ev)[:, None])  # (10,) P(click slot k)
    buy = p_click * sigmoid(v - TAU_BUY)
    return float(buy.sum()), float(buy @ price[np.asarray(items)])


def check_expected(samples=100_000):
    """expected() must match the average of choose() (python world.py check)."""
    rng = np.random.default_rng(3)
    for shift, scale in ((-1, 1.0), (-1, 3.0), (-4, 1.0)):  # the last one is the live regime (~1-3%)
        n = 40
        V = U0 + shift + scale * rng.normal(size=n) + 2 * (np.arange(n) < 10)
        items, price = rng.permutation(n)[:10], rng.uniform(5, 100, n)
        p, r = expected(V, items, price)
        hits = rev = 0.0
        for _ in range(samples):
            dr = {"exam": rng.random(10), "g": rng.gumbel(size=n), "g0": rng.gumbel(), "buy": rng.random(n)}
            i, bought = World.choose(None, V, items, dr)
            hits += bought
            rev += price[i] if bought else 0.0
        se = np.sqrt(p * (1 - p) / samples)
        print(f"expected {p:.4f} simulated {hits / samples:.4f} (se {se:.4f}); revenue {r:.3f} vs {rev / samples:.3f}")
        assert abs(hits / samples - p) < 4 * se, "expected() disagrees with choose()"


if __name__ == "__main__":
    import sys
    check_expected() if sys.argv[1:] == ["check"] else calibrate()
