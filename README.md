# RawTree challenge · Datathon FME 2026 · the simulator

This is the code behind the **RawTree real-time recommendation challenge** at Datathon FME 2026 (UPC Barcelona, 14–15 November 2026). For 26 hours, 42 teams answered every shopper of a simulated store in under 3 seconds, without knowing how those shoppers decided what to click and buy. Now that the challenge is over, here is the hidden model, the live simulator and the scoring, so you can see what you were up against.

- Brief and starter kit: https://github.com/rawtreedb/datathon-fme-2026
- Challenge page and final leaderboard: https://datathon.rawtree.dev

## Results

Final round, Sunday 10:00–12:00 CET, 21,600 shoppers, score out of 100 (the jury added up to 30 more):

| | Team | Score | Conversion |
|---|---|---|---|
| 🥇 | Vamos Breakfast | 58.3 | 2.252% |
| 🥈 | Skore Money | 57.7 | 2.238% |
| 🥉 | toievskithon | 56.2 | 2.189% |

On the same shoppers, the **baseline** (best sellers) converted 0.75%, the **organisers' reference** 2.14% and the **oracle** 3.61%. Four teams beat the reference. The full ranking is on the leaderboard.

## How it worked

```
                 seeds (WORLD_SEED, SEED_A practice, SEED_B final)
                            │
                 ┌──────────▼───────────┐
                 │     simulator.py     │  requests 3/s, organic events ~50/s, catalogue changes
                 └──┬──────────────▲────┘  decides clicks and purchases with world.py
     shop.requests  │              │ polls team_NN.submissions every 0.5 s
     shop.events    │              │
     shop.catalog_… ▼              │
      ┌────────────────┐   ┌───────┴──────────┐   ┌───────────────────────┐
      │    RawTree     │   │ team_NN.feedback │   │ organizer.outcomes    │──► leaderboard
      │    cluster     │   │ (only team NN)   │   │ (ground truth)        │    (leaderboard/state.py)
      └───────┬────────┘   └──────────────────┘   └───────────────────────┘
              │ read shop.*, write team_NN.submissions
         your systems
```

Everything is a pure function of the seeds and the clock, so the simulator can restart at any time and resume where the clock is.

## The hidden model (`simulator/world.py`)

**The world.** 5,000 products in 10 categories and 90 subcategories, 60,000 shoppers with history, and a pool of 20,000 new shoppers per round. Products and shoppers live in the same 16-dimensional taste space.

- **Products.** A product's vector is its subcategory's vector, plus the vectors of the two attributes in its description ("premium materials", "great value for money", "made from recycled materials"…), plus noise. Its appeal `b` is `POP × (brand appeal + noise)`. Its price comes from the subcategory's typical price times the attributes' multipliers ("luxury" ×2.2, "budget" ×0.6…).
- **Shoppers.** Seven segments (electronics fans, home and family, fashion and beauty, sports, pets, books and media, bargain hunters), each with category affinities and a price sensitivity `β`. A shopper's taste is their segment's mean plus personal noise. Android shoppers in Andalucía, Murcia and Canarias are twice as price sensitive.

**Utility** of product *i* for shopper *u* in one request:

```
V = b_i                                  product appeal (brand + product)
  + P_u · q_i                            personal taste, scaled to a spread of TASTE
  − β_u · relprice_i                     price relative to its subcategory (log)
  + GAMMA · mean(q_v · q_i)              session intent: similarity to every product viewed in this session
  + KAPPA · [i complements a viewed subcategory]   "goes with" pairs: case after a phone, capsules after a coffee machine…
  + TREND · [i in the trending subcategory]
  − DELTA · [u already bought i]         satiation
```

**Choice** (`choose`). The shopper examines slot *k* (0 = first) with probability `1 / log2(k + 2)`: always the first slot, 63% the second, 29% the last. Among the products they examined and the option of buying nothing (utility `U0`), each with Gumbel noise, they click the best one. After a click they buy it with probability `sigmoid(V − TAU_BUY)`.

**Scoring without luck** (`expected`). The leaderboard did not count the coin flips. It used the exact probability of a purchase for the list each team showed, so the same list always scored the same.

| Constant | Value | What it does |
|---|---|---|
| `TASTE` | 1.6 | Weight of personal taste |
| `POP` | 0.7 | Weight of product appeal |
| `GAMMA` | 0.35 | Session intent |
| `KAPPA` | 1.2 | Complements (30 subcategory pairs) |
| `TREND` | 1.6 | Boost of the trending subcategory |
| `DELTA` | 4.0 | Satiation: already bought (the largest penalty in the model) |
| `U0` | 10.0 | Utility of buying nothing |
| `TAU_BUY` | 8.2 | Click → purchase threshold |

### The three benchmarks

- **Baseline**, which was also the **fallback** for missing, late or invalid answers: the first 10 products in stock of the all-time best-seller ranking, frozen when the stream started. It ignores the shopper, the session and the catalogue changes.
- **Reference**: the organisers' recommender. It uses the true model, but with noisy product vectors and noisy shopper tastes; it avoids what the shopper already bought and knows nothing about trends.
- **Oracle**: the 10 products with the highest true utility for that shopper, in order. Nobody can beat it.

`python simulator/world.py` measures them on a calibration population: baseline 3.0%, reference 5.4%, oracle 7.1% conversion. Live they were lower (0.75%, 2.14% and 3.61% in the final), because returning shoppers had already bought many of the best sellers and the store kept changing.

## The live stream (`simulator/simulator.py`)

- **Requests.** 3 per second. 30% come from new shoppers (`is_new_user`); the rest from shoppers with history, chosen by how active they are. `recent_views` are simulated with the same model.
- **Organic activity.** About 50 events per second (view → add to cart → purchase) in `shop.events`.
- **The store moved** (practice round):
  - every 5 minutes, 3 products go out of stock (best sellers more often) and come back an hour later;
  - 100 new products every 4 hours;
  - a 20% price drop on a whole category for 2 hours, every 3 hours;
  - a 45-minute trend on one subcategory every 2 hours.
- **The final round** (`SEED_B`): a new pool of new shoppers, 300 new products at 10:00, one trend (T-Shirts, 11:06–11:51 CET) and one price drop. Stock-outs continued as before.
- **Timing.** An answer had to reach RawTree within 3 s of the request's `ts`. The simulator polled every 0.5 s, so answers seen within 3.5 s counted, and resolved each request 4.2 s after its `ts`. A late answer was `late`; one that never arrived, or arrived after the request was resolved, was `missing`.
- **Validity.** Exactly 10 distinct products, all in stock when the request was made. From Saturday 16:44, a product that went out of stock less than 5 s before the request was accepted, but could not be bought (see below).
- **Satiation per team.** Each team had its own copy of each shopper: what a shopper bought through your recommendations only affected your later requests. Organic purchases affected everyone.

## Scoring (`leaderboard/state.py`)

```
conversion      = expected purchases per 100 requests
cold start      = the same, on cold requests (a new shopper, or a session that viewed a product added during the event)
reliability     = % of answers valid and on time
score (0-100)   = (50 · conversion score + 10 · cold-start score + 10 · reliability) / 70
```

The conversion and cold-start scores go from 0 (the baseline) to 100 (the oracle), clipped to that range. Ties were broken by expected revenue per request. Only the final round counted.

## What mattered

How much each idea was worth, according to the model (the organisers' notes, most important first):

| Idea | Why |
|---|---|
| **Stay up** | Every missing, late or invalid answer shows the baseline instead of your list, and reliability is 10 of the 70 points. |
| **Don't recommend what they already bought** | Satiation (`DELTA = 4`) is the largest penalty in the model: purchases are in `history_events` and in your `feedback`. |
| **Personalise by category and price** | Personal taste and price sensitivity are the biggest share of the gap between the baseline and the reference. |
| **Order your list** | The first slot is always examined, the last one only 29% of the time. |
| **Use the whole session** | Session intent averages over *every* product viewed, not just the last one. |
| **React to the catalogue** | Stock-outs (invalid answers), price drops (price enters the utility directly) and new products (cold start). |
| Complements | Large weight, but only fires when the session hits a pair. |
| Cold start and descriptions | The description words carry the attribute vectors: the only taste signal for new products and new shoppers. |
| Trends | One subcategory gets +1.6 for 45 minutes; visible early as a surge in `shop.events`. |
| Brands, device and region | Real but small effects. |

What the data says: in the final round, the top 10 teams filled 13.3% of their lists with new products (the others 5.0%), showed products the shopper had already bought 0.7% of the time (others 2.9%), and answered 99.6% of requests on time (others 96.4%).

## Known quirks, found during the event

- **Stock-outs reached `shop.catalog_updates` 0.3–1.8 s late.** Answers were judged against the stock at the request's `ts`, so for a moment after each stock-out an answer could be invalid for a product teams could not yet know had sold out. Fixed on Saturday at 16:44 with the 5 s grace (`GRACE` in `simulator.py`). Thanks to the team that reported it.
- **Repeat purchases in the data.** The generated history and the live organic events do not apply satiation, so shoppers there re-buy products; only the carousel (what was scored) does. Purchases in `history_sessions` (the old carousel) are not counted for satiation either.

## Run it yourself

```sh
pip install -r simulator/requirements.txt
cd simulator
python world.py              # calibration: baseline, reference, oracle and random conversion
python world.py check        # checks that expected() matches simulated choices
python test_baseline.py      # fallback and validity rules
```

To replay the whole challenge you need a RawTree cluster and its admin key:

```sh
cd simulator
cp .env.example .env         # admin key, seeds and times; then: set -a; . ./.env; set +a
python admin.py init         # shop, organizer: databases and tables
python generate.py --out data/ && python admin.py load data/   # catalogue + 60 days of history
python admin.py add-team 3   # team_01..team_03: database, tables, role and key each
python simulator.py          # the live stream, between PRACTICE_START and FINAL_END
```

### Reproduce the event

These are the seeds used at Datathon FME 2026 (they are also in `simulator/.env.example`, with the event's times):

```
WORLD_SEED=329180804   # catalogue, shoppers and the 60-day history
SEED_A=805223772       # practice round stream (Saturday 10:00 → Sunday 10:00 CET)
SEED_B=2062992712       # final round stream (Sunday 10:00–12:00 CET)
```

With them, `generate.py` rebuilds the same catalogue and history, and `simulator.py` replays the same shoppers, trends, price drops, new products and stock-outs at the same times, so you can score any list of answers against the final round.

| File | What it is |
|---|---|
| `simulator/world.py` | The hidden model: catalogue, shoppers, utility, choice, benchmarks, calibration |
| `simulator/simulator.py` | The live stream: requests, organic events, catalogue changes, polling, validation, outcomes, feedback |
| `simulator/generate.py` | The catalogue and 60 days of history (`history_events`, `history_sessions`) |
| `simulator/admin.py` | RawTree setup: databases, tables, team roles and keys, loading, status |
| `simulator/rawtree.py` | Minimal RawTree HTTP client |
| `simulator/test_baseline.py` | Tests for the fallback and validity rules |
| `leaderboard/state.py` | The leaderboard's scoring and phases (a Vercel Python function) |
