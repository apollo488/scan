#!/usr/bin/env python3
"""fast.py - tests the same fast rules on 5, 15 and 30 second candles built from every real trade.

No wallet, no trades placed. Fills use the real pool balance at each moment, 1 second late, with
fees. Stops are checked on every trade, so a crash inside a candle is caught as it happens.

    pip install pyarrow pandas numpy
    python fast.py download --days 7
    python fast.py scan
"""
import argparse, calendar, glob, json, multiprocessing as mp, os, re, sys, tarfile, tempfile, time, urllib.request
import numpy as np

REPO, DATA = "wahazzin/memecoin-trader", "data_fast"
FEE, FIX, SIZE, LAT, CAP = 0.0125, 0.0012, 0.1, 1.0, 3.0     # fee per side, SOL per tx, trade size, delay (s), max +300%
BARS, HOLDS, STOPS, TPS = (5, 15, 30), (30, 120, 300), (0.1, 0.2), (0.1, 0.3, 1e9)
EXITS = [(h, s, t) for h in HOLDS for s in STOPS for t in TPS]
AGES = [("any", 0, 1e12), ("<2m", 0, 120), ("2-10m", 120, 600), ("10m+", 600, 1e12)]
COLS = ["ts", "mint", "buy", "sol", "vsol", "vtok", "slot"]
MIN_N = 50


def get(url):
    h = {"User-Agent": "fast-scan"}
    if os.environ.get("GITHUB_TOKEN") and url.startswith("https://api.github.com/"):
        h["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120)


def take(tar_path, out_dir, label):
    with tarfile.open(tar_path) as tar:
        for m in tar:
            base = os.path.basename(m.name)
            if m.isfile() and re.fullmatch(r"trades_[\w.-]+\.parquet", base):
                with open(os.path.join(out_dir, re.sub(r"\W", "_", label) + "_" + base), "wb") as f:
                    f.write(tar.extractfile(m).read())


def download(days):
    rel = []
    for page in range(1, 4):
        b = json.load(get(f"https://api.github.com/repos/{REPO}/releases?per_page=100&page={page}"))
        rel += b
        if len(b) < 100:
            break
    today = "data-" + time.strftime("%Y-%m-%d", time.gmtime())
    rel = sorted((r for r in rel if re.fullmatch(r"data-\d{4}-\d{2}-\d{2}", r["tag_name"]) and r["tag_name"] != today), key=lambda r: r["tag_name"])[-days:]
    for r in rel:
        d = os.path.join(DATA, r["tag_name"][5:])
        os.makedirs(d, exist_ok=True)
        for a in (x for x in r["assets"] if x["name"].endswith(".tar")):
            mark = os.path.join(d, a["name"] + ".done")
            if os.path.exists(mark):
                continue
            for _ in range(3):
                fd, path = tempfile.mkstemp(dir=d, suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as tmp, get(a["browser_download_url"]) as src:
                        while True:
                            c = src.read(1 << 20)
                            if not c:
                                break
                            tmp.write(c)
                    take(path, d, a["name"])
                    open(mark, "w").close()
                    break
                except Exception as e:
                    print("retry:", e, flush=True)
                    time.sleep(5)
                finally:
                    if os.path.exists(path):
                        os.remove(path)
        print("got", r["tag_name"][5:], flush=True)


def load_day(d):
    import pandas as pd, pyarrow.parquet as pq
    fr = [pq.read_table(f, columns=COLS).to_pandas() for f in sorted(glob.glob(os.path.join(d, "*.parquet")))]
    df = pd.concat(fr, ignore_index=True)
    del fr
    df = df[(df.vsol > 0) & (df.vtok > 0)].drop_duplicates(["mint", "slot", "sol", "buy", "vsol"]).sort_values(["mint", "slot"], kind="stable")
    return coins_from(df)


def coins_from(df):
    ts, v, k, sol, buy, mint = df.ts.values.astype(float), df.vsol.values.astype(float), df.vtok.values.astype(float), df.sol.values.astype(float), df.buy.values.astype(bool), df.mint.values
    cuts = np.nonzero(mint[1:] != mint[:-1])[0] + 1
    out = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(mint)]):
        if b - a >= 5:       # no rule can fire on fewer trades, so this adds no look-ahead
            out.append(dict(ts=ts[a:b], v=v[a:b], k=k[a:b], sol=sol[a:b], buy=buy[a:b], t0=ts[a]))
    return out


def make_bars(c, B):
    ts, p = c["ts"], c["v"] / c["k"]
    idx = ((ts - c["t0"]) // B).astype(np.int64)
    u, first = np.unique(idx, return_index=True)
    last = np.r_[first[1:] - 1, len(ts) - 1]
    return dict(end=c["t0"] + (u + 1) * B, o=p[first], c=p[last], h=np.maximum.reduceat(p, first), l=np.minimum.reduceat(p, first),
                buy=np.add.reduceat(c["sol"] * c["buy"], first), sell=np.add.reduceat(c["sol"] * ~c["buy"], first), n=np.diff(np.r_[first, len(ts)]))


def first(m):
    w = np.nonzero(m)[0]
    return int(w[0]) if len(w) else None


def signals(b):
    o, h, l, cl = b["o"], b["h"], b["l"], b["c"]
    idx = np.arange(len(cl))
    peak = np.maximum.accumulate(h)
    vol = b["buy"] + b["sell"]
    cs = np.r_[0.0, np.cumsum(vol)]
    st = np.maximum(idx - 30, 0)
    prev = (cs[idx] - cs[st]) / np.maximum(idx - st, 1)
    out = {}
    for dd in (0.3, 0.5, 0.7):
        i = first((l <= peak * (1 - dd)) & (idx >= 1))
        if i is None:
            continue
        pk = peak[i]
        for bk in (0.1, 0.2, 0.3):
            j = first((idx > i) & (cl >= pk * (1 - bk)))
            if j is not None:
                out[f"second-leg fall{int(dd*100)} back-within{int(bk*100)}"] = j
        j = first((idx > i) & (cl > o))
        if j is not None:
            out[f"dip fall{int(dd*100)} first-green"] = j
    ph = np.r_[np.inf, peak[:-1]]
    for k in (5, 10, 30):
        for x in (1, 3, 5):
            j = first((cl > ph) & (idx >= k) & (vol >= x * prev))
            if j is not None:
                out[f"breakout after{k}bars vol{x}x"] = j
    for x in (3, 5, 10):
        for u in (3, 5, 10):
            j = first((vol >= x * prev) & (cl > o) & (b["n"] >= u) & (idx >= 5))
            if j is not None:
                out[f"volume-burst {x}x trades{u}"] = j
    with np.errstate(divide="ignore", invalid="ignore"):
        s = b["buy"] / vol
    for th in (0.7, 0.8):
        for u in (3, 5):
            j = first((s >= th) & (b["n"] >= u) & (cl > o) & (idx >= 5))
            if j is not None:
                out[f"buy-pressure {int(th*100)}% trades{u}"] = j
    return {k: v for k, v in out.items() if v >= 4}


def trades(c, when):
    """All exit versions for one entry made LAT seconds after `when`. Returns a list or None."""
    ts, v, k = c["ts"], c["v"], c["k"]
    i = int(np.searchsorted(ts, when + LAT))
    if i >= len(ts):
        return None
    net = SIZE * (1 - FEE)
    tok = k[i] - v[i] * k[i] / (v[i] + net)
    lim = int(np.searchsorted(ts, ts[i] + max(HOLDS) + 60))
    tw = ts[i + 1:lim]
    dead = (net * 0.85 * (1 - FEE) - 2 * FIX) / SIZE - 1
    if len(tw) == 0:
        return [dead] * len(EXITS)
    a, b_ = v[i + 1:lim] + net, np.maximum(k[i + 1:lim] - tok, 1.0)
    r = ((a - a * b_ / (b_ + tok)) * (1 - FEE) - 2 * FIX) / SIZE - 1
    out = []
    for hold, stop, tp in EXITS:
        end = ts[i] + hold
        n = int(np.searchsorted(tw, end))
        t = first((r[:n] >= tp) | (r[:n] <= -stop))
        if t is not None:
            e = int(np.searchsorted(tw, tw[t] + LAT))
        else:
            e = n
        val = r[e] if e < len(r) else (1 + r[-1]) * 0.85 - 1
        out.append(float(min(val, CAP)))
    return out


def scan_day(a):
    d, cut1, cut2, seed = a
    coins = load_day(d)
    rng = np.random.default_rng(seed)
    dur = np.array([c["ts"][-1] - c["t0"] for c in coins])
    order = np.argsort(-dur)
    sdur = -np.sort(-dur)
    S, N = {}, {}
    for c in coins:
        for B in BARS:
            b = make_bars(c, B)
            for name, j in signals(b).items():
                when = float(b["end"][j])
                age = when - c["t0"]
                p = 0 if when < cut1 else (1 if when < cut2 else 2)
                groups = [g[0] for g in AGES if g[1] <= age < g[2]]
                rs = trades(c, when)
                # fair control: same moment of life, on a random coin still trading at that age
                n_alive = int(np.searchsorted(-sdur, -(age + LAT), side="right"))
                other = coins[order[rng.integers(n_alive)]] if n_alive else None
                rn = trades(other, other["t0"] + age) if other is not None else None
                for store, vals in ((S, rs), (N, rn)):
                    if vals is None:
                        continue
                    for ex, r in zip(EXITS, vals):
                        for g in groups:
                            key = (B, name, ex, g)
                            x = store.get(key)
                            if x is None:
                                x = store[key] = np.zeros((3, 2))
                            x[p, 0] += 1
                            x[p, 1] += r
    return S, N, len(coins)


def detail_day(a):
    d, keys = a
    want = {}
    for k in keys:
        want.setdefault(k[0], []).append(k)
    ag = {g[0]: g for g in AGES}
    out = {k: [] for k in keys}
    for c in load_day(d):
        for B, ks in want.items():
            sig = signals(make_bars(c, B))
            bars = None
            for k in ks:
                j = sig.get(k[1])
                if j is None:
                    continue
                bars = bars or make_bars(c, B)
                when = float(bars["end"][j])
                if ag[k[3]][1] <= when - c["t0"] < ag[k[3]][2]:
                    rs = trades(c, when)
                    if rs is not None:
                        out[k].append((when, rs[EXITS.index(k[2])]))
    return out


def passed(S):
    return sorted(((float((v[:, 1] / v[:, 0]).min()), k) for k, v in S.items() if v[:, 0].min() >= MIN_N and (v[:, 1] / v[:, 0]).min() > 0), key=lambda x: -x[0])


def label(k):
    B, n, (h, s, t), g = k
    return f"{B}s candles | {n} | hold {h}s, stop -{int(s*100)}%, " + ("no target" if t > 1e6 else f"take +{int(t*100)}%") + f" | age {g}"


def account(rows, stake, hold):
    eq, ends = 200.0, []
    for t, r in rows:
        ends = [e for e in ends if e > t]
        if len(ends) < int(1 / stake):
            ends.append(t + hold)
            eq = max(eq * (1 + stake * r), 0.0)
    return eq


def scan():
    t_start = time.time()
    days = sorted(d for d in glob.glob(os.path.join(DATA, "20*")) if glob.glob(os.path.join(d, "*.parquet")))
    if not days:
        sys.exit("No data. Run: python fast.py download --days 7")
    ts = lambda d: calendar.timegm(time.strptime(os.path.basename(d), "%Y-%m-%d"))
    lo, hi = ts(days[0]), ts(days[-1]) + 86400
    cut1, cut2 = lo + 0.6 * (hi - lo), lo + 0.8 * (hi - lo)
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(scan_day, [(d, cut1, cut2, 11 + i) for i, d in enumerate(days)], chunksize=1)
        S, N, n = {}, {}, 0
        for s, x, c in res:
            n += c
            for dst, src in ((S, s), (N, x)):
                for k, v in src.items():
                    dst[k] = dst[k] + v if k in dst else v
        good, null = passed(S), passed(N)
        keys = [k for _, k in good[:100]]
        rows = {k: [] for k in keys}
        if keys:
            for part in pool.map(detail_day, [(d, keys) for d in days], chunksize=1):
                for k, v in part.items():
                    rows[k] += v
    means = lambda v: " / ".join(f"{x*100:+.1f}%" for x in v[:, 1] / v[:, 0]) + "   trades: " + " / ".join(str(int(x)) for x in v[:, 0])
    L = [f"Data {len(days)} days, coins {n:,}, fills at real pool prices, {LAT:.0f}s late, {FEE*200:.1f}% fees plus {2*FIX} SOL per round trip on {SIZE} SOL.",
         f"Rule versions: {len(S):,}. Positive in all 3 periods: {len(good)}. Fair random control, same test: {len(null)}."]
    conf = [(k, sorted(rows[k])) for _, k in good[:100] if len(rows[k]) > 5 and sum(sorted(r for _, r in rows[k])[:-5]) > 0]
    L.append(f"Also positive without the best 5 trades: {len(conf)}.")
    for k, rr in conf[:10]:
        test = [x for x in rr if x[0] >= cut2]
        L += ["", label(k), "  per trade: " + means(S[k]), "  last period on GBP200: " + ", ".join(f"{int(f*100)}% stake -> GBP{account(test, f, k[2][0]):.0f}" for f in (0.05, 0.1, 0.25))]
    near = sorted(((float((v[:, 1] / v[:, 0]).min()), k) for k, v in S.items() if v[:, 0].min() >= MIN_N), key=lambda x: -x[0])[:6]
    L += ["", "Closest rules (worst period shown first in each ranking):"]
    for _, k in near:
        L += [label(k), "  per trade: " + means(S[k])]
    L += ["", "VERDICT: " + ("candidates above; paper-test the best before risking anything." if conf and len(good) > len(null) else "no rule beat the random control. Do not trade these with real money."),
          f"(took {(time.time()-t_start)/60:.0f} min)"]
    open("fast_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("download").add_argument("--days", type=int, default=7)
    s.add_parser("scan")
    a = p.parse_args()
    download(a.days) if a.cmd == "download" else scan()
