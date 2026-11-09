#!/usr/bin/env python3
"""swing.py - tests holding surviving pump.fun coins for hours or days instead of minutes.

No wallet, no trades placed. Uses 5, 15 and 60 minute candles from public 1-minute recordings, enters at the
next candle's open and pays fees and slippage. Compares results with random coins of the same age.

    pip install pyarrow pandas numpy
    python swing.py download --days 14
    python swing.py scan
"""
import argparse, calendar, glob, json, os, re, sys, tarfile, tempfile, time, urllib.request
import numpy as np

REPO, DATA = "wahazzin/memecoin-trader", "data_swing"
COST, SLIP, HAIR, CAP = 0.05, 0.02, 0.02, 20.0
BARS, HOLDS, STOPS, TPS = (5, 15, 60), (1, 4, 12, 24, 72), (0.3, 0.5, 0.99), (0.5, 1.0, 3.0, 1e9)
EXITS = [(h, s, t) for h in HOLDS for s in STOPS for t in TPS]
COLS = ["mint", "t", "o", "h", "l", "c", "buy_sol", "sell_sol", "n_buys", "n_sells", "n_users"]
MIN_N = 50


def get(url):
    h = {"User-Agent": "swing-scan"}
    if os.environ.get("GITHUB_TOKEN") and url.startswith("https://api.github.com/"):
        h["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120)


def take(tar_path, out_dir, label):
    with tarfile.open(tar_path) as tar:
        for m in tar:
            base = os.path.basename(m.name)
            if m.isfile() and re.fullmatch(r"candles_[\w.-]+\.parquet", base):
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


def load():
    import pandas as pd, pyarrow.parquet as pq
    fr = []
    for f in sorted(glob.glob(os.path.join(DATA, "*", "*.parquet"))):
        d = pq.read_table(f, columns=COLS).to_pandas()
        for k in COLS[2:]:
            d[k] = d[k].astype("float32")
        fr.append(d)
    if not fr:
        sys.exit("No data. Run: python swing.py download --days 14")
    df = pd.concat(fr, ignore_index=True)
    df["_n"] = df.n_buys + df.n_sells
    df = df.sort_values(["mint", "t", "_n"]).drop_duplicates(["mint", "t"], keep="last").sort_values(["mint", "t"], kind="stable")
    return coins_from(df)


def coins_from(df):
    arr = {k: df[k].values.astype(float) for k in ("t", "o", "h", "l", "c", "buy_sol", "sell_sol", "n_users")}
    m = df.mint.values
    cuts = np.nonzero(m[1:] != m[:-1])[0] + 1
    return [{k: v[a:b] for k, v in arr.items()} for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(m)]) if b - a >= 10]


def bars(c, mins):
    idx = (c["t"] // (mins * 60)).astype(np.int64)
    u, f = np.unique(idx, return_index=True)
    last = np.r_[f[1:] - 1, len(idx) - 1]
    return dict(t=u * mins * 60.0, o=c["o"][f], c=c["c"][last], h=np.maximum.reduceat(c["h"], f), l=np.minimum.reduceat(c["l"], f),
                vol=np.add.reduceat(c["buy_sol"] + c["sell_sol"], f), buy=np.add.reduceat(c["buy_sol"], f), users=np.maximum.reduceat(c["n_users"], f))


def first(m):
    w = np.nonzero(m)[0]
    return int(w[0]) if len(w) else None


def signals(b, t0):
    o, h, l, cl, vol = b["o"], b["h"], b["l"], b["c"], b["vol"]
    idx = np.arange(len(cl))
    peak = np.maximum.accumulate(h)
    cs = np.r_[0.0, np.cumsum(vol)]
    st = np.maximum(idx - 30, 0)
    prev = (cs[idx] - cs[st]) / np.maximum(idx - st, 1)
    age = (b["t"] - t0) / 3600.0
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
        for u in (5, 10, 20):
            j = first((vol >= x * prev) & (cl > o) & (b["users"] >= u) & (idx >= 5))
            if j is not None:
                out[f"volume-burst {x}x users{u}"] = j
    for a_ in (1, 6, 24):
        for x in (1, 2, 5):
            j = first((age >= a_) & (cl >= x * o[0]))
            if j is not None:
                out[f"survivor age{a_}h at{x}x launch"] = j
    return {k: v for k, v in out.items() if v >= 4 and v + 1 < len(cl)}


def trade_all(b, j, data_end):
    """Enter at the next candle's open. Returns one result per exit (None if the data ended first)."""
    t, o, cl = b["t"], b["o"], b["c"]
    e = j + 1
    ep = o[e] * (1 + SLIP)
    out = []
    if ep <= 0:
        return [None] * len(EXITS)
    x = cl[e:] / ep
    for h, s, tp in EXITS:
        end = t[e] + h * 3600
        if end > data_end:
            out.append(None)
            continue
        n = int(np.searchsorted(t[e:], end))
        w = first((x[:n] >= 1 + tp) | (x[:n] <= 1 - s))
        if w is not None:
            b_ = e + w
            px = o[b_ + 1] if b_ + 1 < len(o) else cl[b_] * 0.85
        elif e + n < len(o):
            px = o[e + n]
        else:
            px = cl[-1] * 0.85       # stopped trading before the hold ended: treat as hard to sell
        out.append(float(min(px * (1 - HAIR) / ep - 1 - COST, CAP)))
    return out


def run(coins, cut1, cut2, data_end):
    rng = np.random.default_rng(5)
    dur = np.array([c["t"][-1] - c["t"][0] for c in coins])
    order, sd = np.argsort(-dur), -np.sort(-dur)
    S, N, rows = {}, {}, {}
    for B in BARS:
        cache = {}
        for ci, c in enumerate(coins):
            t0 = c["t"][0]
            b = bars(c, B)
            for name, j in signals(b, t0).items():
                when = b["t"][j]
                age = when - t0
                p = 0 if when < cut1 else (1 if when < cut2 else 2)
                rs = trade_all(b, j, data_end)
                na = int(np.searchsorted(-sd, -age, side="right"))
                rn = None
                if na:
                    oi = int(order[rng.integers(na)])
                    ob = cache.get(oi)
                    if ob is None:
                        ob = cache[oi] = bars(coins[oi], B)
                    k = int(np.searchsorted(ob["t"], coins[oi]["t"][0] + age))
                    if 4 <= k and k + 1 < len(ob["t"]):
                        rn = trade_all(ob, k, data_end)
                for store, vals in ((S, rs), (N, rn)):
                    if vals is None:
                        continue
                    for ex, r in zip(EXITS, vals):
                        if r is None:
                            continue
                        key = (B, name, ex)
                        v = store.get(key)
                        if v is None:
                            v = store[key] = np.zeros((3, 2))
                        v[p, 0] += 1
                        v[p, 1] += r
                        if store is S:
                            rows.setdefault(key, []).append((when, r))
    return S, N, rows


def passed(S):
    return sorted(((float((v[:, 1] / v[:, 0]).min()), k) for k, v in S.items() if v[:, 0].min() >= MIN_N and (v[:, 1] / v[:, 0]).min() > 0), key=lambda x: -x[0])


def label(k):
    B, n, (h, s, t) = k
    return f"{B}m candles | {n} | hold up to {h}h, stop " + ("none" if s > 0.9 else f"-{int(s*100)}%") + ", " + ("no target" if t > 1e6 else f"take +{int(t*100)}%")


def account(rows, stake, hold_h):
    eq, ends = 200.0, []
    for t, r in sorted(rows):
        ends = [e for e in ends if e > t]
        if len(ends) < int(1 / stake):
            ends.append(t + hold_h * 3600)
            eq = max(eq * (1 + stake * r), 0.0)
    return eq


def scan():
    start = time.time()
    days = sorted(glob.glob(os.path.join(DATA, "20*")))
    if not days:
        sys.exit("No data. Run: python swing.py download --days 14")
    ts = lambda d: calendar.timegm(time.strptime(os.path.basename(d), "%Y-%m-%d"))
    lo, hi = ts(days[0]), ts(days[-1]) + 86400
    cut1, cut2 = lo + 0.6 * (hi - lo), lo + 0.8 * (hi - lo)
    coins = load()
    S, N, rows = run(coins, cut1, cut2, hi)
    good, null = passed(S), passed(N)
    means = lambda v: " / ".join(f"{x*100:+.1f}%" for x in v[:, 1] / v[:, 0]) + "   trades: " + " / ".join(str(int(x)) for x in v[:, 0])
    conf = [k for _, k in good if len(rows[k]) > 5 and sum(sorted(r for _, r in rows[k])[:-5]) > 0]
    L = [f"{len(days)} days, {len(coins):,} coins. Costs: {COST*100:.0f}% round trip plus {SLIP*100:.0f}% slippage each way.",
         f"Rule versions: {len(S):,}. Positive in all 3 periods: {len(good)}. Random coins of the same age, same test: {len(null)}.",
         f"Also positive without the best 5 trades: {len(conf)}."]
    for k in conf[:8]:
        test = [x for x in rows[k] if x[0] >= cut2]
        L += ["", label(k), "  per trade: " + means(S[k]), "  last period on GBP200: " + ", ".join(f"{int(f*100)}% stake -> GBP{account(test, f, k[2][0]):.0f}" for f in (0.05, 0.1, 0.25))]
    near = sorted(((float((v[:, 1] / v[:, 0]).min()), k) for k, v in S.items() if v[:, 0].min() >= MIN_N), key=lambda x: -x[0])[:6]
    L += ["", "Closest rules:"]
    for _, k in near:
        L += [label(k), "  per trade: " + means(S[k])]
    L += ["", "VERDICT: " + ("candidates above; paper-test the best before risking anything." if conf and len(good) > 2 * len(null) else "no rule clearly beat random coins of the same age. Do not trade these with real money."), f"(took {(time.time()-start)/60:.0f} min)"]
    open("swing_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("download").add_argument("--days", type=int, default=14)
    s.add_parser("scan")
    a = p.parse_args()
    download(a.days) if a.cmd == "download" else scan()
