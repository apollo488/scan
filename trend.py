#!/usr/bin/env python3
"""trend.py - tests buying coins when a new name suddenly gets launched several times in minutes.

That burst is what a viral post leaves on chain: launch bots copy the name within seconds, so several
coins appear with the same ticker. No wallet, no trades placed. Fills use the real pool balance, 1 second
late, with fees. It compares every result with similar coins launched at the same time.

    pip install pyarrow pandas numpy
    python trend.py download --days 7
    python trend.py scan
"""
import argparse, calendar, collections, glob, heapq, json, multiprocessing as mp, os, re, sys, tarfile, tempfile, time, urllib.request
import numpy as np

REPO, DATA = "wahazzin/memecoin-trader", "data_trend"
FEE, FIX, SIZE, LAT, CAP = 0.0125, 0.0012, 0.1, 1.0, 9.0
WINDOWS, COUNTS, MODES = (60, 300, 900), (2, 3, 5, 10), ("first", "newest", "biggest", "all")
EXITS = [(h, s, t) for h in (30, 120, 300, 900) for s in (0.1, 0.2, 0.5) for t in (0.1, 0.3, 1.0, 1e9)]
MIN_N, QUIET_S, QUIET_MAX = 30, 6 * 3600, 2
TCOLS = ["ts", "mint", "buy", "sol", "vsol", "vtok", "slot"]


def get(url):
    h = {"User-Agent": "trend-scan"}
    if os.environ.get("GITHUB_TOKEN") and url.startswith("https://api.github.com/"):
        h["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120)


def take(tar_path, out_dir, label):
    with tarfile.open(tar_path) as tar:
        for m in tar:
            base = os.path.basename(m.name)
            if m.isfile() and re.fullmatch(r"(trades|creates)_[\w.-]+\.parquet", base):
                with open(os.path.join(out_dir, base.split("_")[0] + "__" + re.sub(r"\W", "_", label) + "_" + base), "wb") as f:
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


def read(d, kind, cols):
    import pandas as pd, pyarrow.parquet as pq
    fr = [pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(os.path.join(d, kind + "__*.parquet")))]
    return pd.concat(fr, ignore_index=True) if fr else None


def load_day(d):
    cr = read(d, "creates", ["ts", "mint", "symbol"])
    tr = read(d, "trades", TCOLS)
    tr = tr[(tr.vsol > 0) & (tr.vtok > 0)].drop_duplicates(["mint", "slot", "sol", "buy", "vsol"]).sort_values(["mint", "slot"], kind="stable")
    return cr.drop_duplicates("mint").sort_values("ts"), coins_from(tr)


def coins_from(df):
    ts, v, k, sol, mint = df.ts.values.astype(float), df.vsol.values.astype(float), df.vtok.values.astype(float), df.sol.values.astype(float), df.mint.values
    cuts = np.nonzero(mint[1:] != mint[:-1])[0] + 1
    return {mint[a]: dict(ts=ts[a:b], v=v[a:b], k=k[a:b], sol=sol[a:b]) for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(mint)])}


def key_of(sym):
    return re.sub(r"[^a-z0-9]", "", str(sym).lower())


def bursts(cr):
    """Every moment a quiet ticker reaches COUNT launches inside WINDOW seconds. Uses only launches already seen."""
    hist, cool, out = collections.defaultdict(list), {}, []
    for t, mint, sym in zip(cr.ts.values.astype(float), cr.mint.values, cr.symbol.values):
        key = key_of(sym)
        if len(key) < 2:
            continue
        h = hist[key]
        h.append((t, mint))
        while h and h[0][0] < t - QUIET_S - max(WINDOWS):
            h.pop(0)
        for W in WINDOWS:
            recent = [m for x, m in h if x >= t - W]
            before = sum(1 for x, _ in h if t - W - QUIET_S <= x < t - W)
            for C in COUNTS:
                if len(recent) >= C and before <= QUIET_MAX and cool.get((W, C, key), 0) <= t:
                    cool[(W, C, key)] = t + 3600
                    out.append((t, W, C, key, list(recent)))
    return out


def trades(c, when):
    ts, v, k = c["ts"], c["v"], c["k"]
    i = int(np.searchsorted(ts, when + LAT))
    net = SIZE * (1 - FEE)
    dead = (net * 0.85 * (1 - FEE) - 2 * FIX) / SIZE - 1
    if i >= len(ts):
        return [dead] * len(EXITS)
    tok = k[i] - v[i] * k[i] / (v[i] + net)
    lim = int(np.searchsorted(ts, ts[i] + 960))
    tw = ts[i + 1:lim]
    if len(tw) == 0:
        return [dead] * len(EXITS)
    a, b_ = v[i + 1:lim] + net, np.maximum(k[i + 1:lim] - tok, 1.0)
    r = ((a - a * b_ / (b_ + tok)) * (1 - FEE) - 2 * FIX) / SIZE - 1
    out = []
    for hold, stop, tp in EXITS:
        n = int(np.searchsorted(tw, ts[i] + hold))
        w = np.nonzero((r[:n] >= tp) | (r[:n] <= -stop))[0]
        e = int(np.searchsorted(tw, tw[w[0]] + LAT)) if len(w) else n
        out.append(float(min(r[e] if e < len(r) else (1 + r[-1]) * 0.85 - 1, CAP)))
    return out


class Tally:
    __slots__ = ("n", "s", "top")

    def __init__(self):
        self.n, self.s, self.top = np.zeros(3), np.zeros(3), []

    def add(self, p, r):
        self.n[p] += 1
        self.s[p] += r
        if len(self.top) < 5:
            heapq.heappush(self.top, r)
        elif r > self.top[0]:
            heapq.heapreplace(self.top, r)

    def merge(self, o):
        self.n += o.n
        self.s += o.s
        self.top = heapq.nlargest(5, self.top + o.top)
        heapq.heapify(self.top)

    def without_best(self):
        return (self.s.sum() - sum(self.top)) / max(self.n.sum() - 5, 1)


def scan_day(a):
    d, cut1, cut2, seed = a
    cr, coins = load_day(d)
    rng = np.random.default_rng(seed)
    birth = {m: float(t) for m, t in zip(cr.mint.values, cr.ts.values)}
    times = cr.ts.values.astype(float)
    mints = cr.mint.values
    S, N = collections.defaultdict(Tally), collections.defaultdict(Tally)
    names, nb = collections.Counter(), 0
    for t, W, C, key, members in bursts(cr):
        have = [m for m in members if m in coins]
        if not have:
            continue
        nb += 1
        names[key] += 1
        p = 0 if t < cut1 else (1 if t < cut2 else 2)
        sold = {m: coins[m]["sol"][coins[m]["ts"] <= t].sum() for m in have}
        for mode in MODES:
            picks = have if mode == "all" else [have[0] if mode == "first" else have[-1] if mode == "newest" else max(have, key=lambda m: sold[m])]
            for m in picks:
                rs = trades(coins[m], t)
                age = t - birth.get(m, t)
                lo, hi = int(np.searchsorted(times, t - age - 120)), int(np.searchsorted(times, t - age + 120))
                rn = None
                for _ in range(8):
                    if hi <= lo:
                        break
                    o = mints[rng.integers(lo, hi)]
                    if o in coins and o not in members:
                        rn = trades(coins[o], birth.get(o, t) + age)
                        break
                for store, vals in ((S, rs), (N, rn)):
                    if vals is not None:
                        for ex, r in zip(EXITS, vals):
                            store[(W, C, mode, ex)].add(p, r)
    return dict(S), dict(N), names, nb, len(cr)


def label(k):
    W, C, mode, (h, s, t) = k
    return f"{C}+ launches of a new ticker in {W}s, buy {mode} | hold {h}s, stop -{int(s*100)}%, " + ("no target" if t > 1e6 else f"take +{int(t*100)}%")


def scan():
    start = time.time()
    days = sorted(d for d in glob.glob(os.path.join(DATA, "20*")) if glob.glob(os.path.join(d, "trades__*.parquet")))
    if not days:
        sys.exit("No data. Run: python trend.py download --days 7")
    ts = lambda d: calendar.timegm(time.strptime(os.path.basename(d), "%Y-%m-%d"))
    lo, hi = ts(days[0]), ts(days[-1]) + 86400
    cut1, cut2 = lo + 0.6 * (hi - lo), lo + 0.8 * (hi - lo)
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(scan_day, [(d, cut1, cut2, 21 + i) for i, d in enumerate(days)], chunksize=1)
    S, N, names, nb, launches = {}, {}, collections.Counter(), 0, 0
    for s, x, nm, b, c in res:
        names += nm
        nb += b
        launches += c
        for dst, src in ((S, s), (N, x)):
            for k, v in src.items():
                if k in dst:
                    dst[k].merge(v)
                else:
                    dst[k] = v
    ok = lambda v: v.n.min() >= MIN_N
    pos = lambda v: ok(v) and (v.s / v.n).min() > 0
    good = sorted((k for k, v in S.items() if pos(v)), key=lambda k: -(S[k].s / S[k].n).min())
    null = [k for k, v in N.items() if pos(v)]
    conf = [k for k in good if S[k].without_best() > 0]
    fmt = lambda v: " / ".join(f"{x*100:+.1f}%" for x in v.s / v.n) + "   trades: " + " / ".join(str(int(x)) for x in v.n)
    L = [f"{len(days)} days, {launches:,} launches, {nb:,} name bursts found. Real pool prices, {LAT:.0f}s late, {FEE*200:.1f}% fees plus {2*FIX} SOL per round trip on {SIZE} SOL.",
         f"Rule versions: {len(S):,}. Positive in all 3 periods: {len(good)}. Same-time random coins, same test: {len(null)}.",
         f"Also positive without the best 5 trades: {len(conf)}.", "Most bursting tickers: " + ", ".join(k for k, _ in names.most_common(15))]
    for k in conf[:8]:
        L += ["", label(k), "  per trade: " + fmt(S[k]), "  same-time random coins: " + (fmt(N[k]) if k in N else "n/a")]
    near = sorted((k for k, v in S.items() if ok(v)), key=lambda k: -(S[k].s / S[k].n).min())[:6]
    L += ["", "Closest rules:"]
    for k in near:
        L += [label(k), "  per trade: " + fmt(S[k]), "  same-time random coins: " + (fmt(N[k]) if k in N else "n/a")]
    L += ["", "VERDICT: " + ("candidates above; paper-test the best before risking anything." if conf and len(good) > len(null) else "buying name bursts did not beat costs. Do not trade this with real money."), f"(took {(time.time()-start)/60:.0f} min)"]
    open("trend_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("download").add_argument("--days", type=int, default=7)
    s.add_parser("scan")
    a = p.parse_args()
    download(a.days) if a.cmd == "download" else scan()
