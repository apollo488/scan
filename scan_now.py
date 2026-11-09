#!/usr/bin/env python3
"""scan_now.py - tests fast 1-minute pump.fun rules on the last few weeks of real data.

It reads no wallet and places no trades. It downloads public 1-minute candles for every
pump.fun coin (recorded daily by the open project github.com/wahazzin/memecoin-trader),
then tests second-leg, breakout, volume-burst, dip and buy-pressure rules after costs.

    pip install pyarrow pandas numpy        (needs Python 3.9 or newer; on Windows use: py -m pip install ...)
    python3 scan_now.py download --days 3      roughly 0.3-1.7 GB downloaded per day, only candles are kept
    python3 scan_now.py scan                   prints the result and saves scan_report.txt

A rule only passes if it made money in all three time periods, had at least 50 trades in each,
stayed positive without its best 5 trades, and passes more often than random entries do.
"""
import argparse, glob, json, os, re, sys, tarfile, tempfile, time, urllib.request
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("Please run: pip3 install pyarrow pandas numpy")

REPO, DATA = "wahazzin/memecoin-trader", "data_now"
COST, SLIP, HAIR = 0.05, 0.02, 0.02     # round-trip fees, entry slippage, exit slippage
HOLDS, STOPS, TPS = (2, 5, 15, 30), (0.1, 0.2), (0.3, 1e9)
EXITS = [(h, s, t) for h in HOLDS for s in STOPS for t in TPS]
AGES = [("any", 0, 1e9), ("<5m", 0, 5), ("5-30m", 5, 30), ("30-120m", 30, 120), (">2h", 120, 1e9)]
CAPS = [("any", 0, 1e12), ("<60", 0, 60), ("60-150", 60, 150), ("150-400", 150, 400), (">400", 400, 1e12)]
COLS = ["mint", "t", "o", "h", "l", "c", "buy_sol", "sell_sol", "n_buys", "n_sells", "n_users"]
MIN_BARS, MIN_N = 6, 50


def get(url):
    headers = {"User-Agent": "scan-now"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):      # avoids rate limits when run on GitHub's servers
        headers["Authorization"] = "Bearer " + token
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def download(days):
    rel = []
    for page in range(1, 4):
        batch = json.load(get(f"https://api.github.com/repos/{REPO}/releases?per_page=100&page={page}"))
        rel += batch
        if len(batch) < 100:
            break
    today = "data-" + time.strftime("%Y-%m-%d", time.gmtime())
    rel = sorted((r for r in rel if re.fullmatch(r"data-\d{4}-\d{2}-\d{2}", r["tag_name"]) and r["tag_name"] != today), key=lambda r: r["tag_name"])[-days:]
    print("Days:", rel[0]["tag_name"][5:], "to", rel[-1]["tag_name"][5:], flush=True)
    for r in rel:
        d = os.path.join(DATA, r["tag_name"][5:])
        os.makedirs(d, exist_ok=True)
        assets = [a for a in r["assets"] if a["name"].endswith(".tar")]
        for i, a in enumerate(assets):
            mark = os.path.join(d, a["name"] + ".done")
            if os.path.exists(mark):
                continue
            print(r["tag_name"][5:], f"file {i+1}/{len(assets)}", flush=True)
            for attempt in range(3):
                fd, path = tempfile.mkstemp(dir=d, suffix=".tmp")
                try:
                    with os.fdopen(fd, "wb") as tmp, get(a["browser_download_url"]) as src:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            tmp.write(chunk)
                    take_candles(path, d, a["name"])
                    open(mark, "w").close()
                    break
                except Exception as e:
                    print("  retry:", e, flush=True)
                    time.sleep(5)
                finally:
                    if os.path.exists(path):
                        os.remove(path)


def take_candles(tar_path, out_dir, label):
    """Copy only the candle files out of a downloaded archive; never extract by archive paths."""
    with tarfile.open(tar_path) as tar:
        for m in tar:
            base = os.path.basename(m.name)
            if m.isfile() and re.fullmatch(r"candles_[\w.-]+\.parquet", base):
                with open(os.path.join(out_dir, re.sub(r"\W", "_", label) + "_" + base), "wb") as f:
                    f.write(tar.extractfile(m).read())


def day_dirs():
    days = sorted(d for d in glob.glob(os.path.join(DATA, "20*")) if glob.glob(os.path.join(d, "*.parquet")))
    if not days:
        sys.exit("No data yet. Run: python3 scan_now.py download --days 3")
    return days


FIRST_SEEN = {}      # coin -> first time seen, kept across days for the age filter


def load_day(d):
    """One day at a time, so memory stays low however many days are scanned."""
    import pandas as pd, pyarrow.parquet as pq
    frames = []
    for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
        df = pq.read_table(f, columns=COLS).to_pandas()
        for k in COLS[2:]:
            df[k] = df[k].astype("float32")
        df["t"] = df.t.astype("int32")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["_n"] = df.n_buys + df.n_sells
    df = df.sort_values(["mint", "t", "_n"]).drop_duplicates(["mint", "t"], keep="last")
    return coins_from_frame(df)


def coins_from_frame(df):
    arr = {k: df[k].values.astype(float) for k in ("t", "o", "h", "l", "c", "buy_sol", "sell_sol", "n_users")}
    mint = df.mint.values
    cuts = np.nonzero(mint[1:] != mint[:-1])[0] + 1
    out = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(mint)]):
        t0 = FIRST_SEEN.setdefault(mint[a], float(arr["t"][a]))
        if b - a >= MIN_BARS:      # a safe floor: no rule can fire earlier, so this adds no look-ahead
            c = {k: v[a:b] for k, v in arr.items()}
            c["t0"] = t0
            out.append(c)
    return out


def first(cond):
    w = np.nonzero(cond)[0]
    return int(w[0]) if len(w) else None


def signals(c):
    """Every rule's first firing bar for one coin. Decisions use closed candles only."""
    o, h, l, cl = c["o"], c["h"], c["l"], c["c"]
    n, idx = len(cl), np.arange(len(cl))
    peak = np.maximum.accumulate(h)
    vol = c["buy_sol"] + c["sell_sol"]
    cs = np.r_[0.0, np.cumsum(vol)]
    start = np.maximum(idx - 30, 0)
    prev_mean = (cs[idx] - cs[start]) / np.maximum(idx - start, 1)
    out = {}
    for dd in (0.3, 0.5, 0.7):
        i = first((l <= peak * (1 - dd)) & (idx >= 1))
        if i is None:
            continue
        pk = peak[i]
        for back in (0.1, 0.2, 0.3):
            j = first((idx > i) & (cl >= pk * (1 - back)))
            if j is not None:
                out[f"second-leg fall{int(dd*100)} back-within{int(back*100)}"] = j
        j = first((idx > i) & (cl > o))
        if j is not None:
            out[f"dip fall{int(dd*100)} first-green"] = j
    prev_hi = np.r_[np.inf, peak[:-1]]
    for k in (5, 10, 30):
        for v in (1, 3, 5):
            j = first((cl > prev_hi) & (idx >= k) & (vol >= v * prev_mean))
            if j is not None:
                out[f"breakout after{k}m vol{v}x"] = j
    for x in (3, 5, 10):
        for u in (5, 10, 20):
            j = first((vol >= x * prev_mean) & (cl > o) & (c["n_users"] >= u) & (idx >= 5))
            if j is not None:
                out[f"volume-burst {x}x users{u}"] = j
    with np.errstate(divide="ignore", invalid="ignore"):
        share = c["buy_sol"] / vol
    for th in (0.7, 0.8):
        for u in (10, 20):
            j = first((share >= th) & (c["n_users"] >= u) & (cl > o) & (idx >= 5))
            if j is not None:
                out[f"buy-pressure {int(th*100)}% users{u}"] = j
    return {k: v for k, v in out.items() if 4 <= v and v + 1 < n}


def trade(c, j, hold, stop, tp):
    """Enter at the next candle's open after bar j; exits are decided on candle closes."""
    t, o, cl = c["t"], c["o"], c["c"]
    e = j + 1
    ep = o[e] * (1 + SLIP)
    if ep <= 0:
        return None
    end = t[e] + hold * 60
    n = len(cl)
    for b in range(e, n):
        if t[b] >= end:
            return min(o[b] * (1 - HAIR) / ep - 1 - COST, 10.0)
        x = cl[b] / ep
        if x >= 1 + tp or x <= 1 - stop:
            px = o[b + 1] if b + 1 < n else cl[b] * 0.85
            return min(px * (1 - HAIR) / ep - 1 - COST, 10.0)
    return min(cl[-1] * 0.85 * (1 - HAIR) / ep - 1 - COST, 10.0)


def buckets(c, j):
    age = (c["t"][j] - c["t0"]) / 60.0
    cap = c["c"][j] * 1e9
    a = [x[0] for x in AGES if x[1] <= age < x[2]]
    b = [x[0] for x in CAPS if x[1] <= cap < x[2]]
    return a, b


def run(coins, cut1, cut2, null_seed=None):
    rng = np.random.default_rng(null_seed) if null_seed else None
    S = defaultdict(lambda: np.zeros((3, 3)))
    for c in coins:
        for name, j in signals(c).items():
            if rng is not None:
                j = int(rng.integers(4, len(c["t"]) - 1))
            p = 0 if c["t"][j] < cut1 else (1 if c["t"][j] < cut2 else 2)
            ages, caps = buckets(c, j)
            for ex in EXITS:
                r = trade(c, j, *ex)
                if r is None:
                    continue
                for a in ages:
                    for b in caps:
                        v = S[(name, ex, a, b)]
                        v[p, 0] += 1
                        v[p, 1] += r
                        v[p, 2] += r > 0
    return S


def passed(S):
    out = []
    for k, v in S.items():
        if v[:, 0].min() >= MIN_N:
            m = v[:, 1] / v[:, 0]
            if m.min() > 0:
                out.append((float(m.min()), k, m, v[:, 0]))
    return sorted(out, key=lambda x: -x[0])


def details(days, keys):
    A, B = dict((x[0], x) for x in AGES), dict((x[0], x) for x in CAPS)
    out = {k: [] for k in keys}
    for d in days:
        for c in load_day(d):
            sig = signals(c)
            for k in keys:
                name, ex, age, cap = k
                j = sig.get(name)
                if j is None:
                    continue
                if A[age][1] <= (c["t"][j] - c["t0"]) / 60.0 < A[age][2] and B[cap][1] <= c["c"][j] * 1e9 < B[cap][2]:
                    r = trade(c, j, *ex)
                    if r is not None:
                        out[k].append((c["t"][j], r))
    return {k: sorted(v) for k, v in out.items()}


def account(rows, stake, hold_min, start=200.0):
    """Rough account path: at most 1/stake trades open at once; extra signals are skipped."""
    eq, peak, worst, ends = start, start, 0.0, []
    for t, r in rows:
        ends = [e for e in ends if e > t]
        if len(ends) >= int(1 / stake):
            continue
        ends.append(t + hold_min * 60)
        eq = max(eq + eq * stake * r, 0.0)
        peak = max(peak, eq)
        worst = min(worst, eq / peak - 1)
    return eq, worst


def label(k):
    n, (h, s, t), a, b = k
    tp = "no target" if t > 1e6 else f"take +{int(t*100)}%"
    return f"{n} | hold {h}m, stop -{int(s*100)}%, {tp} | age {a}, market cap {b} SOL"


def lower_priority():
    """Let everything else on the machine go first (works on Windows, Mac and Linux)."""
    try:
        if hasattr(os, "nice"):
            os.nice(10)
        else:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def peak_memory_gb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1 << 30 if sys.platform == "darwin" else 1 << 20)
    except Exception:
        return None


def merge(total, part):
    for k, v in part.items():
        total[k] = total[k] + v if k in total else v.copy()


def scan():
    import calendar
    lower_priority()
    t_start = time.time()
    days = day_dirs()
    day_ts = lambda d: calendar.timegm(time.strptime(os.path.basename(d), "%Y-%m-%d"))
    lo, hi = day_ts(days[0]), day_ts(days[-1]) + 86400
    cut1, cut2 = lo + 0.6 * (hi - lo), lo + 0.8 * (hi - lo)
    fmt = lambda x: time.strftime("%d %b %H:%M", time.gmtime(x))
    lines = [f"Data {fmt(lo)} to {fmt(hi)} UTC, {len(days)} days. Periods split at {fmt(cut1)} and {fmt(cut2)}.",
             f"Costs: {COST*100:.0f}% round trip plus {SLIP*100:.0f}% slippage each way."]
    print(lines[0], flush=True)
    S, N, n_coins = {}, {}, 0
    for i, d in enumerate(days):
        coins = load_day(d)
        n_coins += len(coins)
        print(f"{os.path.basename(d)}: {len(coins):,} coins", flush=True)
        merge(S, run(coins, cut1, cut2))
        merge(N, run(coins, cut1, cut2, null_seed=7 + i))
        del coins
    good, null = passed(S), passed(N)
    lines.append(f"Coins tested: {n_coins:,}. Rule versions: {len(S):,}. Positive in all 3 periods: {len(good)}. Random entries, same test: {len(null)}.")
    confirmed = []
    rows_by_key = details(days, [x[1] for x in good[:200]]) if good else {}
    for m, k, means, ns in good[:200]:
        rows = rows_by_key[k]
        vals = sorted(r for _, r in rows)
        if len(vals) > 5 and sum(vals[:-5]) > 0:
            confirmed.append((m, k, means, ns, rows))
    lines.append(f"Also positive without the best 5 trades: {len(confirmed)}.")
    for m, k, means, ns, rows in confirmed[:12]:
        test = [x for x in rows if x[0] >= cut2]
        lines.append("")
        lines.append(label(k))
        lines.append("  per trade: " + " / ".join(f"{x*100:+.1f}%" for x in means) + "   trades: " + " / ".join(str(int(x)) for x in ns))
        lines.append("  last period on GBP200: " + ", ".join(f"{int(f*100)}% stake -> GBP{account(test, f, k[1][0])[0]:.0f} (worst drop {account(test, f, k[1][0])[1]*100:.0f}%)" for f in (0.05, 0.1, 0.25)))
    if len(good) <= len(null) or not confirmed:
        lines.append("")
        lines.append("VERDICT: no rule beat what random entries achieve. Do not trade these with real money.")
    else:
        lines.append("")
        lines.append("VERDICT: candidates above. Paper-test the best one before risking anything.")
    mem = peak_memory_gb()
    lines.append(f"(took {(time.time()-t_start)/60:.0f} min" + (f", peak memory {mem:.1f} GB)" if mem else ")"))
    open("scan_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--days", type=int, default=3)
    sub.add_parser("scan")
    a = p.parse_args()
    download(a.days) if a.cmd == "download" else scan()
