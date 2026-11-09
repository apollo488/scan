#!/usr/bin/env python3
"""Recent-data edge scan across every market Yahoo Finance gives us.

Uses the last two years of hourly bars, the last 59 days of 5-minute bars and
the last five years of daily bars. Tests fast rules both ways round (follow and
inverted), time-of-day effects, news days (CPI, jobs report, FOMC), and picking
the strongest market each day. All results are after spread costs, split into
separate time periods and compared with randomised prices.

Writes everything to recent_report.txt.
"""
import time
import traceback

import numpy as np
import pandas as pd

SEED = 77
HOUR_NULLS = 20
FIVE_NULLS = 10
EXAM_NULLS = 200

# name, Yahoo ticker, round-trip cost as a fraction, UK maximum leverage
MARKETS = [
    ("Nasdaq", "NQ=F", 0.00015, 20), ("S&P 500", "ES=F", 0.00015, 20), ("Dow", "YM=F", 0.00015, 20),
    ("Russell", "RTY=F", 0.0003, 20), ("DAX", "^GDAXI", 0.00015, 20), ("FTSE", "^FTSE", 0.0002, 20),
    ("Gold", "GC=F", 0.00015, 20), ("Silver", "SI=F", 0.0008, 10), ("Copper", "HG=F", 0.001, 10),
    ("Platinum", "PL=F", 0.0015, 10), ("US oil", "CL=F", 0.0008, 10), ("Brent", "BZ=F", 0.0008, 10),
    ("Nat gas", "NG=F", 0.0025, 10), ("Soybeans", "ZS=F", 0.0015, 10), ("Corn", "ZC=F", 0.0015, 10),
    ("Wheat", "ZW=F", 0.0015, 10), ("Coffee", "KC=F", 0.002, 10), ("Sugar", "SB=F", 0.002, 10),
    ("Cocoa", "CC=F", 0.002, 10),
    ("EURUSD", "EURUSD=X", 0.00012, 30), ("GBPUSD", "GBPUSD=X", 0.00012, 30), ("USDJPY", "USDJPY=X", 0.00012, 30),
    ("AUDUSD", "AUDUSD=X", 0.00015, 20), ("USDCAD", "USDCAD=X", 0.00015, 30), ("USDCHF", "USDCHF=X", 0.00015, 30),
    ("NZDUSD", "NZDUSD=X", 0.0002, 20), ("EURJPY", "EURJPY=X", 0.0002, 20), ("GBPJPY", "GBPJPY=X", 0.0002, 20),
    ("Bitcoin", "BTC-USD", 0.006, 1), ("Ether", "ETH-USD", 0.006, 1), ("Solana", "SOL-USD", 0.006, 1),
    ("Nvidia", "NVDA", 0.0008, 5), ("Tesla", "TSLA", 0.0008, 5), ("AMD", "AMD", 0.0008, 5),
    ("Meta", "META", 0.0008, 5), ("Apple", "AAPL", 0.0008, 5), ("Microsoft", "MSFT", 0.0008, 5),
    ("Amazon", "AMZN", 0.0008, 5), ("Google", "GOOGL", 0.0008, 5), ("MicroStrategy", "MSTR", 0.0012, 5),
    ("Coinbase", "COIN", 0.0012, 5), ("Palantir", "PLTR", 0.0012, 5),
]

CPI = ("2024-11-13 2024-12-11 2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 "
       "2025-08-12 2025-09-11 2025-10-24 2025-12-18 2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 "
       "2026-06-10 2026-07-14 2026-08-12 2026-09-11 2026-10-14").split()
NFP = ("2024-12-06 2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 "
       "2025-09-05 2025-11-20 2025-12-16 2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 "
       "2026-07-02 2026-08-07 2026-09-04 2026-10-02 2026-11-06").split()
FOMC = ("2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 "
        "2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28").split()
EVENTS = (("CPI", CPI, 480), ("Jobs", NFP, 480), ("FOMC", FOMC, 840))
EVENT_MARKETS = ("Nasdaq", "S&P 500", "Dow", "Russell", "Gold", "Silver", "US oil", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD")

# The best 5-minute rules from Nov 2021 to Sept 2023, re-tested on the last 59 days.
OLD_RULES = (
    ("USDJPY", "London", "brk96", "up", 24, "long"),
    ("Gold", "NY open", "brk24", "up", 6, "long"),
    ("Gold", "London", "brk96", "up", 24, "short"),
    ("Nasdaq", "NY day", "brk12", "up", 24, "long"),
    ("Nasdaq", "NY close", "mom12z1.5", "up", 6, "long"),
)

WINDOWS = (("NY day", 570, 960, 1.0), ("NY open", 570, 630, 1.5), ("NY close", 900, 960, 1.0),
           ("London", 180, 480, 1.25), ("Asia", -300, 180, 1.75))

lines = []


def say(text=""):
    lines.append(str(text))
    print(text, flush=True)


# ------------------------------------------------------------------ data


def fetch(ticker, period, interval):
    import yfinance as yf

    last = "nothing returned"
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                             prepost=False, progress=False, threads=False)
            if df is not None and len(df) > 0:
                return clean(df)
            last = "empty"
        except Exception as exc:  # the download library raises many different errors
            last = repr(exc)[:120]
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(last)


def clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).lower())[["open", "high", "low", "close"]].dropna()
    df = df[(df > 0).all(axis=1)]
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert("UTC")
    return df[~df.index.duplicated()].sort_index()


def arrays(df, name, cost, cap):
    utc = df.index.tz_convert("UTC")
    ny = df.index.tz_convert("America/New_York")
    mins = np.asarray(ny.hour * 60 + ny.minute, int)
    # The 5pm New York rollover: spreads blow out and bid-only prices fake a dip and bounce, so no trades there.
    bad = np.zeros(len(df), bool) if cap in (1, 5) else ((mins >= 990) & (mins < 1110))
    return dict(name=name, cost=cost, cap=cap, bad=bad,
                o=df["open"].to_numpy(float), h=df["high"].to_numpy(float),
                l=df["low"].to_numpy(float), c=df["close"].to_numpy(float),
                sec=np.asarray((utc - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1), float),
                mins=mins,
                day=np.asarray(ny.strftime("%Y-%m-%d")))


def mirrored(d, rng):
    """Every bar randomly mirrored about the previous close: no direction, no pattern."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    pc = c[:-1]
    ro, rh, rl, rc = o[1:] / pc, h[1:] / pc, l[1:] / pc, c[1:] / pc
    f = rng.random(len(pc)) < 0.5
    ro2, rh2, rl2, rc2 = np.where(f, 2 - ro, ro), np.where(f, 2 - rl, rh), np.where(f, 2 - rh, rl), np.where(f, 2 - rc, rc)
    c2 = c[0] * np.cumprod(np.r_[1.0, rc2])
    prev = c2[:-1]
    e = dict(d)
    e.update(o=np.r_[o[0], prev * ro2], h=np.r_[h[0], prev * rh2], l=np.r_[l[0], prev * rl2], c=c2)
    return e


# ------------------------------------------------------------------ rules


def signal_masks(d, ks, zs, ns, hours, vol_window):
    """Every entry signal. A signal on bar i means enter at the open of bar i+1."""
    h, l, c, mins = d["h"], d["l"], d["c"], d["mins"]
    n = len(c)
    logr = np.r_[0.0, np.log(c[1:] / c[:-1])]
    sd = pd.Series(logr).rolling(vol_window, min_periods=max(20, vol_window // 4)).std().to_numpy()
    out = {}
    for k in ks:
        r = np.full(n, np.nan)
        r[k:] = c[k:] / c[:-k] - 1
        s = sd * np.sqrt(k)
        for z in zs:
            th = z * s if z > 0 else 0.0
            out["mom%dz%g|up" % (k, z)] = r > th
            out["mom%dz%g|dn" % (k, z)] = r < -th
    for nn in ns:
        top = pd.Series(h).rolling(nn).max().shift(1).to_numpy()
        bot = pd.Series(l).rolling(nn).min().shift(1).to_numpy()
        out["brk%d|up" % nn] = c > top
        out["brk%d|dn" % nn] = c < bot
    if hours:
        nxt = np.r_[mins[1:], -60]
        for hh in range(24):
            out["hour%02d|any" % hh] = (nxt // 60) == hh
    return out


def forwards(d, holds, bar, slack):
    o, sec = d["o"], d["sec"]
    n = len(o)
    res = {}
    for hh in holds:
        f = np.full(n, np.nan)
        a = np.arange(0, max(n - 1 - hh, 0))
        en, ex = a + 1, a + 1 + hh
        ok = ((sec[ex] - sec[en]) <= hh * bar + slack) & ~d["bad"][en] & ~d["bad"][ex]
        f[a[ok]] = o[ex[ok]] / o[en[ok]] - 1
        res[hh] = f
    return res


def window_mask(mins, a, b):
    return ((mins >= 1440 + a) | (mins < b)) if a < 0 else ((mins >= a) & (mins < b))


def scan(d, holds, ks, zs, ns, hours, windows, cuts, nmin, bar, slack, vol_window):
    n = len(d["c"])
    position = np.arange(n) / max(n, 1)
    per = np.searchsorted(np.array(cuts), position, side="right")
    nper = len(cuts) + 1
    masks = signal_masks(d, ks, zs, ns, hours, vol_window)
    fw = forwards(d, holds, bar, slack)
    rows = []
    total = 0
    win_list = windows if windows else (("all", None, None, 1.0),)
    for wn, a, b, mult in win_list:
        wm = np.ones(n, bool) if a is None else window_mask(d["mins"], a, b)
        cost = d["cost"] * mult
        for label, mask in masks.items():
            idx = np.flatnonzero(mask & wm)
            if len(idx) < nmin * nper:
                continue
            for hold in holds:
                x = fw[hold][idx]
                ok = ~np.isnan(x)
                x = x[ok]
                p = per[idx][ok]
                cnt = np.bincount(p, minlength=nper)
                if cnt.min() < nmin:
                    continue
                g = np.bincount(p, x, minlength=nper) / cnt
                for side, sg in (("long", 1), ("short", -1)):
                    total += 1
                    net = sg * g - cost
                    if (net > 0).all():
                        m = sg * x.mean() - cost
                        sd = x.std()
                        rows.append(dict(name=d["name"], window=wn, signal=label, hold=hold, side=side, n=len(x),
                                         net=m, t=m / (sd / np.sqrt(len(x))) if sd > 0 else 0.0,
                                         periods=[float(v) for v in net]))
    return rows, total


def trades(d, rule, windows, bar, slack, vol_window, ks, zs, ns, hours):
    """Non-overlapping trades for one rule, in time order."""
    o, h, l, sec = d["o"], d["h"], d["l"], d["sec"]
    mask = signal_masks(d, ks, zs, ns, hours, vol_window)[rule["signal"]]
    cost = d["cost"]
    if rule["window"] != "all":
        wn = [w for w in windows if w[0] == rule["window"]][0]
        mask = mask & window_mask(d["mins"], wn[1], wn[2])
        cost *= wn[3]
    sg = 1 if rule["side"] == "long" else -1
    hold = rule["hold"]
    n = len(o)
    out = []
    i = 0
    for i0 in np.flatnonzero(mask):
        if i0 < i or i0 + 1 + hold >= n:
            continue
        en, ex = i0 + 1, i0 + 1 + hold
        if sec[ex] - sec[en] > hold * bar + slack or d["bad"][en] or d["bad"][ex]:
            continue
        entry = o[en]
        worst = float(np.min(l[en:ex]) / entry - 1) if sg == 1 else float(1 - np.max(h[en:ex]) / entry)
        out.append((float(sec[ex]), sg * (o[ex] / entry - 1) - cost, worst))
        i = ex
    return out


def grow(tr, leverage, since):
    """Pounds after compounding every trade at the given leverage, from 200."""
    money = 200.0
    for when, r, worst in tr:
        if when < since:
            continue
        if leverage * worst <= -0.95:
            return 0.0
        money *= max(1 + leverage * r, 0.0)
        if money <= 0:
            return 0.0
    return money


# ------------------------------------------------------------------ hourly scan


def hourly_section(data, rng):
    ks, zs, ns, holds = (1, 2, 4, 8, 24), (0, 1, 2, 3), (8, 24, 72, 120), (1, 2, 3, 4, 6, 8, 12, 24)
    args = dict(holds=holds, ks=ks, zs=zs, ns=ns, hours=True, windows=None, cuts=(0.6, 0.8), nmin=20, bar=3600, slack=5400)
    say("")
    say("=== 1. HOURLY RULES, LAST 2 YEARS (every market, follow and inverted) ===")
    say("Pass = profit after costs in all 3 time periods. 'Random' = same test on randomised prices.")
    say("Market | rules | pass real | pass random | beat the best random")
    survivors = []
    total_real, total_null = 0, 0.0
    for d in data:
        per_day = max(1, int(round(len(d["c"]) / max(len(set(d["day"])), 1))))
        vol = per_day * 20
        rows, total = scan(d, vol_window=vol, **args)
        tops, counts = [], []
        for _ in range(HOUR_NULLS):
            r2, _t = scan(mirrored(d, rng), vol_window=vol, **args)
            counts.append(len(r2))
            tops.append(max([x["t"] for x in r2], default=0.0))
        bar_t = float(np.quantile(tops, 0.95))
        good = [r for r in rows if r["t"] > bar_t]
        for r in good:
            r["vol"], r["bar_t"] = vol, bar_t
        survivors += good
        total_real += len(rows)
        total_null += float(np.mean(counts))
        say("%s | %d | %d | %.0f | %d" % (d["name"], total, len(rows), np.mean(counts), len(good)))
    say("ALL MARKETS: pass real %d, pass random %.0f" % (total_real, total_null))
    if not survivors:
        say("Nothing beat the best random result.")
        return
    say("")
    say("Rules that beat the best random result (best version of each idea):")
    best = {}
    for r in sorted(survivors, key=lambda x: -x["t"]):
        best.setdefault((r["name"], r["side"], r["signal"].split("|")[0][:3]), r)
    by_name = {d["name"]: d for d in data}
    table = []
    for r in best.values():
        d = by_name[r["name"]]
        tr = trades(d, r, (), 3600, 5400, r["vol"], ks, zs, ns, True)
        if len(tr) < 20:
            continue
        ret = np.array([x[1] for x in tr])
        now = d["sec"][-1]
        table.append((grow(tr, d["cap"], now - 60 * 86400), grow(tr, d["cap"] / 2, now - 150 * 86400), grow(tr, d["cap"], now - 150 * 86400),
                      r, len(ret), (ret > 0).mean(), ret.mean()))
    table.sort(key=lambda x: -x[2])
    for m60, half150, full150, r, nt, wr, avg in table[:25]:
        say("%s %s %s hold %dh | %d trades, win %.0f%%, avg %+.3f%%, score %.1f | £200 last 60d at max lev: £%.0f | last 150d half lev: £%.0f, max lev: £%.0f" % (
            r["name"], r["side"], r["signal"], r["hold"], nt, wr * 100, avg * 100, r["t"], m60, half150, full150))


# ------------------------------------------------------------------ five-minute scan


def five_section(data, rng):
    ks, zs, ns, holds = (1, 2, 3, 6, 12), (0, 1, 1.5, 2, 3), (12, 24, 48, 96), (1, 2, 3, 6, 12, 24)
    args = dict(holds=holds, ks=ks, zs=zs, ns=ns, hours=False, windows=WINDOWS, cuts=(0.5,), nmin=12, bar=300, slack=120, vol_window=288)
    say("")
    say("=== 3. 5-MINUTE RULES, LAST 59 DAYS (high frequency) ===")
    say("Pass = profit after costs in both halves. Only a short period, so treat as a live check.")
    say("Market | rules | pass real | pass random | beat the best random")
    survivors = []
    for d in data:
        rows, total = scan(d, **args)
        tops, counts = [], []
        for _ in range(FIVE_NULLS):
            r2, _t = scan(mirrored(d, rng), **args)
            counts.append(len(r2))
            tops.append(max([x["t"] for x in r2], default=0.0))
        bar_t = float(np.quantile(tops, 0.9))
        good = [r for r in rows if r["t"] > bar_t]
        survivors += good
        say("%s | %d | %d | %.0f | %d" % (d["name"], total, len(rows), np.mean(counts), len(good)))
    survivors.sort(key=lambda x: -x["t"])
    by_name = {d["name"]: d for d in data}
    if survivors:
        say("")
        say("Best 5-minute rules right now:")
        for r in survivors[:12]:
            d = by_name[r["name"]]
            tr = trades(d, r, WINDOWS, 300, 120, 288, ks, zs, ns, False)
            ret = np.array([x[1] for x in tr]) if tr else np.array([0.0])
            say("%s %s %s %s hold %d bars | %d trades, win %.0f%%, avg %+.3f%%, score %.1f, £200 at max lev: £%.0f" % (
                r["name"], r["window"], r["side"], r["signal"], r["hold"], len(tr), (ret > 0).mean() * 100, ret.mean() * 100,
                r["t"], grow(tr, d["cap"], 0.0)))
    say("")
    say("Did the best 5-minute rules of 2021-23 still work in the last 59 days?")
    for name, wn, sig, when, hold, side in OLD_RULES:
        if name not in by_name:
            continue
        d = by_name[name]
        rule = dict(window=wn, signal=sig + "|" + when, hold=hold, side=side)
        tr = trades(d, rule, WINDOWS, 300, 120, 288, ks, zs, ns, False)
        if tr:
            ret = np.array([x[1] for x in tr])
            say("  %s %s %s %s hold %d: %d trades, win %.0f%%, avg %+.3f%%" % (name, wn, side, rule["signal"], hold, len(tr), (ret > 0).mean() * 100, ret.mean() * 100))


# ------------------------------------------------------------------ news days


def events_section(data):
    say("")
    say("=== 2. NEWS DAYS, LAST 2 YEARS (hourly bars) ===")
    say("Follow or fade the first hour after CPI, jobs report and FOMC. Costs tripled for news spreads.")
    results = []
    for d in data:
        if d["name"] not in EVENT_MARKETS:
            continue
        key = {(d["day"][i], int(d["mins"][i])): i for i in range(len(d["c"]))}
        o, c = d["o"], d["c"]
        for ev, dates, t0 in EVENTS:
            for hold in (1, 2, 3, 6, 8):
                xs = []
                for day in dates:
                    i = key.get((day, t0))
                    if i is None or i + 1 + hold >= len(c):
                        continue
                    move = c[i] / o[i] - 1
                    if move == 0:
                        continue
                    xs.append(np.sign(move) * (o[i + 1 + hold] / o[i + 1] - 1))
                if len(xs) < 8:
                    continue
                x = np.array(xs)
                cost = d["cost"] * 3
                half = len(x) // 2
                for mode, sg in (("follow", 1), ("fade", -1)):
                    net = sg * x - cost
                    if net[:half].mean() > 0 and net[half:].mean() > 0:
                        t = net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else 0.0
                        results.append((t, d["name"], ev, mode, hold, len(x), net.mean(), (net > 0).mean(), d["cap"]))
    results.sort(key=lambda r: -r[0])
    say("Tested combinations that made money in both halves: %d" % len(results))
    say("Market | event | action | hold hours | times | avg net | win | score | per event at max lev")
    for t, name, ev, mode, hold, n, avg, wr, cap in results[:15]:
        say("%s | %s | %s | %d | %d | %+.2f%% | %.0f%% | %.1f | %+.0f%%" % (name, ev, mode, hold, n, avg * 100, wr * 100, t, avg * cap * 100))


# ------------------------------------------------------------------ strongest market


def daily_section(frames):
    say("")
    say("=== 4. STRONGEST MARKET, LAST 5 YEARS (daily bars) ===")
    closes = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    closes.index = pd.DatetimeIndex(closes.index).tz_convert("UTC").normalize()
    closes = closes.groupby(closes.index).last().sort_index().ffill(limit=3)
    rets = closes.pct_change(fill_method=None)
    costs = {m[0]: m[2] for m in MARKETS}
    caps = {m[0]: m[3] for m in MARKETS}
    last = closes.index[-1]
    say("Biggest moves (price change):")
    rows = []
    for col in closes:
        s = closes[col].dropna()
        for span, label in ((365, "1y"), (730, "2y")):
            past = s[s.index <= last - pd.Timedelta(days=span)]
            if len(past):
                rows.append((label, col, s.iloc[-1] / past.iloc[-1] - 1))
    for label in ("1y", "2y"):
        top = sorted([r for r in rows if r[0] == label], key=lambda r: -abs(r[2]))[:6]
        say("  last %s: " % label + ", ".join("%s %+.0f%%" % (r[1], r[2] * 100) for r in top))
    say("Each day hold only the market moving hardest (either direction), after costs and 3% a year financing:")
    say("Lookback days | 5y yearly | Sharpe | worst fall | £200 over last 12m at 3x / 5x / 10x | last 24m at 3x / 5x")
    vol = rets.rolling(60, min_periods=30).std()
    for look in (5, 10, 20, 60):
        score = (closes / closes.shift(look) - 1) / (vol * np.sqrt(look))
        strength = score.abs()
        usable = strength.notna().any(axis=1)
        pick = pd.Series([None] * len(strength), index=strength.index, dtype=object)
        pick[usable] = strength[usable].idxmax(axis=1)
        chosen, sign = [], []
        for day, name in pick.items():
            if isinstance(name, str) and not np.isnan(score.loc[day, name]):
                chosen.append(name)
                sign.append(np.sign(score.loc[day, name]))
            else:
                chosen.append(None)
                sign.append(0.0)
        out = np.zeros(len(closes))
        for t in range(1, len(closes)):
            nm, sg = chosen[t - 1], sign[t - 1]
            if nm is None:
                continue
            r = rets.iloc[t][nm]
            if np.isnan(r):
                continue
            changed = (t < 2 or chosen[t - 2] != nm or sign[t - 2] != sg)
            out[t] = sg * r - (costs[nm] if changed else 0.0) - 0.03 / 252
        series = pd.Series(out, index=closes.index)
        ann = series.mean() * 252
        sharpe = series.mean() / series.std() * 252 ** 0.5 if series.std() > 0 else 0.0
        eq = (1 + series).cumprod()
        dd = float((eq / eq.cummax() - 1).min())

        def account(span, lev):
            part = series[series.index > last - pd.Timedelta(days=span)]
            m = 200.0
            for r in part:
                m *= max(1 + lev * r, 0.0)
            return m

        say("%d | %+.0f%% | %.2f | %.0f%% | £%.0f / £%.0f / £%.0f | £%.0f / £%.0f" % (
            look, ann * 100, sharpe, dd * 100, account(365, 3), account(365, 5), account(365, 10), account(730, 3), account(730, 5)))
    say("(UK retail leverage caps apply to each market. Crypto cannot be leveraged.)")
    _ = caps


# ------------------------------------------------------------------ Nasdaq exam


def to_matrix(df):
    ny = df.index.tz_convert("America/New_York")
    minutes = np.asarray(ny.hour * 60 + ny.minute, int)
    slot = (minutes - 570) // 60
    table = df.assign(slot=slot, day=ny.tz_localize(None).normalize())
    table = table[(minutes >= 570) & (minutes % 60 == 30) & (table["slot"] < 7)]
    days, o, h, l, c = [], [], [], [], []
    for day, g in table.groupby("day"):
        if len(g) != 7 or sorted(g["slot"]) != list(range(7)):
            continue
        g = g.sort_values("slot")
        days.append(day)
        o.append(g["open"].to_numpy(float))
        h.append(g["high"].to_numpy(float))
        l.append(g["low"].to_numpy(float))
        c.append(g["close"].to_numpy(float))
    return pd.DatetimeIndex(days), np.array(o), np.array(h), np.array(l), np.array(c)


def nasdaq_rule(o, h, l, c, cost):
    days, bars = o.shape
    moves = np.abs(c / o[:, [0]] - 1)
    running = np.cumsum(np.vstack([np.zeros(bars), moves]), axis=0)
    band = np.full((days, bars), np.nan)
    band[14:] = (running[14:days] - running[0:days - 14]) / 14
    prev = np.r_[np.nan, c[:-1, -1]]
    avg = np.cumsum((h + l + c) / 3, axis=1) / np.arange(1, bars + 1)
    daily = np.zeros(days)
    count = 0
    for d in range(15, days):
        upper = max(o[d, 0], prev[d]) * (1 + band[d])
        pos, entry, pnl = 0, 0.0, 0.0
        for k in range(bars - 1):
            px = c[d, k]
            if pos == 1 and px < max(upper[k], avg[d, k]):
                pnl += o[d, k + 1] / entry - 1 - cost
                pos = 0
                count += 1
            elif pos == 0 and px > upper[k]:
                pos, entry = 1, o[d, k + 1]
        if pos == 1:
            pnl += c[d, bars - 1] / entry - 1 - cost
            count += 1
        daily[d] = pnl
    return daily, count


def nasdaq_flip_version(o, h, l, c, cost, stop):
    """First trade of the day only, with a stop. Returns each day's result and its worst move."""
    days, bars = o.shape
    moves = np.abs(c / o[:, [0]] - 1)
    running = np.cumsum(np.vstack([np.zeros(bars), moves]), axis=0)
    band = np.full((days, bars), np.nan)
    band[14:] = (running[14:days] - running[0:days - 14]) / 14
    prev = np.r_[np.nan, c[:-1, -1]]
    avg = np.cumsum((h + l + c) / 3, axis=1) / np.arange(1, bars + 1)
    daily, worst = np.zeros(days), np.zeros(days)
    count = 0
    for d in range(15, days):
        upper = max(o[d, 0], prev[d]) * (1 + band[d])
        pos, entry, first_bar, done = 0, 0.0, 0, False
        for k in range(bars):
            if pos == 1 and l[d, k] <= entry * (1 - stop):
                fill = min(entry * (1 - stop), o[d, k]) if k > first_bar else entry * (1 - stop)
                daily[d] = fill / entry - 1 - cost
                worst[d] = fill / entry - 1
                pos, done = 0, True
                count += 1
            if k < bars - 1:
                px = c[d, k]
                if pos == 1 and px < max(upper[k], avg[d, k]):
                    daily[d] = o[d, k + 1] / entry - 1 - cost
                    pos, done = 0, True
                    count += 1
                elif pos == 0 and not done and px > upper[k]:
                    pos, entry, first_bar = 1, o[d, k + 1], k + 1
        if pos == 1:
            daily[d] = c[d, bars - 1] / entry - 1 - cost
            count += 1
    return daily, worst, count


def rolling_flips(daily, worst, lev, span):
    """From every possible start day: did £200 reach each target before falling under £20?"""
    n = len(daily)
    hits = {400: 0, 1000: 0, 2000: 0}
    wiped = 0
    starts = max(n - span, 0)
    for s in range(starts):
        money = 200.0
        got = set()
        dead = False
        for t in range(s, s + span):
            if lev * worst[t] <= -1:
                dead = True
                break
            money *= max(1 + lev * daily[t], 0.0)
            for target in hits:
                if target not in got and money >= target:
                    got.add(target)
            if money < 20:
                dead = True
                break
        for target in got:
            hits[target] += 1
        wiped += 1 if dead else 0
    return {k: v / max(starts, 1) for k, v in hits.items()}, wiped / max(starts, 1)


def prop_version(o, h, l, c, cost, stop, lookback, max_trades):
    """Buy side only, up to max_trades a day, with a stop, using a longer lookback."""
    days, bars = o.shape
    moves = np.abs(c / o[:, [0]] - 1)
    running = np.cumsum(np.vstack([np.zeros(bars), moves]), axis=0)
    band = np.full((days, bars), np.nan)
    band[lookback:] = (running[lookback:days] - running[0:days - lookback]) / lookback
    prev = np.r_[np.nan, c[:-1, -1]]
    avg = np.cumsum((h + l + c) / 3, axis=1) / np.arange(1, bars + 1)
    daily, worst = np.zeros(days), np.zeros(days)
    count = 0
    for d in range(lookback + 1, days):
        upper = max(o[d, 0], prev[d]) * (1 + band[d])
        pos, entry, first_bar, taken, cum, low = 0, 0.0, 0, 0, 0.0, 0.0
        for k in range(bars):
            if pos == 1 and k >= first_bar:
                move = l[d, k] / entry - 1
                if move <= -stop:
                    fill = min(-stop, o[d, k] / entry - 1) if k > first_bar else -stop
                    cum += fill - cost
                    low = min(low, cum)
                    pos = 0
                    taken += 1
                else:
                    low = min(low, cum + move)
            if k < bars - 1:
                px = c[d, k]
                if pos == 1 and px < max(upper[k], avg[d, k]):
                    cum += o[d, k + 1] / entry - 1 - cost
                    low = min(low, cum)
                    pos = 0
                    taken += 1
                elif pos == 0 and taken < max_trades and px > upper[k]:
                    pos, entry, first_bar = 1, o[d, k + 1], k + 1
        if pos == 1:
            cum += c[d, bars - 1] / entry - 1 - cost
            low = min(low, cum)
            taken += 1
        daily[d] = cum
        worst[d] = low
        count += taken
    return pd.Series(daily, index=pd.RangeIndex(days)), pd.Series(worst, index=pd.RangeIndex(days)), count


def prop_section():
    say("")
    say("=== 6. PROP FIRM VERSION, LAST 2 YEARS (lookback 90 days, 0.35% stop, up to 2 trades a day) ===")
    series = {}
    for name, ticker in (("Nasdaq", "QQQ"), ("S&P", "SPY")):
        days, o, h, l, c = to_matrix(fetch(ticker, "729d", "1h"))
        d1, w1, n1 = prop_version(o, h, l, c, 0.0002, 0.0035, 90, 2)
        d1.index = days
        w1.index = days
        series[name] = (d1.iloc[91:], w1.iloc[91:], n1)
    both = pd.concat([series["Nasdaq"][0], series["S&P"][0] * 0.5], axis=1, sort=True).fillna(0).sum(axis=1)
    both_low = pd.concat([series["Nasdaq"][1], series["S&P"][1] * 0.5], axis=1, sort=True).fillna(0).sum(axis=1)
    for label, ret, low, size in (("Nasdaq only at 4.4x", series["Nasdaq"][0], series["Nasdaq"][1], 4.4),
                                  ("Nasdaq plus half S&P at 3.1x", both, both_low, 3.1)):
        x = ret * size
        eq = x.cumsum()
        peak = eq.cummax()
        fall = float((eq - peak).min())
        inner = float((eq.shift(1, fill_value=0.0) + low * size - peak.shift(1, fill_value=0.0)).min())
        m = x.groupby(x.index.to_period("M")).sum()
        say("%s: from %s to %s" % (label, x.index[0].date(), x.index[-1].date()))
        say("  Average month %+.2f%%, worst %+.2f%%, best %+.2f%%, months up %d of %d" % (
            m.mean() * 100, m.min() * 100, m.max() * 100, (m > 0).sum(), len(m)))
        say("  Biggest fall %.1f%% (%.1f%% inside trades), lowest vs start %.1f%%, total %+.1f%%" % (
            fall * 100, min(inner, fall) * 100, eq.min() * 100, eq.iloc[-1] * 100))
        say("  By year: " + " ".join("%d:%+.1f%%" % (y, g.sum() * 100) for y, g in x.groupby(x.index.year)))
        say("  Months: " + " ".join("%s:%+.1f" % (str(k)[2:], v * 100) for k, v in m.items()))


def exam_section(hourly):
    say("")
    say("=== 5. THE NASDAQ RULE ON THE LAST 2 YEARS (never seen before) ===")
    days, o, h, l, c = to_matrix(hourly)
    if len(days) < 200:
        say("Not enough days: %d" % len(days))
        return
    for cost in (0.0001, 0.00015, 0.0002, 0.0003):
        daily, n = nasdaq_rule(o, h, l, c, cost)
        x = daily[15:]
        say("Cost %.3f%%: %d trades, yearly %+.1f%%, Sharpe %+.2f" % (cost * 100, n, x.mean() * 25200, x.mean() / x.std() * 252 ** 0.5))
    daily, n = nasdaq_rule(o, h, l, c, 0.0002)
    s = pd.Series(daily, index=days).iloc[15:]
    mid = len(s) // 2
    sharpe = s.mean() / s.std() * 252 ** 0.5
    say("Halves: %+.1f%% then %+.1f%%. By year: %s" % (s.iloc[:mid].sum() * 100, s.iloc[mid:].sum() * 100,
        " ".join("%d:%+.1f%%" % (y, g.sum() * 100) for y, g in s.groupby(s.index.year))))
    money = 200.0
    for r in s:
        money = max(money + max(0.1, np.floor(money / 200) / 10) * 22900 * r, 0.0) if money >= 115 else money
    say("£200 at 0.1 lot per £200 would have ended at £%.0f" % money)
    say("Nasdaq rule verdict: %s" % ("PASS" if sharpe >= 0.5 and s.iloc[:mid].sum() > 0 and s.iloc[mid:].sum() > 0 else "FAIL"))
    say("")
    say("FLIP VERSION: first trade of the day only, with a stop (0.02% cost):")
    for stop in (0.0025, 0.005):
        d2, w2, n2 = nasdaq_flip_version(o, h, l, c, 0.0002, stop)
        x = pd.Series(d2, index=days).iloc[15:]
        say("Stop %.2f%%: %d trades, yearly %+.1f%%, Sharpe %+.2f, halves %+.1f%% then %+.1f%%" % (
            stop * 100, n2, x.mean() * 25200, x.mean() / x.std() * 252 ** 0.5, x.iloc[:len(x) // 2].sum() * 100, x.iloc[len(x) // 2:].sum() * 100))
    d2, w2, n2 = nasdaq_flip_version(o, h, l, c, 0.0002, 0.0025)
    say("From every start day in the last 2 years, stop 0.25%, did £200 reach the target before being wiped out?")
    say("Size | time | £400 | £1000 | £2000 | wiped")
    for span, label in ((63, "3 months"), (126, "6 months")):
        for lev in (50, 75, 100, 150):
            res, dead = rolling_flips(d2[15:], w2[15:], lev, span)
            say("%dx | %s | %.0f%% | %.0f%% | %.0f%% | %.0f%%" % (lev, label, res[400] * 100, res[1000] * 100, res[2000] * 100, dead * 100))


# ------------------------------------------------------------------ main


def main():
    rng = np.random.default_rng(SEED)
    say("RECENT EDGE SCAN, run on %s" % time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    hourly, five, daily, failed = [], [], {}, []
    for name, ticker, cost, cap in MARKETS:
        got = []
        for period, interval, store in (("729d", "1h", "h"), ("59d", "5m", "f"), ("5y", "1d", "d")):
            try:
                df = fetch(ticker, period, interval)
                if store == "d":
                    daily[name] = df
                elif store == "h" and len(df) > 1000:
                    hourly.append(arrays(df, name, cost, cap))
                elif store == "f" and len(df) > 1000:
                    five.append(arrays(df, name, cost, cap))
                got.append(interval)
            except Exception as exc:  # keep going if one market fails
                failed.append("%s %s: %s" % (name, interval, str(exc)[:70]))
            time.sleep(0.6)
    say("Markets with hourly data: %d, 5-minute: %d, daily: %d" % (len(hourly), len(five), len(daily)))
    if hourly:
        say("Hourly data from %s to %s" % (hourly[0]["day"][0], hourly[0]["day"][-1]))
    if failed:
        say("Could not download: " + "; ".join(failed[:12]) + (" ...and %d more" % (len(failed) - 12) if len(failed) > 12 else ""))
    if not hourly and not five and not daily:
        raise RuntimeError("No data could be downloaded. Yahoo may be blocking this runner.")
    if hourly:
        hourly_section(hourly, rng)
        events_section(hourly)
    if five:
        five_section(five, rng)
    if daily:
        daily_section(daily)
    try:
        exam_section(fetch("QQQ", "729d", "1h"))
    except Exception as exc:  # the exam is one part of a bigger report
        say("Nasdaq rule test could not run: %r" % (exc,))
    try:
        prop_section()
    except Exception as exc:  # the prop test is one part of a bigger report
        say("Prop firm test could not run: %r" % (exc,))
    say("")
    say("All percentages before leverage unless it says max lev. Past results do not promise future ones.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        say("")
        say("THE SCAN STOPPED WITH AN ERROR. Paste this whole report back:")
        say(traceback.format_exc())
    finally:
        with open("recent_report.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
