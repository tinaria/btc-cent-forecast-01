"""Backtest for btc-cent-forecast-01: pseudo-cases cut from the 6 practice files.

Each practice file holds 4320 minute closes (72h). A pseudo-case takes a
checkpoint every 4h inside the file (at -4h, -8h, ..., -68h), uses only the
prices up to that checkpoint, and scores the guess against the price 240 min
later (which is inside the same file). That gives ~17 pseudo-cases per file.
Plus the 5 real practice targets (p00->p01 ... p04->p05).

Models are compared by mean absolute error in cents, against persistence.
"""
import json, math

KIT = '/home/hatch/workspace/makefun/btc-cent-forecast-01-kit/input'

def load(i):
    d = json.load(open(f'{KIT}/p0{i}.in'))
    return d['markets'][0]['prices']

cases = [load(i) for i in range(6)]

# ---- pseudo-cases: (prices_up_to_checkpoint, actual_240min_later) ----
pseudo = []
for prices in cases:
    n = len(prices)
    # checkpoint index: last index is n-1 (at). Step back 240 min at a time.
    for back in range(240, n - 240 + 1, 240):
        ck = n - 1 - back
        pseudo.append((prices[:ck + 1], prices[ck + 240]))
# real practice targets
real = []
for i in range(5):
    real.append((cases[i], cases[i + 1][-1]))

def mae(model, dataset):
    errs = []
    for prices, actual in dataset:
        g = model(prices)
        if g is None or not (g > 0) or not math.isfinite(g):
            g = prices[-1]
        errs.append(abs(g - actual) * 100)
    return sum(errs) / len(errs), max(errs)

# ---- candidate models ----
def persistence(p):
    return p[-1]

def lin_trend(p, hours, damp):
    # OLS on log prices over the last `hours`, extrapolate 240 min, damped.
    n = len(p)
    m = int(hours * 60)
    xs = p[-m:]
    if len(xs) < 10:
        return p[-1]
    logs = [math.log(x) for x in xs]
    t = list(range(len(xs)))
    mt = sum(t) / len(t)
    ml = sum(logs) / len(logs)
    den = sum((x - mt) ** 2 for x in t)
    slope = sum((t[i] - mt) * (logs[i] - ml) for i in range(len(xs))) / den if den else 0.0
    # slope is log-return per minute; extrapolate 240 min, damped
    return p[-1] * math.exp(slope * 240 * damp)

def ewma_drift(p, halflife_min, shrink):
    # EWMA of 1-min log returns, extrapolated 240 min, shrunk.
    alpha = 1 - 0.5 ** (1.0 / halflife_min)
    r = 0.0
    w = 0.0
    for i in range(1, len(p)):
        lr = math.log(p[i] / p[i - 1])
        r = (1 - alpha) * r + alpha * lr
        w = (1 - alpha) * w + alpha
    drift = (r / w) if w else 0.0
    return p[-1] * math.exp(drift * 240 * shrink)

def mean_revert(p, hours, a):
    m = int(hours * 60)
    xs = p[-m:]
    mean = sum(xs) / len(xs)
    return p[-1] + a * (mean - p[-1])

def momentum(p, lookback_min, k):
    m = int(lookback_min)
    if len(p) <= m:
        return p[-1]
    ret = p[-1] / p[-m] - 1
    return p[-1] * (1 + k * ret)

def combo(p, hours, damp, a):
    # damped trend + mean reversion to the same window's mean
    t = lin_trend(p, hours, damp)
    mr = mean_revert(p, hours, a)
    return (t + mr) / 2

cands = {
    'M0 persistence': persistence,
    'M1 trend 1h d0.2': lambda p: lin_trend(p, 1, 0.2),
    'M1 trend 1h d0.5': lambda p: lin_trend(p, 1, 0.5),
    'M1 trend 4h d0.2': lambda p: lin_trend(p, 4, 0.2),
    'M1 trend 4h d0.5': lambda p: lin_trend(p, 4, 0.5),
    'M1 trend 12h d0.2': lambda p: lin_trend(p, 12, 0.2),
    'M1 trend 12h d0.5': lambda p: lin_trend(p, 12, 0.5),
    'M1 trend 24h d0.3': lambda p: lin_trend(p, 24, 0.3),
    'M1 trend 24h d1.0': lambda p: lin_trend(p, 24, 1.0),
    'M2 ewma 60m k0.3': lambda p: ewma_drift(p, 60, 0.3),
    'M2 ewma 240m k0.3': lambda p: ewma_drift(p, 240, 0.3),
    'M2 ewma 240m k0.5': lambda p: ewma_drift(p, 240, 0.5),
    'M3 revert 24h a0.2': lambda p: mean_revert(p, 24, 0.2),
    'M3 revert 24h a0.4': lambda p: mean_revert(p, 24, 0.4),
    'M3 revert 72h a0.2': lambda p: mean_revert(p, 72, 0.2),
    'M5 momentum 4h k0.2': lambda p: momentum(p, 240, 0.2),
    'M5 momentum 4h k0.5': lambda p: momentum(p, 240, 0.5),
    'M5 momentum 24h k0.2': lambda p: momentum(p, 1440, 0.2),
    'M4 combo 12h d0.3 a0.2': lambda p: combo(p, 12, 0.3, 0.2),
    'M4 combo 24h d0.3 a0.2': lambda p: combo(p, 24, 0.3, 0.2),
}

print(f'pseudo-cases: {len(pseudo)}, real practice targets: {len(real)}')
print(f'{"model":28s} {"pseudo MAE":>11s} {"pseudo worst":>12s} {"real MAE":>10s}')
for name, fn in cands.items():
    pm, pw = mae(fn, pseudo)
    rm, rw = mae(fn, real)
    print(f'{name:28s} {pm:11.1f} {pw:12.1f} {rm:10.1f}')
base_p, _ = mae(persistence, pseudo)
base_r, _ = mae(persistence, real)
print(f'persistence pseudo MAE: {base_p:.1f} cents, real MAE: {base_r:.1f} cents')
print('note: stated practice baseline (6 cases incl p05) = 44521.6 cents')
