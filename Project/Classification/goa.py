"""GOA (grasshopper optimisation, manuscript Eq. 32-34) over the Table 3 LSTM search space; maximises the fitness (Eq. 35-36).

Positions live in [0, 1]^6 (lb = 0, ub = 1) and are decoded to the Table 3 hyperparameters.
"""
import math

import numpy as np

CHOICES = {"batch": [8, 16, 32, 64], "layers": [1, 2, 3], "seq_len": [5, 8, 10]}
LR_RANGE, DROPOUT_RANGE, HIDDEN_RANGE = (1e-4, 1e-2), (0.1, 0.5), (64, 256)
NAMES = ("lr", "dropout", "hidden", "batch", "layers", "seq_len")
DIM = len(NAMES)


def decode(x):
    """x in [0, 1]^6 -> Table 3 hyperparameters (learning rate on a log scale; discrete ones snap to the nearest option)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    pick = lambda opts, v: opts[min(int(v * len(opts)), len(opts) - 1)]
    lo, hi = map(math.log10, LR_RANGE)
    return {
        "lr": float(10 ** (lo + x[0] * (hi - lo))),
        "dropout": float(DROPOUT_RANGE[0] + x[1] * (DROPOUT_RANGE[1] - DROPOUT_RANGE[0])),
        "hidden": int(round(HIDDEN_RANGE[0] + x[2] * (HIDDEN_RANGE[1] - HIDDEN_RANGE[0]))),
        "batch": pick(CHOICES["batch"], x[3]), "layers": pick(CHOICES["layers"], x[4]), "seq_len": pick(CHOICES["seq_len"], x[5]),
    }


def social_force(r, f=0.5, l=1.5):
    """Eq. 34: s(r) = f e^(-r/l) - e^(-r)."""
    return f * np.exp(-r / l) - np.exp(-r)


def goa(fitness, pop=10, iters=15, cmax=1.0, cmin=1e-5, f=0.5, l=1.5, seed=0, log=print):
    """fitness(params dict) -> float, maximised. Returns (best params, best fitness, history, evaluated candidates)."""
    rng = np.random.default_rng(seed)
    memo = {}

    def evaluate(x):
        p = decode(x)
        key = tuple(p.values())
        if key not in memo:
            memo[key] = {"params": p, "fitness": float(fitness(p))}
        return memo[key]["fitness"]

    X = rng.random((pop, DIM))
    fit = np.array([evaluate(x) for x in X])
    best = int(np.argmax(fit))
    target, f_best = X[best].copy(), float(fit[best])
    history = [{"iter": 0, "best_fitness": f_best, "best_params": decode(target), "candidates": len(memo)}]
    log(f"GOA iter 0/{iters}: best fitness={f_best:.4f} params={decode(target)}")
    for t in range(1, iters + 1):
        c = cmax - t * (cmax - cmin) / iters  # Eq. 33
        new = np.empty_like(X)
        for i in range(pop):
            s_i = np.zeros(DIM)
            for j in range(pop):
                if j == i:
                    continue
                diff = X[j] - X[i]
                dist = np.linalg.norm(diff) + 1e-12
                r = 2.0 + dist % 2.0  # distance mapped to [2, 4) as in the reference GOA implementation
                s_i += c * (1.0 - 0.0) / 2 * social_force(r, f, l) * diff / dist  # (ub - lb) = 1
            new[i] = np.clip(c * s_i + target, 0.0, 1.0)  # Eq. 32
        X = new
        fit = np.array([evaluate(x) for x in X])
        best = int(np.argmax(fit))
        if fit[best] > f_best:
            target, f_best = X[best].copy(), float(fit[best])
        history.append({"iter": t, "best_fitness": f_best, "best_params": decode(target), "candidates": len(memo)})
        log(f"GOA iter {t}/{iters}: c={c:.3f} best fitness={f_best:.4f} params={decode(target)} (distinct candidates {len(memo)})")
    return decode(target), f_best, history, list(memo.values())
