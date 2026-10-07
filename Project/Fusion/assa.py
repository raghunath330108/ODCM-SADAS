"""ASSA (manuscript Eq. 16-23, Algorithm 1) searching the global fusion-weight vector (wc, wl, wr).

Minimisation of `fitness(w)` (the caller passes -Fitness of Eq. 12). Each solution x lives in [0, 1]^3 and is
normalised to w = x / sum(x) before evaluation (Eq. 9-10); a zero vector maps to equal weights.
"""
import math

import numpy as np


def normalise(x):
    s = float(np.sum(x))
    return np.full(len(x), 1.0 / len(x)) if s <= 1e-12 else np.asarray(x, float) / s


def levy_step(dim, lam, rng):
    """Mantegna's algorithm (Eq. 20)."""
    sigma = (math.gamma(1 + lam) * math.sin(math.pi * lam / 2) / (math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
    return rng.normal(0, sigma, dim) / np.abs(rng.normal(0, 1, dim)) ** (1 / lam)


def adaptive_c1(a, t, t_max, cmin=0.05, cmax=0.95):
    """Eq. 22."""
    return float(np.clip(cmax + (cmin - cmax) * math.log10(a + 10 * t / t_max), cmin, cmax))


def assa(fitness, dim=3, n_max=20, n_min=5, iters=30, lam=1.5, cmin=0.05, cmax=0.95, xmin=0.0, xmax=1.0, seed=0, log=print):
    rng = np.random.default_rng(seed)
    fes_max = n_max + iters * (n_max + n_min) / 2  # evaluation budget the Eq. 23 schedule is spread over

    def evaluate(x):
        return float(fitness(normalise(x)))

    pop = xmin + rng.random((n_max, dim)) * (xmax - xmin)  # Eq. 16
    fit = np.array([evaluate(x) for x in pop])
    fes = n_max
    best = int(np.argmin(fit))
    x_best, f_best = pop[best].copy(), float(fit[best])
    history = [{"iter": 0, "n": n_max, "best_fitness": f_best, "best_w": normalise(x_best).tolist(), "evals": fes}]
    log(f"ASSA iter 0/{iters}: n={n_max} best -Fitness={f_best:.5f} w={np.round(normalise(x_best), 3).tolist()}")

    for t in range(1, iters + 1):
        a = 2.0 * (1 - t / iters)  # linearly 2 -> 0
        c1 = adaptive_c1(a, t, iters, cmin, cmax)
        for i in range(len(pop)):
            ref = xmin + rng.random(dim) * (xmax - xmin)  # random Eq. 16 solution drives the update (Eq. 17)
            xs = []
            for _ in range(3):
                A = 2 * a * rng.random(dim) - a
                C = 2 * rng.random(dim)
                xs.append(pop[i] - A * C * np.abs(ref - pop[i]))  # Eq. 17
            x_new = np.mean(xs, axis=0)  # Eq. 18
            x_new = x_new + c1 * levy_step(dim, lam, rng) * (x_best - x_new)  # Eq. 19 (alpha = c1, Eq. 22)
            x_new = np.clip(x_new, xmin, xmax)
            f_new = evaluate(x_new)
            fes += 1
            if f_new < fit[i]:  # Eq. 21 (greedy, minimisation)
                pop[i], fit[i] = x_new, f_new
        best = int(np.argmin(fit))
        if fit[best] < f_best:
            x_best, f_best = pop[best].copy(), float(fit[best])
        n_next = max(n_min, int(round((n_min - n_max) / fes_max * fes + n_max)))  # Eq. 23
        if n_next < len(pop):
            keep = np.argsort(fit)[:n_next]
            pop, fit = pop[keep], fit[keep]
        history.append({"iter": t, "n": len(pop), "best_fitness": f_best, "best_w": normalise(x_best).tolist(), "evals": fes, "c1": c1})
        log(f"ASSA iter {t}/{iters}: n={len(pop)} evals={fes} c1={c1:.3f} best -Fitness={f_best:.5f} w={np.round(normalise(x_best), 3).tolist()}")
    return normalise(x_best), f_best, history
