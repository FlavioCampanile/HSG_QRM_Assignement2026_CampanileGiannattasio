"""Empirical value-at-risk and expected shortfall of a simulated loss sample  [Q viii, ix, xi].

Conventions follow Chapter 1 of the lecture notes.  With X = -L the profit-and-loss,
    VaR_a(X) = -inf{x : P[X <= x] >= 1-a},
    ES_a(X)  = (1/(1-a)) { E[-X 1{X <= -VaR}] - VaR (P[X <= -VaR] - (1-a)) }.
For N equally likely simulated losses l_(1) >= l_(2) >= ... (sorted in *decreasing* order)
this gives VaR = l_(k*) with k* = min{k : k/N >= 1-a}, and ES equal to the average of the
N(1-a) largest losses (with the tie/discreteness correction of the second term).
"""
import numpy as np


def var_es(loss: np.ndarray, alpha: float) -> tuple[float, float]:
    loss = np.asarray(loss, dtype=float)
    n = loss.size
    tail = n * (1.0 - alpha)                      # N(1-a), e.g. 10 for N=10,000 and a=99.9%
    k = max(int(np.ceil(tail - 1e-9)), 1)         # k* (guard against floating point noise)
    top = np.partition(loss, n - k)[n - k:]       # the k largest losses
    v = top.min()                                 # VaR = k*-th largest loss
    m = np.count_nonzero(loss >= v)               # N * P[X <= -VaR]  (>= k when there are ties)
    s = loss[loss >= v].sum()                     # N * E[-X 1{X <= -VaR}]
    es = (s - v * (m - tail)) / tail
    return float(v), float(es)


def var(loss, alpha):
    return var_es(loss, alpha)[0]


def es(loss, alpha):
    return var_es(loss, alpha)[1]
