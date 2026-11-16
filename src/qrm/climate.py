"""Physical climate-risk overlay: winter storms erode recovery values  [Q xi].

    R_k(e) = R_k (1 - phi_k MDR_k(e)),        N_e ~ Poisson(f_e), independent across events and of Theta.

In every simulated year the events with N_e >= 1 "occur"; a counterparty hit by several events
takes the single worst one, m_k = max{MDR_k(e) : N_e >= 1} (m_k = 0 if no damaging event occurs).
The overlay never changes who defaults; it changes the loss given default:

    L_clim = sum_k E_k (1 - R_k (1 - phi m_k)) I_k = L + phi * sum_k E_k R_k m_k I_k .

This module is pure numpy: it works on a *storm catalogue*, i.e. the vector of annual frequencies
f_e and the matrix MDR_k(e) (events x counterparties).  The catalogue is produced from CLIMADA by
`qrm.climada_io` (script 06) and stored as an .npz file, so that everything downstream can be run
and tested on a machine without CLIMADA.
"""
from dataclasses import dataclass, field
import json
import numpy as np


# ------------------------------------------------------------------ the catalogue
@dataclass
class StormCatalogue:
    freq: np.ndarray                     # (n_events,)     annual frequencies f_e
    mdr: np.ndarray                      # (n_events, K)   MDR_k(e) at the counterparty's coordinates
    mdr_canton: np.ndarray               # (n_events, K)   MDR at the centroid of the counterparty's canton
    extra: dict = field(default_factory=dict)   # everything else stored in the .npz (footprints, impact function, ...)
    meta: dict = field(default_factory=dict)    # provenance (data set, CLIMADA version, synthetic flag, ...)

    @property
    def synthetic(self):
        return bool(self.meta.get("synthetic", False))

    def save(self, path):
        np.savez_compressed(path, freq=self.freq, mdr=self.mdr, mdr_canton=self.mdr_canton,
                            meta=json.dumps(self.meta), **self.extra)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        core = {"freq", "mdr", "mdr_canton", "meta"}
        return cls(freq=z["freq"].astype(float), mdr=z["mdr"].astype(float), mdr_canton=z["mdr_canton"].astype(float),
                   extra={k: z[k] for k in z.files if k not in core}, meta=json.loads(str(z["meta"])))


# ------------------------------------------------------------------ exact results (no simulation)
def worst_mdr_distribution(freq, mdr_k):
    """Exact law of m_k = max{MDR_k(e) : N_e >= 1} for one counterparty.

    Sort the damaging events by decreasing MDR, m_(1) >= m_(2) >= ...  The worst event of the year is
    event (j) iff none of the more damaging events (1..j-1) occurs and (j) does:
        P[m_k = m_(j)] = exp(-(f_(1)+...+f_(j-1))) (1 - exp(-f_(j))),      P[m_k = 0] = exp(-sum_j f_(j)).
    Returns (values, probabilities) with values in decreasing order, the last one being 0."""
    pos = mdr_k > 0
    order = np.argsort(-mdr_k[pos], kind="stable")
    m, f = mdr_k[pos][order], freq[pos][order]
    cum_before = np.r_[0.0, np.cumsum(f)[:-1]] if len(f) else np.zeros(0)
    p = np.exp(-cum_before) * -np.expm1(-f)
    return np.r_[m, 0.0], np.r_[p, np.exp(-f.sum())]


def expected_worst_mdr(freq, mdr):
    """E[m_k] for every counterparty, exactly (used for the expected loss with overlay)."""
    return np.array([np.dot(*worst_mdr_distribution(freq, mdr[:, k])) for k in range(mdr.shape[1])])


# ------------------------------------------------------------------ simulation of storm years
def sample_worst_literal(freq, mdr, n, rng):
    """The assignment's algorithm, literally: for every event e draw n trials N_e ~ Poisson(f_e);
    e occurs in a trial (a simulated year) if N_e >= 1; every counterparty keeps its worst event.

    Events with zero damage at all locations are skipped: they cannot change any recovery rate.
    Returns (worst, n_damaging_events_per_year); worst has shape (n, K)."""
    worst = np.zeros((n, mdr.shape[1]))
    count = np.zeros(n, dtype=int)
    for e in np.flatnonzero(mdr.max(axis=1) > 0):
        occ = rng.poisson(freq[e], size=n) >= 1
        if occ.any():
            count += occ
            worst[occ] = np.maximum(worst[occ], mdr[e])
    return worst, count


def draw_event_years(freq, active, n, rng):
    """Fast sampler, equivalent in law to the literal one.

    Independent Poisson counts N_e with means f_e are the same thing as one Poisson number of events
    M ~ Poisson(F), F = sum_e f_e, each event being e with probability f_e / F (superposition property).
    Only the set of events that occur matters for the overlay, so the two samplers give the same law
    of the worst damage ratios.  `active` restricts the catalogue to the events that can cause damage.
    Returns (year, event, starts): arrays of the simulated occurrences sorted by year, and the index
    of the first occurrence of every year that has at least one."""
    f = freq[active]
    cnt = rng.poisson(f.sum(), size=n)
    year = np.repeat(np.arange(n), cnt)
    event = rng.choice(active, size=len(year), p=f / f.sum()) if len(year) else np.zeros(0, dtype=int)
    starts = np.flatnonzero(np.r_[True, year[1:] != year[:-1]]) if len(year) else np.zeros(0, dtype=int)
    return year, event, starts


def worst_from_events(mdr, year, event, starts, n, dtype=np.float32):
    """Worst MDR per simulated year and counterparty from the occurrences drawn by draw_event_years."""
    worst = np.zeros((n, mdr.shape[1]), dtype=dtype)
    if len(year):
        worst[year[starts]] = np.maximum.reduceat(mdr[event].astype(dtype), starts, axis=0)
    return worst


def sample_worst_fast(freq, mdr, n, rng, dtype=np.float32):
    active = np.flatnonzero(mdr.max(axis=1) > 0)
    if active.size == 0:
        return np.zeros((n, mdr.shape[1]), dtype=dtype)
    return worst_from_events(mdr, *draw_event_years(freq, active, n, rng), n, dtype)


def sample_worst_independent(freq, mdr, n, rng, dtype=np.float32):
    """Counterfactual 'perfectly diversified' portfolio: every counterparty draws its own storm year.

    Each m_k keeps exactly its marginal law (same hazard at its location, hence the same expected
    add-on), but the m_k are independent across names, as if the 100 sites lay in 100 regions that
    are never hit by the same event.  Sampling is by inversion of the exact law of m_k."""
    worst = np.empty((n, mdr.shape[1]), dtype=dtype)
    for k in range(mdr.shape[1]):
        val, p = worst_mdr_distribution(freq, mdr[:, k])
        worst[:, k] = val[np.minimum(np.searchsorted(np.cumsum(p), rng.random(n)), len(val) - 1)]
    return worst


def comonotone_order(severity, loss):
    """Permutation that pairs the most severe storm years with the largest credit losses.

    Returns idx such that worst[idx] is the comonotone coupling of the storm years with `loss`
    (rank of the loss = rank of the storm severity; ties are broken by the order of simulation)."""
    rank_loss = np.argsort(np.argsort(loss, kind="stable"), kind="stable")
    return np.argsort(severity, kind="stable")[rank_loss]


# ------------------------------------------------------------------ losses
def adjusted_lgd(R, worst, phi):
    """1 - R_k(e) = 1 - R_k (1 - phi m_k); equals 1 - R_k when no damaging event hits k."""
    return 1.0 - np.asarray(R) * (1.0 - phi * worst)


def overlay_losses(I, E, R, worst, phi):
    """Portfolio loss per scenario with climate-adjusted recovery rates; I is the (n, K) default matrix."""
    return (I * (np.asarray(E) * adjusted_lgd(R, worst, phi))).sum(axis=1)


def addon(I, ER, worst):
    """sum_k E_k R_k m_k I_k per scenario: the overlay loss is L + phi * addon (linear in phi)."""
    return np.einsum("nk,nk->n", I * np.asarray(ER, dtype=worst.dtype), worst)


# ------------------------------------------------------------------ synthetic catalogue (tests only)
def demo_catalogue(lat, lon, canton_codes, rng, n_events=1500):
    """SYNTHETIC catalogue with the same structure as the CLIMADA export - for testing the code
    path without CLIMADA.  It is flagged as synthetic and the report builder refuses to use it."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    K = len(lat)
    freq = np.full(n_events, 3.0 / n_events) * rng.uniform(0.5, 1.5, n_events)
    # gust field: common event strength + smooth west-east gradient + local noise (m/s)
    strength = 16 + 4.5 * rng.gamma(2.0, 1.0, n_events)
    grad = rng.normal(0, 1.5, n_events)[:, None] * (lon - lon.mean())[None, :]
    inten = np.clip(strength[:, None] + grad + rng.normal(0, 2.0, (n_events, K)), 0, None)
    impf_x = np.arange(0, 121, 1.0)
    impf_mdr = 1.2e-6 * np.clip(impf_x - 20, 0, None) ** 2.6                    # invented vulnerability curve
    impf_mdr = np.minimum(impf_mdr, 1.0)
    mdr = np.interp(inten, impf_x, impf_mdr)
    codes = np.asarray(canton_codes)
    mdr_canton = np.empty_like(mdr)
    for c in np.unique(codes):                                                   # canton value: mean over its names
        mdr_canton[:, codes == c] = mdr[:, codes == c].mean(axis=1, keepdims=True)
    top = int(np.argmax(mdr.sum(axis=1)))
    g_lat, g_lon = np.meshgrid(np.linspace(45.8, 47.8, 40), np.linspace(5.9, 10.5, 70), indexing="ij")
    foot = strength[top] + grad[top, 0] / max(lon[0] - lon.mean(), 1e-9) * (g_lon - lon.mean()) if K else g_lon * 0
    extra = {"inten": inten.astype(np.float32), "inten_canton": inten.astype(np.float32),
             "impf_intensity": impf_x, "impf_mdr": impf_mdr,
             "cent_lat": g_lat.ravel(), "cent_lon": g_lon.ravel(),
             "footprint_top": foot.ravel().astype(np.float32), "footprint_max": (foot.ravel() + 5).astype(np.float32),
             "event_name": np.array([f"demo{e:04d}" for e in range(n_events)]),
             "nearest_km": np.zeros(K), "top_event": np.array(top)}
    meta = {"synthetic": True, "dataset": "SYNTHETIC demo catalogue", "impact_function": "invented",
            "intensity_unit": "m/s", "climada_version": "n/a"}
    return StormCatalogue(freq=freq, mdr=mdr, mdr_canton=mdr_canton, extra=extra, meta=meta)
