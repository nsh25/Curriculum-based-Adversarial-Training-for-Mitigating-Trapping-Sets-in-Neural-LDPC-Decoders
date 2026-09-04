"""Frame-error-rate estimation: plain Monte-Carlo and mean-shift importance
sampling around a trapping set.

Channel model (all-zero codeword, BPSK, AWGN):
    transmit x = +1 on every coordinate
    receive  y = x + n,   n ~ N(0, sigma^2 I),  sigma^2 = 1 / (2 R Eb/N0)

Monte-Carlo draws n from the true density f and counts decoder failures. It is
correct but hopeless in the error floor, where a failure needs ~1e-9 of the
samples.

Importance sampling draws n from a *biased* density g that pushes the received
values on a chosen trapping set T toward the wrong sign, so failures are common,
then de-biases each sample with the likelihood ratio w = f(n)/g(n). For a mean
shift mu (supported on T) with the same covariance,

    w = exp( (||mu||^2 - 2 n . mu) / (2 sigma^2) ),      n = y - x

and  FER ~= mean( w * 1[decoder fails] ).  The estimator is unbiased for any mu;
a good mu (pointing at the trapping set, magnitude ~ a couple of sigma) collapses
the variance by many orders of magnitude versus MC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decoder import decode
from .tanner import Tanner
from .trap_search import TrapSet


def ebn0_to_sigma(ebn0_db: float, rate: float) -> float:
    """sigma of the AWGN noise for Es=1 BPSK at the given Eb/N0 (dB)."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    esn0 = rate * ebn0                      # energy per *coded* bit
    sigma2 = 1.0 / (2.0 * esn0)
    return float(np.sqrt(sigma2))


@dataclass
class Estimate:
    fer: float
    stderr: float
    rel_err: float
    n_samples: int
    n_events: float                       # (weighted) failure count
    ber: float = 0.0                      # coded BER over tx bits (MC only)
    n_bit_errors: float = 0.0             # coded bit-error count (MC only)

    def __str__(self) -> str:
        s = (f"FER={self.fer:.3e}  +/-{self.stderr:.2e}  "
             f"(rel {self.rel_err:.1%}, N={self.n_samples}, "
             f"events={self.n_events:.1f})")
        if self.n_bit_errors > 0 or self.ber > 0:
            s += f"  BER={self.ber:.3e} ({self.n_bit_errors:.0f} bits)"
        return s


def _fail_mask(t: Tanner, res, target: TrapSet | None) -> np.ndarray:
    """A frame error, optionally restricted to failures that land on T."""
    bad = ~res["converged"]
    if target is None:
        return bad
    Tset = set(target.vns)
    out = np.zeros_like(bad)
    for row in np.nonzero(bad)[0]:
        supp = set(np.nonzero(res["bits"][row])[0].tolist())
        # "attributable to T": every variable of the trapping set ended up in
        # error (T is contained in the residual error support). Absorbing sets
        # typically settle to exactly T, but a few stray bits are allowed.
        if supp and Tset.issubset(supp):
            out[row] = True
    return out


def make_llr(y, sigma2, quant_step, quant_gain, punctured_mask):
    """Channel LLR front-end. Float decoder uses the exact LLR 2y/sigma^2; the
    fixed-point decoder instead feeds an AGC-scaled sample ``quant_gain*y`` into
    the quantizer grid (min-sum is scale-invariant, so only the ratio of the LLR
    to the quantization step/clip matters -- and raw 2y/sigma^2 would saturate
    the +/-clip at high SNR and wreck the decoder)."""
    Lch = (quant_gain * y) if quant_step is not None else (2.0 * y / sigma2)
    if punctured_mask is not None:
        Lch[:, punctured_mask] = 0.0
    return Lch


def monte_carlo(
    t: Tanner,
    sigma: float,
    trials: int,
    batch: int = 5000,
    max_iter: int = 50,
    rule: str = "minsum",
    target: TrapSet | None = None,
    rng: np.random.Generator | None = None,
    punctured_mask: np.ndarray | None = None,
    quant_step: float | None = None,
    quant_clip: float = 7.5,
    quant_gain: float = 2.0,
    target_errors: int | None = None,
) -> Estimate:
    """Plain Monte-Carlo FER.

    Runs at least ``batch`` words and stops when either ``trials`` words have
    been simulated or (if ``target_errors`` is set) that many frame errors have
    been collected -- the usual "run to N errors" rule for a reliable FER point.
    ``trials`` then acts as a hard budget cap so deep-floor points still return.
    """
    rng = rng or np.random.default_rng(1)
    sigma2 = sigma * sigma
    fails = 0
    done = 0
    while done < trials:
        B = min(batch, trials - done)
        y = 1.0 + rng.normal(0.0, sigma, size=(B, t.N))
        Lch = make_llr(y, sigma2, quant_step, quant_gain, punctured_mask)
        res = decode(t, Lch, max_iter=max_iter, rule=rule,
                     quant_step=quant_step, quant_clip=quant_clip)
        fails += int(_fail_mask(t, res, target).sum())
        done += B
        if target_errors is not None and fails >= target_errors:
            break
    fer = fails / done
    stderr = np.sqrt(max(fer * (1 - fer), 0) / done)
    rel = stderr / fer if fer > 0 else float("inf")
    return Estimate(fer, stderr, rel, done, float(fails))


def importance_sampling(
    t: Tanner,
    target: TrapSet | None,
    sigma: float,
    trials: int,
    shift: float = 3.0,
    batch: int = 5000,
    max_iter: int = 50,
    rule: str = "minsum",
    attribute_to_target: bool = True,
    rng: np.random.Generator | None = None,
    punctured_mask: np.ndarray | None = None,
    quant_step: float | None = None,
    quant_clip: float = 7.5,
    quant_gain: float = 2.0,
) -> Estimate:
    """Mean-shift importance-sampling FER estimate.

    * ``target`` is a ``TrapSet`` (the useful mode) – bias only that set's
      variable nodes by ``-shift*sigma`` (toward the -1 region) so the decoder
      falls into that trapping set. Estimates the set's *floor contribution*
      (with ``attribute_to_target=True``), unreachable by Monte-Carlo. The shift
      lives on a low-dimensional subspace, which keeps the likelihood ratio -
      and hence the estimator variance - under control.

    * ``target is None`` – bias every (un-punctured) coordinate. Kept only for
      experimentation: over ~600 dimensions ``||mu||^2`` is huge, the weights
      span dozens of orders of magnitude, and the estimate is unreliable. Do
      NOT use this to validate against MC; use restricted-MC vs set-IS instead.

    Weights de-bias exactly, so the estimate is unbiased for any ``shift``.
    """
    rng = rng or np.random.default_rng(2)
    sigma2 = sigma * sigma

    mu = np.zeros(t.N)
    if target is not None:
        mu[np.asarray(target.vns, dtype=np.int64)] = -shift * sigma
    else:
        mu[:] = -shift * sigma
        if punctured_mask is not None:
            mu[punctured_mask] = 0.0   # punctured coords carry no channel info
    mu_sq = float(mu @ mu)

    wsum = 0.0
    w2sum = 0.0
    done = 0
    while done < trials:
        B = min(batch, trials - done)
        n = rng.normal(0.0, sigma, size=(B, t.N)) + mu      # draw from g
        y = 1.0 + n
        Lch = make_llr(y, sigma2, quant_step, quant_gain, punctured_mask)
        res = decode(t, Lch, max_iter=max_iter, rule=rule,
                     quant_step=quant_step, quant_clip=quant_clip)
        fail = _fail_mask(t, res, target if attribute_to_target else None)

        # likelihood ratio w = f(n)/g(n)
        logw = (mu_sq - 2.0 * (n @ mu)) / (2.0 * sigma2)
        w = np.exp(logw)
        w = np.where(fail, w, 0.0)
        wsum += float(w.sum())
        w2sum += float((w * w).sum())
        done += B

    fer = wsum / trials
    var = max(w2sum / trials - fer * fer, 0.0) / trials
    stderr = np.sqrt(var)
    rel = stderr / fer if fer > 0 else float("inf")
    # "effective events" = wsum^2/w2sum, the IS analogue of a failure count
    eff = (wsum * wsum / w2sum) if w2sum > 0 else 0.0
    return Estimate(fer, stderr, rel, trials, eff)


def importance_sampling_mixture(
    t: Tanner,
    sets: list[TrapSet],
    sigma: float,
    trials: int,
    shift: float = 2.0,
    batch: int = 5000,
    max_iter: int = 50,
    rule: str = "minsum",
    rng: np.random.Generator | None = None,
    punctured_mask: np.ndarray | None = None,
    quant_step: float | None = None,
    quant_clip: float = 7.5,
    quant_gain: float = 2.0,
) -> Estimate:
    """Multi-set (mixture) importance sampling for the **overall** floor FER.

    A single narrow shift only covers one trapping set; the error floor is the
    union of many. Here the proposal is a uniform mixture over all K given sets,

        g(n) = (1/K) sum_k N(n; mu_k, sigma^2 I),   mu_k = -shift on T_k,

    so each drawn sample is biased toward a randomly chosen set. ``shift`` is an
    *absolute* mean displacement in signal units (transmitted amplitude = +1):
    shift ~ 2 pushes the set's received samples from +1 to about -1, reliably
    triggering the trapping set at every SNR. (A shift measured in sigma-units
    would shrink as SNR grows and stop covering the failure region.) Every
    decoder failure is counted, weighted by the exact ratio w = f(n)/g(n):

        w = K / sum_k exp( (2 n.mu_k - ||mu_k||^2) / (2 sigma^2) ).

    This estimates the total probability of falling into *any* of the enumerated
    trapping sets -- the quantity plain MC measures -- and is what should be
    compared against Monte-Carlo.
    """
    rng = rng or np.random.default_rng(3)
    sigma2 = sigma * sigma
    K = len(sets)
    if K == 0:
        raise ValueError("need at least one trapping set")

    # indicator matrix A[K, N]: 1 on each set's variable nodes
    A = np.zeros((K, t.N), dtype=np.float64)
    for k, s in enumerate(sets):
        A[k, np.asarray(s.vns, dtype=np.int64)] = 1.0
    a_size = A.sum(axis=1)                         # [K] set sizes a_k
    amp = shift                                    # |mu| per coordinate (absolute)

    wsum = 0.0
    w2sum = 0.0
    done = 0
    while done < trials:
        B = min(batch, trials - done)
        ks = rng.integers(0, K, size=B)            # chosen set per sample
        n = rng.normal(0.0, sigma, size=(B, t.N))
        n -= amp * A[ks]                           # add mu_k (shift toward set)
        y = 1.0 + n
        Lch = make_llr(y, sigma2, quant_step, quant_gain, punctured_mask)
        res = decode(t, Lch, max_iter=max_iter, rule=rule,
                     quant_step=quant_step, quant_clip=quant_clip)
        fail = ~res["converged"]

        # per-set exponents: (2 n.mu_k - ||mu_k||^2)/(2 sigma^2)
        #   n.mu_k = -amp * S_k,  S_k = sum_{v in T_k} n_v = (n @ A.T)
        #   ||mu_k||^2 = a_k * amp^2
        S = n @ A.T                                # [B, K]
        expo = (-amp * S) / sigma2 - a_size[None, :] * (amp * amp) / (2.0 * sigma2)
        # w = K / sum_k exp(expo_k)   (log-sum-exp for stability)
        m = expo.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(expo - m).sum(axis=1))
        w = np.exp(np.log(K) - lse)
        w = np.where(fail, w, 0.0)
        wsum += float(w.sum())
        w2sum += float((w * w).sum())
        done += B

    fer = wsum / trials
    var = max(w2sum / trials - fer * fer, 0.0) / trials
    stderr = np.sqrt(var)
    rel = stderr / fer if fer > 0 else float("inf")
    eff = (wsum * wsum / w2sum) if w2sum > 0 else 0.0
    return Estimate(fer, stderr, rel, trials, eff)
