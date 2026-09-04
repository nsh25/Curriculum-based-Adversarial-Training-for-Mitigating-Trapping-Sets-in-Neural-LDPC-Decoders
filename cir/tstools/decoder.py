"""Vectorised belief-propagation decoder for the Tanner container.

Flooding-schedule, log-domain, batched over many received words at once.
Two check-node rules are provided:

* ``"minsum"``  – normalised (scaled) min-sum, alpha≈0.75. This is what real
  5G NR receivers use and it is numerically bullet-proof, so it is the default.
* ``"spa"``     – true sum-product (tanh rule), a touch slower but exact.

All decoding is done under the standard all-zero-codeword assumption for a
symmetric channel: BPSK maps bit 0 -> +1, so the transmitted signal is +1 on
every un-punctured coordinate and the channel LLR is 2*y/sigma^2. This is valid
for FER / trapping-set work and needs no generator matrix.
"""
from __future__ import annotations

import numpy as np

from .tanner import Tanner


def channel_llr(y: np.ndarray, sigma2: float, punctured_mask: np.ndarray | None = None) -> np.ndarray:
    """LLR for BPSK over AWGN.  y shape [B, N].  Punctured VNs get LLR 0."""
    llr = 2.0 * y / sigma2
    if punctured_mask is not None:
        llr = llr.copy()
        llr[:, punctured_mask] = 0.0
    return llr


def quantize(x: np.ndarray, step: float, clip: float) -> np.ndarray:
    """Uniform mid-tread quantizer: round to a multiple of ``step``, clip to
    +/-``clip``. With step=0.5, clip=7.5 the alphabet is {-7.5,-7.0,...,7.5}
    (31 levels, i.e. 5-bit signed) -- the fixed-point grid a hardware min-sum
    decoder actually uses.
    """
    q = np.round(x / step) * step
    return np.clip(q, -clip, clip, out=q)


def decode(
    t: Tanner,
    Lch: np.ndarray,
    max_iter: int = 50,
    rule: str = "minsum",
    alpha: float = 0.75,
    early_stop: bool = True,
    llr_clip: float = 30.0,
    quant_step: float | None = None,
    quant_clip: float = 7.5,
):
    """Decode a batch of channel-LLR vectors.

    Parameters
    ----------
    Lch : [B, N] channel LLRs (positive => bit 0 more likely).
    quant_step : if set, the decoder runs in fixed point -- the channel LLRs and
        every check->var / var->check message are uniformly quantized to a grid
        of this step and clipped to +/-``quant_clip`` (default 7.5). ``None``
        keeps the float reference decoder. When quantizing, ``quant_clip``
        replaces ``llr_clip`` as the saturation level.
    Returns dict with:
        bits      : [B, N] int8 hard decisions
        converged : [B]    bool, syndrome satisfied
        iters     : [B]    iterations actually run (early-stop aware)
    """
    Lch = np.atleast_2d(Lch).astype(np.float32)
    B, N = Lch.shape
    assert N == t.N
    E = t.E
    cn_edges, cn_mask = t.cn_edges, t.cn_mask          # [M, dc]
    edge_vn = t.edge_vn                                # [E]

    fixed_point = quant_step is not None
    if fixed_point:
        llr_clip = quant_clip                          # saturate at the grid edge
        Lch = quantize(Lch, quant_step, quant_clip)    # quantize channel LLRs

    bits = (Lch < 0).astype(np.int8)
    converged = np.zeros(B, dtype=bool)
    iters = np.full(B, max_iter, dtype=np.int64)

    # Active-set compaction: once a word's syndrome is satisfied it is frozen and
    # dropped from the working arrays, so late iterations only cost what the few
    # stubborn (trapping-set!) words actually need. `act` maps rows of the
    # working arrays back to original batch indices.
    act = np.arange(B)
    La = Lch                                           # active channel LLRs
    mvc = La[:, edge_vn].copy()                        # [Ba, E] var->check msgs
    flat_edges = cn_edges[cn_mask]                     # [E] edge id per (check,slot)
    ar_dc = np.arange(cn_edges.shape[1])[None, None, :]

    for it in range(1, max_iter + 1):
        # ---------- check-node update (exclude-one) ----------
        V = mvc[:, cn_edges]                           # [Ba, M, dc]
        if rule == "minsum":
            sign = np.where(V >= 0, np.float32(1.0), np.float32(-1.0))
            sign = np.where(cn_mask, sign, np.float32(1.0))
            total_sign = np.prod(sign, axis=2, keepdims=True)      # [Ba,M,1]
            a = np.where(cn_mask, np.abs(V), np.float32(np.inf))   # [Ba,M,dc]
            min1 = a.min(axis=2, keepdims=True)
            first = np.argmax(a == min1, axis=2, keepdims=True)    # first argmin
            a2 = a.copy()
            np.put_along_axis(a2, first, np.float32(np.inf), axis=2)
            min2 = a2.min(axis=2, keepdims=True)
            excl_min = np.where(ar_dc == first, min2, min1)
            out = alpha * (total_sign * sign) * excl_min           # divide self sign
        elif rule == "spa":
            th = np.tanh(np.clip(np.where(cn_mask, V, 0.0), -llr_clip, llr_clip) / 2.0)
            th = np.where(cn_mask, th, 1.0)
            th = np.clip(th, -1 + 1e-6, 1 - 1e-6)
            prod = np.prod(th, axis=2, keepdims=True)
            out = 2.0 * np.arctanh(np.clip(prod / th, -1 + 1e-6, 1 - 1e-6))
        else:
            raise ValueError(f"unknown rule {rule!r}")

        Ba = act.size
        mcv = np.zeros((Ba, E), dtype=np.float32)
        mcv[:, flat_edges] = np.where(cn_mask, out, 0.0)[:, cn_mask]
        if fixed_point:
            mcv = quantize(mcv, quant_step, quant_clip)   # quantize check->var msgs
        else:
            np.clip(mcv, -llr_clip, llr_clip, out=mcv)

        # ---------- variable-node update + decision ----------
        sumC = np.zeros((Ba, N), dtype=np.float32)
        np.add.at(sumC.T, edge_vn, mcv.T)              # sumC[:, v] += its mcv
        total = La + sumC
        bits_a = (total < 0).astype(np.int8)
        bits[act] = bits_a

        if early_stop:
            ok = t.syndrome(bits_a).sum(axis=1) == 0
            if ok.any():
                done_rows = act[ok]
                converged[done_rows] = True
                iters[done_rows] = it
            keep = ~ok
            if not keep.any():
                break
            if not keep.all():                          # compact working arrays
                act = act[keep]; La = La[keep]
                sumC = sumC[keep]; mcv = mcv[keep]
                total = total[keep]

        # var->check messages for the next iteration (on surviving rows)
        mvc = La[:, edge_vn] + sumC[:, edge_vn] - mcv
        if fixed_point:
            mvc = quantize(mvc, quant_step, quant_clip)   # quantize var->check msgs
        else:
            np.clip(mvc, -llr_clip, llr_clip, out=mvc)

    return {"bits": bits, "converged": converged, "iters": iters}
