"""GPU (PyTorch/CUDA) batched min-sum decoder + Monte-Carlo / mixture-IS
estimators. Mirrors the NumPy reference but keeps the whole simulation (noise,
decoding, weights) on the GPU.

Speed features:
* fp16 decoder compute (values live on the 0.5 quantization grid, so half
  precision is exact for the fixed-point decoder) -- halves memory bandwidth;
* torch.compile (CUDA-graph / reduce-overhead) fuses the 15-iteration loop into
  a few launches instead of hundreds;
* statistics accumulated on-device, so there is no per-batch CPU sync.

Noise / weights stay in fp32; only the decoder runs in fp16.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .tanner import Tanner
from .estimate import Estimate


def _q(x: torch.Tensor, step: float, clip: float) -> torch.Tensor:
    return torch.clamp(torch.round(x / step) * step, -clip, clip)


class GpuDecoder:
    def __init__(self, t: Tanner, device: str = "cuda", dtype=torch.float16,
                 compile: bool = False):   # compile needs Triton (absent on Windows)
        self.t = t
        self.device = torch.device(device)
        self.dtype = dtype
        d = self.device
        self.M, self.N, self.E = t.M, t.N, t.E
        self.cn_mask = torch.as_tensor(t.cn_mask, device=d)                    # [M,dc] bool
        cn_edges = np.clip(t.cn_edges.astype(np.int64), 0, None)
        self.cn_edges = torch.as_tensor(cn_edges, device=d)                    # [M,dc]
        self.edge_vn = torch.as_tensor(t.edge_vn.astype(np.int64), device=d)   # [E]
        self.edge_cn = torch.as_tensor(t.edge_cn.astype(np.int64), device=d)   # [E]
        # edges are numbered sequentially per check -> cn_mask packs into 0..E-1
        self.flat_edges = torch.as_tensor(t.cn_edges[t.cn_mask].astype(np.int64), device=d)
        self.dc = self.cn_edges.shape[1]
        self._idx = torch.arange(self.dc, device=d).view(1, 1, self.dc)
        self._one = torch.ones((), device=d, dtype=dtype)
        self._inf = torch.tensor(float("inf"), device=d, dtype=dtype)
        self._core = self._core_impl
        if compile:
            try:
                self._core = torch.compile(self._core_impl, mode="reduce-overhead",
                                           dynamic=False)
            except Exception as ex:  # pragma: no cover
                print(f"# torch.compile unavailable ({ex}); running eager")
                self._core = self._core_impl

    def _core_impl(self, Lch, max_iter: int, alpha: float,
                   quant_step: float, quant_clip: float, early_stop: bool = True):
        """Iterate min-sum; return the final total (APP) LLR [B,N].

        early_stop: break once every frame in the batch has a zero syndrome
        (huge win at high SNR where mean iters << max_iter).
        """
        d, dt = self.device, self.dtype
        fixed = quant_step > 0
        clip = quant_clip if fixed else 30.0
        cn_mask, cn_edges, edge_vn = self.cn_mask, self.cn_edges, self.edge_vn
        edge_cn = self.edge_cn
        B = Lch.shape[0]
        if fixed:
            Lch = _q(Lch, quant_step, quant_clip)
        Lch_e = Lch[:, edge_vn]                                    # [B,E]
        mvc = Lch_e.clone()
        total = Lch
        # reused scratch (avoids per-iter cudaMalloc)
        sumC = torch.empty(B, self.N, device=d, dtype=dt)
        syn = torch.empty(B, self.M, device=d, dtype=torch.int32) if early_stop else None
        for it in range(max_iter):
            V = mvc[:, cn_edges]                                   # [B,M,dc]
            sign = torch.where(torch.where(cn_mask, V, self._one) >= 0, self._one, -self._one)
            total_sign = sign.prod(dim=2, keepdim=True)
            a = torch.where(cn_mask, V.abs(), self._inf)
            # two smallest magnitudes (avoids scatter+remin)
            vals, idxs = torch.topk(a, 2, dim=2, largest=False, sorted=True)
            min1, min2 = vals[..., :1], vals[..., 1:]
            arg1 = idxs[..., :1]
            excl_min = torch.where(self._idx == arg1, min2, min1)
            out = (alpha * (total_sign * sign) * excl_min).masked_fill(~cn_mask, 0)

            # cn_edges are sequential -> packed masked slots == edge order 0..E-1
            mcv = out[:, cn_mask]
            mcv = _q(mcv, quant_step, quant_clip) if fixed else mcv.clamp(-clip, clip)

            sumC.zero_()
            sumC.index_add_(1, edge_vn, mcv)
            total = Lch + sumC
            mvc = Lch_e + sumC[:, edge_vn] - mcv
            mvc = _q(mvc, quant_step, quant_clip) if fixed else mvc.clamp(-clip, clip)

            # Infrequent checks: each .item() syncs; at high SNR batches finish by ~5-6 iters.
            if early_stop and it in (3, 5, 7, 9, 11, 13):
                bits = (total < 0).to(torch.int32)
                syn.zero_()
                syn.index_add_(1, edge_cn, bits[:, edge_vn])
                if int((syn & 1).sum().item()) == 0:
                    break
        return total

    @torch.no_grad()
    def converged(self, Lch, max_iter=15, alpha=0.75, quant_step=None, quant_clip=7.5):
        """Return a bool[B] mask of frames whose syndrome is satisfied."""
        Lch = Lch.to(self.dtype)
        qs = quant_step if quant_step is not None else 0.0
        total = self._core(Lch, max_iter, alpha, qs, quant_clip, True)
        bits = (total < 0).to(torch.int32)
        syn = torch.zeros(bits.shape[0], self.M, device=self.device, dtype=torch.int32)
        syn.index_add_(1, self.edge_cn, bits[:, self.edge_vn])
        return (syn & 1).sum(dim=1) == 0


def _front_end(dec, y, sigma2, quant_step, quant_gain, punct, filler=None, clip=7.5,
               use_true_llr=False):
    """Build channel LLRs.

    use_true_llr=True  -> 2y/σ² (main_torch awgn_llrs), then core quantizes
    use_true_llr=False -> if quantized: AGC quant_gain·y; else 2y/σ²
    """
    if use_true_llr or quant_step is None:
        Lch = 2.0 * y / sigma2
    else:
        Lch = quant_gain * y
    if punct is not None:
        Lch[:, punct] = 0.0
    if filler is not None:
        Lch[:, filler] = clip if quant_step is not None else 30.0
    return Lch


def _nbatches(trials, batch):
    nb = max(1, math.ceil(trials / batch))
    return nb, nb * batch


@torch.no_grad()
def monte_carlo_gpu(dec: GpuDecoder, sigma, trials, batch=40000, max_iter=15,
                    target_errors=None, target_bit_errors=None,
                    quant_step=None, quant_clip=7.5,
                    quant_gain=2.0, punct=None, filler=None, seed=0, alpha=0.75):
    """GPU Monte-Carlo FER/BER.

    Bit errors: hard-decision errors on *transmitted* bits only (exclude
    punctured + filler), all-zero codeword => error iff APP LLR < 0.

    Frame errors (BLER): any transmitted bit wrong in the frame — same rule as
    ``main_decoder_run.py`` / ``main_896_ps.py`` (not syndrome-based).

    Stop rule (first hit wins): ``target_bit_errors`` TX bit errors,
    ``target_errors`` frame errors, or ``trials`` cap.
    """
    d = dec.device
    g = torch.Generator(device=d).manual_seed(seed)
    sigma2 = sigma * sigma
    nb, _ = _nbatches(trials, batch)
    fails = torch.zeros((), device=d, dtype=torch.long)
    biterr = torch.zeros((), device=d, dtype=torch.long)
    tx = torch.ones(dec.N, dtype=torch.bool, device=d)
    if punct is not None:
        tx[punct] = False
    if filler is not None:
        tx[filler] = False
    n_tx = int(tx.sum().item())
    done = 0
    for i in range(nb):
        y = 1.0 + sigma * torch.randn(batch, dec.N, device=d, generator=g, dtype=torch.float32)
        Lch = _front_end(dec, y, sigma2, quant_step, quant_gain, punct, filler, quant_clip)
        Lch = Lch.to(dec.dtype)
        qs = quant_step if quant_step is not None else 0.0
        total = dec._core(Lch, max_iter, alpha, qs, quant_clip, True)
        err = total < 0                                    # [B,N] vs all-zero
        # TX coded bit errors + BLER (any TX bit wrong)
        tx_err = err[:, tx]
        biterr += tx_err.sum()
        fails += tx_err.any(dim=1).sum()
        done += batch
        if target_bit_errors is not None or target_errors is not None:
            be = int(biterr.item()); fe = int(fails.item())
            if target_bit_errors is not None and be >= target_bit_errors:
                break
            if target_errors is not None and fe >= target_errors:
                break
    fails = int(fails.item()); be = int(biterr.item())
    fer = fails / done
    ber = be / (done * n_tx) if n_tx > 0 else 0.0
    stderr = np.sqrt(max(fer * (1 - fer), 0) / done)
    rel = stderr / fer if fer > 0 else float("inf")
    return Estimate(fer, stderr, rel, done, float(fails), ber=ber, n_bit_errors=float(be))


@torch.no_grad()
def mixture_is_gpu(dec: GpuDecoder, sets, sigma, trials, shift=1.6, batch=40000,
                   max_iter=15, quant_step=None, quant_clip=7.5, quant_gain=2.0,
                   punct=None, filler=None, seed=0, alpha=0.75, use_true_llr=False):
    """Mixture IS over the given sets for the total floor FER (absolute shift)."""
    d = dec.device
    g = torch.Generator(device=d).manual_seed(seed)
    sigma2 = sigma * sigma
    K = len(sets)
    A = torch.zeros(K, dec.N, device=d)
    for k, s in enumerate(sets):
        A[k, list(s.vns)] = 1.0
    if punct is not None:
        A[:, punct] = 0.0
    if filler is not None:
        A[:, filler] = 0.0
    a_size = A.sum(dim=1)
    amp = shift
    logK = math.log(K)

    wsum = torch.zeros((), device=d, dtype=torch.float64)
    w2sum = torch.zeros((), device=d, dtype=torch.float64)
    nb, actual = _nbatches(trials, batch)
    for _ in range(nb):
        ks = torch.randint(0, K, (batch,), device=d, generator=g)
        n = sigma * torch.randn(batch, dec.N, device=d, generator=g, dtype=torch.float32)
        n -= amp * A[ks]
        y = 1.0 + n
        Lch = _front_end(dec, y, sigma2, quant_step, quant_gain, punct, filler,
                         quant_clip, use_true_llr=use_true_llr)
        fail = (~dec.converged(Lch, max_iter, alpha, quant_step, quant_clip)).to(torch.float32)
        S = n @ A.T
        expo = (-amp * S) / sigma2 - a_size[None, :] * (amp * amp) / (2.0 * sigma2)
        w = torch.exp(logK - torch.logsumexp(expo, dim=1)) * fail
        wsum += w.double().sum()
        w2sum += (w.double() * w.double()).sum()

    wsum = float(wsum.item()); w2sum = float(w2sum.item())
    fer = wsum / actual
    var = max(w2sum / actual - fer * fer, 0.0) / actual
    stderr = np.sqrt(var)
    rel = stderr / fer if fer > 0 else float("inf")
    eff = (wsum * wsum / w2sum) if w2sum > 0 else 0.0
    return Estimate(fer, stderr, rel, actual, eff)
