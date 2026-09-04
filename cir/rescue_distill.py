#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rescue → base distillation.

Train NeuralMSDecoder weights so a single decode imitates TS-Rescue outputs
(especially on frames where base fails but Rescue clears the syndrome).

Saves:
  model5_rescue_distill_weights.npz
  matlab_plot_data/weights_RescueDistill_*.csv
  rescue_distill_log.csv
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── soft_diff setattr patch (same as main_torch launcher) ────────────────────
_orig_setattr = nn.Module.__setattr__


def _setattr(self, name, value):  # type: ignore[no-untyped-def]
    if name in ("w_vn", "w_cn"):
        is_tensor = torch.is_tensor(value)
        is_param = isinstance(value, nn.Parameter)
        if is_tensor and not is_param:
            object.__setattr__(self, name, value)
            return
        if is_param or value is None:
            d = self.__dict__
            if name in d:
                del d[name]
    return _orig_setattr(self, name, value)


nn.Module.__setattr__ = _setattr  # type: ignore[method-assign]

# ───────────────────────────── paths / config ────────────────────────────────
HERE = Path(__file__).resolve().parent
STEM = "5G_LDPC_R0.33_n_dec896_n768_k256_z32_s257_320"
OUTDIR = HERE / "outputs_nnms_torch" / f"{STEM}__graph__track_nnms"
H_PATH = HERE / "pcm" / "896" / f"{STEM}.graph"
BASE_W = OUTDIR / "model5_rlnms_weights.npz"
HARVEST = OUTDIR / "harvest_pool_TS-Rescue.npz"
TRAP = Path(
    "/home/mohammadrezamaleki/ts_enum/trapping-sets-enumeration-master"
    "/out_kb_basegraph/5G896_enum_absorbing.trap"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ITER = 15
GAMMA = 0.05
BETA = 2.5
QMS_ENABLE = True
QMS_MAX = 7.5
QMS_STEP = 0.5
N_PUNCTURED = 64
SHORT_LO, SHORT_HI = 256, 320  # s257_320 → 0-based [256:320)
N_SHORTENED = SHORT_HI - SHORT_LO
SHORTENED_LLR = 50.0
Z_LIFT = 32
RESCUE_CANDIDATES = 4
RESCUE_MAX_POOL = 8000

STEPS = int(os.environ.get("DISTILL_STEPS", "600"))
BATCH = int(os.environ.get("DISTILL_BATCH", "768"))
LR = float(os.environ.get("DISTILL_LR", "5e-3"))
EBNO_TRAIN = (3.5, 4.0, 4.5, 5.0)
TS_MIX_P = 0.70
W_REG = float(os.environ.get("DISTILL_W_REG", "0.0"))  # no pull-to-1; was collapsing σ
FOCUS_REPAIRED_ONLY = True  # train mainly on Rescue repairs
SEED = 0
MIN_REPAIRED = 16
POST_MSE_W = 0.05


# ───────────────────────────── utilities ─────────────────────────────────────
def load_H(path: Path) -> np.ndarray:
    with open(path) as f:
        hdr = f.readline().split()
        n, m = int(hdr[0]), int(hdr[1])
        H0 = np.zeros((m, n), dtype=np.int8)
        for i in range(m):
            for s in f.readline().split():
                H0[i, int(s) - 1] = 1
    return H0


def quantize_msg(x: torch.Tensor) -> torch.Tensor:
    if not QMS_ENABLE:
        return x
    y = torch.clamp(torch.round(x / QMS_STEP) * QMS_STEP, -QMS_MAX, QMS_MAX)
    # straight-through estimator so Adam can update weights under Q-NMS
    return x + (y - x).detach()


def apply_puncture(llr: torch.Tensor) -> torch.Tensor:
    if N_PUNCTURED:
        llr = llr.clone()
        llr[:, :N_PUNCTURED] = 0.0
    else:
        llr = llr.clone()
    if N_SHORTENED:
        llr[:, SHORT_LO:SHORT_HI] = float(QMS_MAX) if QMS_ENABLE else SHORTENED_LLR
    if QMS_ENABLE:
        llr = quantize_msg(llr)
    return llr


def awgn_sigma(ebno_db: float, rate_tx: float) -> float:
    # σ² = 1 / (2 R Eb/N0_lin)
    snr_lin = 10.0 ** (ebno_db / 10.0)
    return float(math.sqrt(1.0 / (2.0 * rate_tx * snr_lin)))


def qc_shift_support(sup: np.ndarray, z_lift: int, t: int) -> np.ndarray:
    return np.sort((sup // z_lift) * z_lift + ((sup % z_lift) + t) % z_lift)


# ───────────────────────────── decoder ───────────────────────────────────────
class NeuralMSDecoder(nn.Module):
    """Soft-CN trainable min-sum (matches main_torch soft training path)."""

    def __init__(self, H: np.ndarray, num_iter: int = NUM_ITER,
                 gamma: float = GAMMA, beta: float = BETA,
                 shared_weights: bool = False):
        super().__init__()
        self.num_iter = int(num_iter)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.quantize = bool(QMS_ENABLE)
        self.quantize_at_multiply = False
        self.shared_weights = bool(shared_weights)
        self.cn_update_mode = "soft"
        self.num_cns, self.num_vns = int(H.shape[0]), int(H.shape[1])

        cn_np, vn_np = np.nonzero(H)
        order = np.argsort(vn_np)
        cn_np = cn_np[order].astype(np.int64)
        vn_np = vn_np[order].astype(np.int64)
        self.num_edges = len(cn_np)
        self.register_buffer("cn_idx", torch.from_numpy(cn_np))
        self.register_buffer("vn_idx", torch.from_numpy(vn_np))

        if self.shared_weights:
            self.w_vn = nn.Parameter(torch.ones(self.num_edges))
            self.w_cn = nn.Parameter(torch.ones(self.num_edges))
        else:
            self.w_vn = nn.Parameter(torch.ones(self.num_iter, self.num_edges))
            self.w_cn = nn.Parameter(torch.ones(self.num_iter, self.num_edges))

    def _iter_w_vn(self, l: int) -> torch.Tensor:
        return self.w_vn if self.shared_weights else self.w_vn[l]

    def _iter_w_cn(self, l: int) -> torch.Tensor:
        return self.w_cn if self.shared_weights else self.w_cn[l]

    def _q_msg(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_msg(x) if self.quantize else x

    def _seg_sum(self, vals: torch.Tensor, idx: torch.Tensor, num: int) -> torch.Tensor:
        B = vals.shape[0]
        out = vals.new_zeros(B, num)
        return out.index_add_(1, idx, vals)

    def _seg_min(self, vals: torch.Tensor, idx: torch.Tensor, num: int,
                 big: float) -> torch.Tensor:
        B = vals.shape[0]
        out = vals.new_full((B, num), big)
        return out.scatter_reduce_(1, idx.unsqueeze(0).expand(B, -1), vals,
                                   reduce="amin", include_self=True)

    def _cn_magnitude(self, abs_v2c: torch.Tensor, need_grad: bool,
                      soft_train: bool, BIG: float) -> torch.Tensor:
        mm = self.num_cns
        if need_grad:
            exp_neg = torch.exp(-self.beta * abs_v2c)
            sum_exp_cn = self._seg_sum(exp_neg, self.cn_idx, mm)
            excl_exp = sum_exp_cn[:, self.cn_idx] - exp_neg + 1e-10
            raw_sm = -(1.0 / self.beta) * torch.log(excl_exp)
            soft_min = raw_sm + (torch.relu(-raw_sm)).detach()
        if soft_train:
            return soft_min
        absv_h = abs_v2c.detach()
        min1 = self._seg_min(absv_h, self.cn_idx, mm, BIG)
        min1_at_e = min1[:, self.cn_idx]
        is_min1 = absv_h == min1_at_e
        min1_cnt = self._seg_sum(is_min1.float(), self.cn_idx, mm)
        unique_min1 = is_min1 & (min1_cnt[:, self.cn_idx] < 1.5)
        absv_mask = torch.where(is_min1, absv_h.new_full((), BIG), absv_h)
        min2 = self._seg_min(absv_mask, self.cn_idx, mm, BIG)
        min2_at_e = min2[:, self.cn_idx]
        hard_min = torch.where(unique_min1, min2_at_e, min1_at_e)
        hard_min = torch.clamp(hard_min, max=BIG * 0.5)
        if need_grad:
            return soft_min + (hard_min - soft_min).detach()
        return hard_min

    def forward(self, llr_ch: torch.Tensor, return_all_posteriors: bool = False):
        B = llr_ch.shape[0]
        mm, nn_ = self.num_cns, self.num_vns
        if self.quantize:
            llr_ch = quantize_msg(llr_ch)
        c2v = llr_ch.new_zeros(B, self.num_edges)
        posts = [] if return_all_posteriors else None
        BIG = 1e10
        need_grad = torch.is_grad_enabled() and (
            llr_ch.requires_grad or self.w_vn.requires_grad or self.w_cn.requires_grad)

        for l in range(self.num_iter):
            w_v = self._iter_w_vn(l)
            w_c = self._iter_w_cn(l)
            agg_c2v = self._seg_sum(c2v, self.vn_idx, nn_)
            v2c_raw = agg_c2v[:, self.vn_idx] - c2v + llr_ch[:, self.vn_idx]
            v2c = v2c_raw * w_v
            if self.quantize:
                v2c = self._q_msg(v2c)

            soft_train = (need_grad and self.cn_update_mode == "soft")
            if soft_train:
                ss = torch.tanh(self.gamma * v2c)
                neg_flag = (ss < 0.0).float()
            else:
                neg_flag = (v2c < 0.0).float()
            neg_cnt_cn = self._seg_sum(neg_flag, self.cn_idx, mm)
            neg_excl = neg_cnt_cn[:, self.cn_idx] - neg_flag
            excl_sign = 1.0 - 2.0 * torch.remainder(torch.round(neg_excl), 2.0)
            mag = self._cn_magnitude(torch.abs(v2c), need_grad, soft_train, BIG)
            c2v = excl_sign * mag * w_c
            if self.quantize:
                c2v = self._q_msg(c2v)
            if return_all_posteriors:
                posts.append(llr_ch + self._seg_sum(c2v, self.vn_idx, nn_))

        if return_all_posteriors:
            return torch.stack(posts, dim=0)
        return llr_ch + self._seg_sum(c2v, self.vn_idx, nn_)


class TSRescueDecoder(nn.Module):
    """Two-phase base + matched-filter TS erasure rescue (from main_torch)."""

    def __init__(self, base: NeuralMSDecoder, ts_pool: list, z_lift: int,
                 H: np.ndarray, num_candidates: int = RESCUE_CANDIDATES,
                 max_pool: int = RESCUE_MAX_POOL):
        super().__init__()
        self.base = base
        self.num_candidates = int(num_candidates)
        m = H.shape[0]
        H_np = np.asarray(H)
        sups: list[np.ndarray] = []
        seen: set = set()
        for _, b_ts, sup in ts_pool:
            if b_ts < 1:
                continue
            sup = np.asarray(sup, dtype=np.int64)
            for t in range(max(1, int(z_lift))):
                s = qc_shift_support(sup, z_lift, t) if z_lift > 1 else sup
                key = s.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                sups.append(s)
                if len(sups) >= max_pool:
                    break
            if len(sups) >= max_pool:
                break
        K = len(sups)
        max_a = max((len(s) for s in sups), default=1)
        sup_pad = np.full((K, max_a), -1, dtype=np.int64)
        odd_sig = np.zeros((K, m), dtype=np.float32)
        for i, s in enumerate(sups):
            sup_pad[i, : len(s)] = s
            odd = (H_np[:, s].sum(axis=1) % 2).astype(np.float32)
            odd_sig[i] = odd / max(odd.sum(), 1.0)
        self.register_buffer("sup_pad", torch.from_numpy(sup_pad))
        self.register_buffer("odd_sig_t", torch.from_numpy(odd_sig.T))
        self.pool_size = K
        self._sups_np = [s for s in (sup_pad[i][sup_pad[i] >= 0] for i in range(K))]
        print(f"  [TS-Rescue] matcher pool: {K} instances "
              f"(z={z_lift}, max_a={max_a}, candidates={self.num_candidates})")

    def _unsat(self, post: torch.Tensor) -> torch.Tensor:
        hard = (post < 0.0).float()
        hard_at_e = hard[:, self.base.vn_idx]
        syn = self.base._seg_sum(hard_at_e, self.base.cn_idx, self.base.num_cns)
        return torch.remainder(torch.round(syn), 2.0) > 0.5

    def forward(self, llr_ch: torch.Tensor):
        post = self.base(llr_ch)
        if self.pool_size == 0:
            return post
        unsat = self._unsat(post)
        fail = unsat.any(dim=1)
        rows = fail.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return post
        post_out = post.clone()
        scores = unsat[rows].float() @ self.odd_sig_t
        R = min(self.num_candidates, self.pool_size)
        top = scores.topk(R, dim=1).indices
        act_rows = torch.arange(rows.numel(), device=llr_ch.device)
        for r in range(R):
            if rows.numel() == 0:
                break
            cand = top[act_rows, r]
            sup_r = self.sup_pad[cand]
            valid = (sup_r >= 0)
            llr_try = llr_ch[rows].clone()
            row_ix = torch.arange(rows.numel(), device=llr_ch.device).unsqueeze(1).expand_as(sup_r)
            llr_try[row_ix[valid], sup_r[valid]] = 0.0
            post_try = self.base(llr_try)
            fixed = ~self._unsat(post_try).any(dim=1)
            if fixed.any():
                post_out[rows[fixed]] = post_try[fixed]
            rows = rows[~fixed]
            act_rows = act_rows[~fixed]
        return post_out


# ───────────────────────────── data helpers ──────────────────────────────────
def load_weights(model: NeuralMSDecoder, path: Path) -> None:
    z = np.load(path)
    w_vn = np.asarray(z["w_vn"], dtype=np.float32)
    w_cn = np.asarray(z["w_cn"], dtype=np.float32)
    with torch.no_grad():
        if model.shared_weights:
            if w_vn.ndim == 2:
                w_vn = w_vn.mean(axis=0)
                w_cn = w_cn.mean(axis=0)
            model.w_vn.copy_(torch.from_numpy(w_vn).to(DEVICE))
            model.w_cn.copy_(torch.from_numpy(w_cn).to(DEVICE))
        else:
            if w_vn.ndim == 1:
                w_vn = np.broadcast_to(w_vn, (model.num_iter, w_vn.shape[0])).copy()
                w_cn = np.broadcast_to(w_cn, (model.num_iter, w_cn.shape[0])).copy()
            # align L
            L = model.num_iter
            if w_vn.shape[0] != L:
                if w_vn.shape[0] == 1:
                    w_vn = np.repeat(w_vn, L, axis=0)
                    w_cn = np.repeat(w_cn, L, axis=0)
                else:
                    w_vn = w_vn[:L] if w_vn.shape[0] > L else np.pad(
                        w_vn, ((0, L - w_vn.shape[0]), (0, 0)), mode="edge")
                    w_cn = w_cn[:L] if w_cn.shape[0] > L else np.pad(
                        w_cn, ((0, L - w_cn.shape[0]), (0, 0)), mode="edge")
            model.w_vn.copy_(torch.from_numpy(w_vn).to(DEVICE))
            model.w_cn.copy_(torch.from_numpy(w_cn).to(DEVICE))
    print(f"  loaded weights {path.name}: "
          f"w_vn μ={float(model.w_vn.mean()):.4f} σ={float(model.w_vn.std()):.4f}  "
          f"w_cn μ={float(model.w_cn.mean()):.4f} σ={float(model.w_cn.std()):.4f}")


def save_weights(model: NeuralMSDecoder, path: Path, ref: Path | None = None) -> None:
    kw = {
        "w_vn": model.w_vn.detach().cpu().numpy().astype(np.float32),
        "w_cn": model.w_cn.detach().cpu().numpy().astype(np.float32),
        "cn_idx": model.cn_idx.detach().cpu().numpy(),
        "vn_idx": model.vn_idx.detach().cpu().numpy(),
        "num_iter": np.int32(model.num_iter),
        "gamma": np.float32(model.gamma),
        "beta": np.float32(model.beta),
    }
    if ref and ref.is_file():
        z = np.load(ref)
        for k in z.files:
            if k not in kw:
                kw[k] = z[k]
    np.savez_compressed(path, **kw)
    print(f"  saved -> {path}")


def build_ts_pool(H: np.ndarray) -> list:
    """Prefer harvest pool supports; fall back to .trap file."""
    pool: list = []
    if HARVEST.is_file():
        z = np.load(HARVEST)
        sup = z["sup"]
        ab = z["ab"]
        for i in range(sup.shape[0]):
            s = sup[i]
            s = s[s >= 0].astype(np.int64)
            if s.size == 0:
                continue
            a = int(ab[i, 0]) if ab is not None else int(s.size)
            b = int(ab[i, 1]) if ab is not None else 1
            pool.append((a, b, s))
        print(f"  TS pool from harvest: {len(pool)}")
        return pool
    if TRAP.is_file():
        # simple .trap parser: lines "a b v1 v2 ..." (1-based VNs)
        with open(TRAP) as f:
            for line in f:
                toks = line.split()
                if len(toks) < 3:
                    continue
                try:
                    a, b = int(toks[0]), int(toks[1])
                    s = np.array([int(x) - 1 for x in toks[2:]], dtype=np.int64)
                except ValueError:
                    continue
                if s.size and s.min() >= 0:
                    pool.append((a, b, s))
        print(f"  TS pool from trap: {len(pool)}")
        return pool
    raise SystemExit("no TS pool found")


def make_llr_batch(B: int, n: int, rate_tx: float, active_idx: torch.Tensor,
                   ts_sups: list[np.ndarray], rng: np.random.Generator) -> torch.Tensor:
    """All-zero codeword AWGN (+ optional TS bias)."""
    ebno = float(rng.choice(EBNO_TRAIN))
    sigma = awgn_sigma(ebno, rate_tx)
    # BPSK +1 for bit0; y = 1 + sigma*N(0,1)
    noise = rng.standard_normal((B, n)).astype(np.float32)
    y = 1.0 + sigma * noise
    # TS-centered bias on a random subset
    n_ts = int(round(B * TS_MIX_P))
    if n_ts and ts_sups:
        alphas = rng.uniform(1.5, 4.0, size=n_ts).astype(np.float32)
        for i in range(n_ts):
            s = ts_sups[int(rng.integers(0, len(ts_sups)))]
            y[i, s] -= alphas[i]
    llr = (2.0 / (sigma * sigma)) * y
    return apply_puncture(torch.from_numpy(llr).to(DEVICE))


# ───────────────────────────── main ──────────────────────────────────────────
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    assert H_PATH.is_file(), H_PATH
    assert BASE_W.is_file(), BASE_W
    OUTDIR.mkdir(parents=True, exist_ok=True)

    H = load_H(H_PATH)
    m, n = H.shape
    k = n - m
    n_tx = n - N_PUNCTURED - N_SHORTENED
    k_tx = k - N_SHORTENED
    rate_tx = k_tx / n_tx
    print(f"H={H.shape}  n_tx={n_tx} k_tx={k_tx} R_tx={rate_tx:.4f}  device={DEVICE}")

    act_mask = np.ones(n, dtype=bool)
    act_mask[:N_PUNCTURED] = False
    act_mask[SHORT_LO:SHORT_HI] = False
    active_idx = torch.as_tensor(np.flatnonzero(act_mask), dtype=torch.long, device=DEVICE)

    pool = build_ts_pool(H)

    # Teacher base (frozen RL weights) + Rescue
    teacher_base = NeuralMSDecoder(H, shared_weights=False).to(DEVICE)
    load_weights(teacher_base, BASE_W)
    teacher_base.eval()
    for p in teacher_base.parameters():
        p.requires_grad_(False)

    rescue = TSRescueDecoder(teacher_base, pool, Z_LIFT, H).to(DEVICE)
    rescue.eval()

    # Student starts from same RL weights
    student = NeuralMSDecoder(H, shared_weights=False).to(DEVICE)
    load_weights(student, BASE_W)
    student.train()
    # keep soft CN for training grads
    student.cn_update_mode = "soft"

    opt = torch.optim.Adam(student.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(STEPS, 1))

    # compact TS list for biasing (unique supports from harvest, already expanded)
    ts_sups = []
    seen = set()
    for a, b, s in pool:
        key = np.asarray(s).tobytes()
        if key in seen:
            continue
        seen.add(key)
        ts_sups.append(np.asarray(s, dtype=np.int64))
        if len(ts_sups) >= 3000:
            break
    print(f"  bias supports: {len(ts_sups)}")

    w0_vn = student.w_vn.detach().clone()
    w0_cn = student.w_cn.detach().clone()

    log_path = OUTDIR / "rescue_distill_log.csv"
    rows = []
    t0 = time.time()
    print(f"\n=== Rescue distillation: steps={STEPS} batch={BATCH} lr={LR} ===")

    for step in range(STEPS):
        llr = make_llr_batch(BATCH, n, rate_tx, active_idx, ts_sups, rng)

        with torch.no_grad():
            teacher_base.eval()
            base_post = teacher_base(llr)
            base_unsat = rescue._unsat(base_post)
            base_fail_f = base_unsat.any(dim=1)
            teach_post = rescue(llr)
            teach_ok = ~rescue._unsat(teach_post).any(dim=1)
            repaired = base_fail_f & teach_ok
            # hard bit targets from Rescue (bit1 when post<0)
            tgt_hard = (teach_post < 0.0).float()

        n_rep = int(repaired.sum().item())
        if FOCUS_REPAIRED_ONLY and n_rep < MIN_REPAIRED:
            # not enough repair examples this step — resample / skip update
            continue

        student.train()
        student.cn_update_mode = "soft"
        stud_post = student(llr)
        logits = -stud_post

        if FOCUS_REPAIRED_ONLY:
            sel = repaired
        else:
            sel = teach_ok  # all frames Rescue thinks are correct

        # hard BCE on selected frames / active bits
        per = F.binary_cross_entropy_with_logits(
            logits[sel][:, active_idx],
            tgt_hard[sel][:, active_idx],
            reduction="none")
        loss_bce = per.mean()
        # match Rescue posterior LLRs on repaired frames (scaled)
        loss_mse = POST_MSE_W * F.mse_loss(
            stud_post[sel][:, active_idx],
            teach_post[sel][:, active_idx])
        loss_reg = W_REG * ((student.w_vn - w0_vn).pow(2).mean()
                            + (student.w_cn - w0_cn).pow(2).mean())
        loss = loss_bce + loss_mse + loss_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 25 == 0 or step == STEPS - 1:
            with torch.no_grad():
                student.eval()
                student.cn_update_mode = "soft"  # eval still soft ok; for hard use soft=False path
                # hard min-sum eval: temporarily disable soft
                student.cn_update_mode = "hard"
                sp = student(llr)
                s_fail = rescue._unsat(sp).any(dim=1).float().mean().item()
                t_fail = (~teach_ok).float().mean().item()
                b_fail = base_fail_f.float().mean().item()
                n_rep = int(repaired.sum().item())
                dw = float((student.w_vn - w0_vn).abs().mean().item())
                student.cn_update_mode = "soft"
                student.train()
            msg = (f"  step {step:4d}  loss={float(loss):.5f}  bce={float(loss_bce):.5f}  "
                   f"mse={float(loss_mse):.5f}  "
                   f"FER~ base={b_fail:.3f} stud={s_fail:.3f} rescue={t_fail:.3f}  "
                   f"repaired={n_rep}/{BATCH}  |Δw|={dw:.4f}  "
                   f"w_vn[μ={float(student.w_vn.mean()):.3f},σ={float(student.w_vn.std()):.3f}]")
            print(msg)
            rows.append({
                "step": step, "loss": float(loss), "bce": float(loss_bce),
                "mse": float(loss_mse),
                "fer_base": b_fail, "fer_stud": s_fail, "fer_rescue": t_fail,
                "n_repaired": n_rep, "dw": dw,
                "w_vn_mean": float(student.w_vn.mean()),
                "w_vn_std": float(student.w_vn.std()),
                "w_cn_mean": float(student.w_cn.mean()),
                "w_cn_std": float(student.w_cn.std()),
            })

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    out_w = OUTDIR / "model5_rescue_distill_weights.npz"
    save_weights(student, out_w, ref=BASE_W)

    # export CSV for MATLAB histograms
    plot_dir = OUTDIR / "matlab_plot_data"
    plot_dir.mkdir(exist_ok=True)
    for key, arr in (("w_vn", student.w_vn), ("w_cn", student.w_cn)):
        flat = arr.detach().cpu().numpy().ravel()
        np.savetxt(plot_dir / f"weights_RescueDistill_{key}.csv", flat,
                   delimiter=",", header=key, comments="")
    # also dump before/after summary
    summary = OUTDIR / "rescue_distill_weight_summary.txt"
    with open(summary, "w") as f:
        f.write("Rescue distillation weight summary\n")
        f.write(f"base: {BASE_W.name}\n")
        f.write(f"out:  {out_w.name}\n")
        f.write(f"steps={STEPS} batch={BATCH} lr={LR} ts_mix_p={TS_MIX_P}\n\n")
        for name, a0, a1 in (
            ("w_vn", w0_vn, student.w_vn.detach()),
            ("w_cn", w0_cn, student.w_cn.detach()),
        ):
            d = (a1 - a0).abs()
            f.write(f"{name}:\n")
            f.write(f"  before  μ={float(a0.mean()):.6f} σ={float(a0.std()):.6f} "
                    f"[{float(a0.min()):.4f},{float(a0.max()):.4f}]\n")
            f.write(f"  after   μ={float(a1.mean()):.6f} σ={float(a1.std()):.6f} "
                    f"[{float(a1.min()):.4f},{float(a1.max()):.4f}]\n")
            f.write(f"  |Δ|     μ={float(d.mean()):.6f} max={float(d.max()):.6f}\n\n")
    print(f"  summary -> {summary}")

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  log -> {log_path}")

    # quick held-out FER compare @ 4.5 dB
    print("\n=== quick FER check @ 4.5 dB (8192 frames, TS-mixed) ===")
    with torch.no_grad():
        student.eval()
        student.cn_update_mode = "hard"
        teacher_base.eval()
        B_eval = 2048
        n_batches = 4
        stats = {"base": 0.0, "stud": 0.0, "rescue": 0.0, "n": 0}
        for _ in range(n_batches):
            llr = make_llr_batch(B_eval, n, rate_tx, active_idx, ts_sups, rng)
            # force mid SNR-ish by regenerating at fixed 4.5 — override
            sigma = awgn_sigma(4.5, rate_tx)
            noise = rng.standard_normal((B_eval, n)).astype(np.float32)
            y = 1.0 + sigma * noise
            n_ts = int(round(B_eval * 0.7))
            for i in range(n_ts):
                s = ts_sups[int(rng.integers(0, len(ts_sups)))]
                y[i, s] -= float(rng.uniform(2.0, 4.5))
            llr = apply_puncture(torch.from_numpy((2.0 / (sigma * sigma)) * y).to(DEVICE))
            bp = teacher_base(llr)
            sp = student(llr)
            rp = rescue(llr)
            stats["base"] += float(rescue._unsat(bp).any(dim=1).float().sum())
            stats["stud"] += float(rescue._unsat(sp).any(dim=1).float().sum())
            stats["rescue"] += float(rescue._unsat(rp).any(dim=1).float().sum())
            stats["n"] += B_eval
        for k in ("base", "stud", "rescue"):
            print(f"  {k:7s} FER ≈ {stats[k]/stats['n']:.4f}  ({int(stats[k])}/{stats['n']})")


if __name__ == "__main__":
    main()
