"""Make training LEARN BEYOND NNMS: fine-tune from the NNMS weights on the
RESIDUAL failures NNMS still makes at the deployment clip +/-7.5, with an
adaptive hard curriculum + focal loss so the signal never saturates.

Why the earlier RL/SAC/Adv froze: the reward/loss was measured on trapping-set
shifts that NNMS already escapes -> reward ~ 0.9997, gradient ~ 0. Fixes:
  * train + mine failures at +/-7.5 (where the floor actually is), not +/-20;
  * WIDE shift range (1.0..4.0) so many frames genuinely fail the current model;
  * FOCAL weighting: each frame weighted by its own failure hardness (detached),
    so learning tracks the residual as the model improves;
  * anchor the weights to NNMS (reg toward NNMS, not toward 1) so we build on it.
Also re-runs RL(ES) with the same hard shifts to show the reward un-saturates.
"""
from __future__ import annotations
import importlib.util, math, re, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent
RD = ROOT / "rescue_distill.py"
GRAPH = ROOT / "BaseGraph/5G_LDPC_R0.50_n_dec1280_n1024_k512_z64_s513_640.graph"
TRAP  = ROOT / "BaseGraph/5G_LDPC_R0.50_n_dec1280_n1024_k512_z64_s513_640.found.trap"
ODIR  = ROOT / "weights"
NNMS_NPZ = ODIR / "model5_nnms_train20_eval75.npz"

GAMMA, BETA = 0.05, 5.0
EVAL_CLIP = 7.5
TRAIN_EBNO = (3.0, 4.5)            # floor-onset region at +/-7.5
SHIFT_RANGE = (1.0, 4.0)          # wide -> mix of easy+hard frames
TS_FRAC = 0.7
FOCAL = 2.0                       # focal exponent on per-frame hardness
STEPS, BATCH, LR = 600, 384, 1e-3
W_ANCHOR, W_LO, W_HI, GCLIP = 3e-3, 0.4, 1.8, 1.0
VAL_EVERY, VAL_SNR, VAL_TRIALS = 40, 4.5, 120_000
IS_SNRS, IS_TRIALS = (4.0, 4.5, 5.0, 5.5, 6.0), 200_000

spec = importlib.util.spec_from_file_location("rescue_distill", RD)
rd = importlib.util.module_from_spec(spec); sys.modules["rescue_distill"] = rd
spec.loader.exec_module(rd)
rd.N_PUNCTURED = 128; rd.SHORT_LO, rd.SHORT_HI = 512, 640; rd.N_SHORTENED = 128; rd.Z_LIFT = 64
rd.QMS_MAX = EVAL_CLIP
DEV = rd.DEVICE
H = rd.load_H(GRAPH); M, N = H.shape
RATE = 512 / (N - 256)
tx = np.ones(N, bool); tx[:128] = False; tx[512:640] = False
tx_t = torch.as_tensor(tx, device=DEV); bad = set(np.nonzero(~tx)[0].tolist())

def load_sets(a_max=None):
    pat = re.compile(r"\((\d+),\s*(\d+)\)\s*(.*)"); out = []
    for ln in TRAP.read_text().splitlines():
        m = pat.match(ln.strip())
        if not m: continue
        a, b = int(m.group(1)), int(m.group(2)); v = [int(x)-1 for x in m.group(3).split()]
        if b <= 2 and 3 <= a <= 10 and not any(x in bad for x in v) and (a_max is None or a <= a_max):
            out.append(v)
    return out
is_sets, focus = load_sets(), load_sets(6)
K, Kf = len(is_sets), len(focus)
A = torch.zeros(K, N, device=DEV)
for k, v in enumerate(is_sets): A[k, v] = 1.0
a_size = A.sum(1)
Af = torch.zeros(Kf, N, device=DEV)
for k, v in enumerate(focus): Af[k, v] = 1.0
ts_bit = (Af.sum(0) > 0).float(); sup_w = 1.0 + 1.0*ts_bit
it_w = torch.linspace(0.3, 1.0, 15, device=DEV); it_w = it_w / it_w.sum()

def new_model(mode):
    m = rd.NeuralMSDecoder(H, num_iter=15, gamma=GAMMA, beta=BETA, shared_weights=False).to(DEV)
    m.cn_update_mode = mode; return m

def nnms_base(model):
    z = np.load(NNMS_NPZ); Mc = model.num_cns
    zkey = z["vn_idx"].astype(np.int64)*Mc + z["cn_idx"].astype(np.int64)
    mkey = model.vn_idx.cpu().numpy().astype(np.int64)*Mc + model.cn_idx.cpu().numpy().astype(np.int64)
    oz = np.argsort(zkey); pos = oz[np.searchsorted(zkey, mkey, sorter=oz)]
    assert np.array_equal(zkey[pos], mkey)
    return (torch.from_numpy(np.ascontiguousarray(z["w_vn"][:, pos])).to(DEV),
            torch.from_numpy(np.ascontiguousarray(z["w_cn"][:, pos])).to(DEV))

@torch.no_grad()
def is_fer(model, ebno, trials, shift=1.2, batch=6144, seed=222):
    g = torch.Generator(device=DEV).manual_seed(seed)
    sig = rd.awgn_sigma(ebno, RATE); s2 = sig*sig; amp = shift; logK = math.log(K)
    nb = max(1, math.ceil(trials/batch)); actual = nb*batch
    wsum = torch.zeros((), device=DEV, dtype=torch.float64); w2 = torch.zeros((), device=DEV, dtype=torch.float64)
    for _ in range(nb):
        ks = torch.randint(0, K, (batch,), device=DEV, generator=g)
        nz = sig*torch.randn(batch, N, device=DEV, generator=g) - amp*A[ks]
        llr = rd.apply_puncture(2.0*(1.0+nz)/s2)
        fail = (model(llr) < 0.0)[:, tx_t].any(1).float()
        expo = (-amp*(nz@A.T))/s2 - a_size[None,:]*(amp*amp)/(2.0*s2)
        w = torch.exp(logK - torch.logsumexp(expo,1))*fail
        wsum += w.double().sum(); w2 += (w.double()**2).sum()
    fer = float(wsum)/actual; ess=(float(wsum)**2/float(w2))/actual if float(w2)>0 else 0
    return fer, ess

# ---------- focal hard-example gradient fine-tune from NNMS ------------------
ms = new_model("hard"); ms.eval()
for p in ms.parameters(): p.requires_grad_(False)
ms_val, _ = is_fer(ms, VAL_SNR, VAL_TRIALS)
model = new_model("soft")
b_vn, b_cn = nnms_base(model)
with torch.no_grad(): model.w_vn.copy_(b_vn); model.w_cn.copy_(b_cn)
nnms_val, _ = is_fer(model, VAL_SNR, VAL_TRIALS)      # NNMS starting point @7.5
print(f"# beat-NNMS | MS@7.5={ms_val:.3e}  NNMS@7.5={nnms_val:.3e}  (target: < NNMS)", flush=True)
model.cn_update_mode = "soft"; model.train()
opt = torch.optim.Adam([model.w_vn, model.w_cn], lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
g = torch.Generator(device=DEV).manual_seed(1)
s_lo, s_hi = SHIFT_RANGE; nsh = int(BATCH*TS_FRAC)
best_f, best_vn, best_cn, best_step = nnms_val, model.w_vn.detach().clone(), model.w_cn.detach().clone(), -1
t0 = time.time(); hard_frac_ema = 0.0
for step in range(STEPS):
    model.cn_update_mode = "soft"; model.train()
    ebno = float(torch.empty(1).uniform_(*TRAIN_EBNO).item())
    sig = rd.awgn_sigma(ebno, RATE); s2 = sig*sig
    y = 1.0 + sig*torch.randn(BATCH, N, device=DEV, generator=g)
    if nsh:
        ks = torch.randint(0, Kf, (nsh,), device=DEV, generator=g)
        sh = torch.empty(nsh, 1, device=DEV).uniform_(s_lo, s_hi)
        y[-nsh:] = y[-nsh:] - sh*Af[ks]
    llr = rd.apply_puncture(2.0*y/s2)
    posts = model(llr, return_all_posteriors=True)         # [L,B,N]
    final = posts[-1]
    # per-frame hardness on tx bits (worst/most-negative margin), detached -> focal weight
    with torch.no_grad():
        worst = (-final[:, tx_t]).max(dim=1).values          # >0 => a tx bit is wrong-ish
        fw = torch.sigmoid(worst).pow(FOCAL)                  # in (0,1), big for hard frames
        fw = fw / (fw.mean() + 1e-9)                          # normalize to mean 1
        hard_frac_ema = 0.9*hard_frac_ema + 0.1*float(((-final[:, tx_t]).amax(1) > 0).float().mean())
    perbit = torch.nn.functional.softplus(-posts) * sup_w[None, None, :]     # [L,B,N]
    per_frame_iter = perbit[:, :, tx_t].mean(dim=2)          # [L,B]
    loss = ((per_frame_iter * fw[None, :]).mean(dim=1) * it_w).sum()
    loss = loss + W_ANCHOR*((model.w_vn - b_vn).pow(2).mean() + (model.w_cn - b_cn).pow(2).mean())
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([model.w_vn, model.w_cn], GCLIP); opt.step(); sch.step()
    with torch.no_grad(): model.w_vn.clamp_(W_LO, W_HI); model.w_cn.clamp_(W_LO, W_HI)
    if step % VAL_EVERY == 0 or step == STEPS-1:
        model.cn_update_mode = "hard"; model.eval()
        vf, _ = is_fer(model, VAL_SNR, VAL_TRIALS, seed=7)
        tag = ""
        if vf < best_f: best_f, best_step = vf, step; best_vn = model.w_vn.detach().clone(); best_cn = model.w_cn.detach().clone(); tag=" *best*"
        print(f"  step {step:4d} loss={loss.item():.4f} hardfrac~{hard_frac_ema:.2f} "
              f"val@7.5={vf:.3e} (NNMS {nnms_val:.3e}) ({time.time()-t0:.0f}s){tag}", flush=True)
with torch.no_grad(): model.w_vn.copy_(best_vn); model.w_cn.copy_(best_cn)
model.cn_update_mode = "hard"; model.eval()
for p in model.parameters(): p.requires_grad_(False)
np.savez(ODIR/"model5_beatnnms.npz", w_vn=model.w_vn.cpu().numpy().astype(np.float32),
         w_cn=model.w_cn.cpu().numpy().astype(np.float32), cn_idx=model.cn_idx.cpu().numpy(),
         vn_idx=model.vn_idx.cpu().numpy(), num_iter=np.int32(15), gamma=np.float32(GAMMA), beta=np.float32(BETA))
print(f"# best step {best_step} val@7.5={best_f:.3e} vs NNMS {nnms_val:.3e} "
      f"(improve {nnms_val/best_f:.2f}x)", flush=True)

nnms_m = new_model("hard")
with torch.no_grad(): nnms_m.w_vn.copy_(b_vn); nnms_m.w_cn.copy_(b_cn)
nnms_m.eval()
print("\n# ===== IS floor @ 7.5: MS vs NNMS vs beat-NNMS (200k) =====", flush=True)
print(f"{'SNR':>5} {'MS':>11} {'NNMS':>11} {'beatNNMS':>11} {'vsNNMS':>7}", flush=True)
for e in IS_SNRS:
    fm,_ = is_fer(ms, e, IS_TRIALS); fn,_ = is_fer(nnms_m, e, IS_TRIALS); fb,es = is_fer(model, e, IS_TRIALS)
    print(f"{e:5.1f} {fm:11.3e} {fn:11.3e} {fb:11.3e} {fn/fb:6.2f}x  {'BEATS NNMS' if fb<fn else ''} (ess {100*es:.2f}%)", flush=True)
print("DONE", flush=True)
