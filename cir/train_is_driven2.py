"""IS-driven training v2 -- effective-sample-size fixed.

v1 diverged because the IS effective sample size was ~0.3%: each micro-batch's
gradient rode on 1-2 high-weight samples. Fixes here:
  * MORE raw samples:   ACC=24 micro-batches of 512 -> ~12k IS samples/step
                        (grad-accum, so peak memory stays one micro-batch);
  * GLOBAL normalization: pre-sample the whole step, weight-normalize across all
                        ~12k samples (not per-512);
  * WEIGHT TRUNCATION:  clip IS weights at TRUNC x mean -> kills the few-sample
                        domination, raising the effective sample size a lot;
  * mild SHIFT RANGE for higher ESS, lower LR + stronger anchor to stop divergence.
Prints the per-step training ESS so the improvement is visible. Warm-start v1.
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
BASE_NPZ = ODIR / "model5_beatnnms.npz"
OUT_NPZ  = ODIR / "model5_isdriven2.npz"

GAMMA, BETA = 0.05, 5.0
TRAIN_EBNO = (4.0, 5.5); SHIFT_RANGE = (0.9, 1.4); TRUNC = 20.0
SMAX_T, SIG_T = 2.0, 2.0; LAM_BCE = 0.3
STEPS, MB, ACC, LR = 500, 512, 24, 2e-4        # ~12k IS samples/step
W_ANCHOR, W_LO, W_HI, GCLIP = 3e-3, 0.4, 1.8, 0.5
VAL_EVERY, VAL_SNR, VAL_TRIALS = 25, 4.5, 250_000
IS_SNRS, IS_TRIALS = (4.0, 4.5, 5.0, 5.5, 6.0), 300_000

spec = importlib.util.spec_from_file_location("rescue_distill", RD)
rd = importlib.util.module_from_spec(spec); sys.modules["rescue_distill"] = rd
spec.loader.exec_module(rd)
rd.N_PUNCTURED = 128; rd.SHORT_LO, rd.SHORT_HI = 512, 640; rd.N_SHORTENED = 128; rd.Z_LIFT = 64
rd.QMS_MAX = 7.5
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
is_sets = load_sets(); K = len(is_sets)
A = torch.zeros(K, N, device=DEV)
for k, v in enumerate(is_sets): A[k, v] = 1.0
a_size = A.sum(1)
it_w = torch.linspace(0.3, 1.0, 15, device=DEV); it_w = it_w / it_w.sum()

def new_model(mode):
    m = rd.NeuralMSDecoder(H, num_iter=15, gamma=GAMMA, beta=BETA, shared_weights=False).to(DEV)
    m.cn_update_mode = mode; return m

def load_npz(path, model):
    z = np.load(path); Mc = model.num_cns
    zkey = z["vn_idx"].astype(np.int64)*Mc + z["cn_idx"].astype(np.int64)
    mkey = model.vn_idx.cpu().numpy().astype(np.int64)*Mc + model.cn_idx.cpu().numpy().astype(np.int64)
    oz = np.argsort(zkey); pos = oz[np.searchsorted(zkey, mkey, sorter=oz)]
    assert np.array_equal(zkey[pos], mkey)
    return (torch.from_numpy(np.ascontiguousarray(z["w_vn"][:, pos])).to(DEV),
            torch.from_numpy(np.ascontiguousarray(z["w_cn"][:, pos])).to(DEV))

@torch.no_grad()
def is_fer(model, ebno, trials, shift=1.2, batch=8192, seed=222):
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

ms = new_model("hard"); ms.eval()
for p in ms.parameters(): p.requires_grad_(False)
nnms = new_model("hard"); nv, nc = load_npz(NNMS_NPZ, nnms)
with torch.no_grad(): nnms.w_vn.copy_(nv); nnms.w_cn.copy_(nc)
nnms.eval()
v1 = new_model("hard"); b_vn, b_cn = load_npz(BASE_NPZ, v1)
with torch.no_grad(): v1.w_vn.copy_(b_vn); v1.w_cn.copy_(b_cn)
v1.eval()
ms_val,_ = is_fer(ms, VAL_SNR, VAL_TRIALS); nn_val,_ = is_fer(nnms, VAL_SNR, VAL_TRIALS); v1_val,_ = is_fer(v1, VAL_SNR, VAL_TRIALS)
print(f"# IS-driven v2 | MS@4.5={ms_val:.3e} NNMS={nn_val:.3e} v1={v1_val:.3e} | {MB}x{ACC}={MB*ACC} smp/step, trunc={TRUNC}", flush=True)

model = new_model("soft")
with torch.no_grad(): model.w_vn.copy_(b_vn); model.w_cn.copy_(b_cn)
model.train()
opt = torch.optim.Adam([model.w_vn, model.w_cn], lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=2e-5)
g = torch.Generator(device=DEV).manual_seed(3)
logK = math.log(K); s_lo, s_hi = SHIFT_RANGE
best_f, best_vn, best_cn, best_step = v1_val, b_vn.clone(), b_cn.clone(), -1
t0 = time.time()
for step in range(STEPS):
    ebno = float(torch.empty(1).uniform_(*TRAIN_EBNO).item())
    sig = rd.awgn_sigma(ebno, RATE); s2 = sig*sig
    # ---- pre-sample the whole step, compute GLOBAL truncated IS weights ----
    nzs, shs = [], []
    with torch.no_grad():
        w_list = []
        for _j in range(ACC):
            ks = torch.randint(0, K, (MB,), device=DEV, generator=g)
            sh = torch.empty(MB, 1, device=DEV).uniform_(s_lo, s_hi)
            nz = sig*torch.randn(MB, N, device=DEV, generator=g) - sh*A[ks]
            expo = (-sh*(nz@A.T))/s2 - a_size[None,:]*(sh*sh)/(2.0*s2)   # per-sample shift
            w = torch.exp(logK - torch.logsumexp(expo, 1))
            nzs.append(nz); w_list.append(w)
        w_all = torch.cat(w_list)
        cap = TRUNC * w_all.mean()
        w_all = w_all.clamp(max=cap)
        denom = w_all.sum() + 1e-12
        ess = float((w_all.sum()**2)/(w_all*w_all).sum()) / (MB*ACC)
        w_all = w_all / denom
    # ---- forward/backward per micro-batch with global weights ----
    model.cn_update_mode = "soft"; model.train(); opt.zero_grad(); li = 0.0
    for _j in range(ACC):
        nz = nzs[_j]; w = w_all[_j*MB:(_j+1)*MB]
        posts = model(rd.apply_puncture(2.0*(1.0+nz)/s2), return_all_posteriors=True)
        final = posts[-1]
        worst = SMAX_T * torch.logsumexp(-final[:, tx_t] / SMAX_T, dim=1)
        fail_soft = torch.sigmoid(worst / SIG_T)
        bce = torch.nn.functional.softplus(-posts)[:, :, tx_t].mean(dim=2)
        loss = (w*fail_soft).sum() + LAM_BCE*((w[None,:]*bce).sum(dim=1)*it_w).sum()
        loss.backward(); li += float((w*fail_soft).sum().item())
    (W_ANCHOR*((model.w_vn-b_vn).pow(2).mean()+(model.w_cn-b_cn).pow(2).mean())).backward()
    torch.nn.utils.clip_grad_norm_([model.w_vn, model.w_cn], GCLIP); opt.step(); sch.step()
    with torch.no_grad(): model.w_vn.clamp_(W_LO, W_HI); model.w_cn.clamp_(W_LO, W_HI)
    del nzs, w_list
    if step % VAL_EVERY == 0 or step == STEPS-1:
        model.cn_update_mode = "hard"; model.eval()
        vf,_ = is_fer(model, VAL_SNR, VAL_TRIALS, seed=7)
        tag = ""
        if vf < best_f: best_f, best_step = vf, step; best_vn = model.w_vn.detach().clone(); best_cn = model.w_cn.detach().clone(); tag=" *best*"
        print(f"  step {step:3d} softFER={li:.4e} trainESS={100*ess:.1f}% val@4.5={vf:.3e} "
              f"(v1 {v1_val:.3e}) ({time.time()-t0:.0f}s){tag}", flush=True)

with torch.no_grad(): model.w_vn.copy_(best_vn); model.w_cn.copy_(best_cn)
model.cn_update_mode = "hard"; model.eval()
for p in model.parameters(): p.requires_grad_(False)
np.savez(OUT_NPZ, w_vn=model.w_vn.cpu().numpy().astype(np.float32), w_cn=model.w_cn.cpu().numpy().astype(np.float32),
         cn_idx=model.cn_idx.cpu().numpy(), vn_idx=model.vn_idx.cpu().numpy(),
         num_iter=np.int32(15), gamma=np.float32(GAMMA), beta=np.float32(BETA))
print(f"# best step {best_step} val@4.5={best_f:.3e} (v1 {v1_val:.3e}, improve {v1_val/best_f:.2f}x); saved {OUT_NPZ.name}", flush=True)

print("\n# ===== IS floor @7.5: MS / NNMS / v1 / IS-driven2 (300k) =====", flush=True)
print(f"{'SNR':>5} {'MS':>11} {'NNMS':>11} {'v1':>11} {'ISdriv2':>11} {'/MS':>6} {'/v1':>6}", flush=True)
for e in IS_SNRS:
    fm,_ = is_fer(ms, e, IS_TRIALS); fn,_ = is_fer(nnms, e, IS_TRIALS)
    f1,_ = is_fer(v1, e, IS_TRIALS); f2,es = is_fer(model, e, IS_TRIALS)
    print(f"{e:5.1f} {fm:11.3e} {fn:11.3e} {f1:11.3e} {f2:11.3e} {fm/f2:5.2f}x {f1/f2:5.2f}x (ess {100*es:.2f}%)", flush=True)
print("DONE", flush=True)
