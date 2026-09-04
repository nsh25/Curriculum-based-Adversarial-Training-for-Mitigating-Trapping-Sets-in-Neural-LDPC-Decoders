"""Train NNMS (gradient) with the decoder clip at +/-20 (clean gradients, no
saturation floor), then EVALUATE by IS at the deployment clip +/-7.5 (where the
floor exists and IS is reliable). Question: do +/-20-trained weights beat MS@7.5?

Uses the optimal soft-decoder shape found by the grid search: gamma=0.05, beta=5.
Mechanism: rescue_distill.QMS_MAX is a module global read at call-time, so we
toggle it -> 20 for training forwards, -> 7.5 for validation/eval forwards.
"""
from __future__ import annotations
import importlib.util, math, re, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent
RD = ROOT / "rescue_distill.py"
GRAPH = ROOT / "BaseGraph/5G_LDPC_R0.50_n_dec1280_n1024_k512_z64_s513_640.graph"
TRAP  = ROOT / "BaseGraph/5G_LDPC_R0.50_n_dec1280_n1024_k512_z64_s513_640.found.trap"
OUT   = ROOT / "weights/model5_nnms_train20_eval75.npz"

TRAIN_CLIP, EVAL_CLIP = 20.0, 7.5
GAMMA, BETA = 0.05, 5.0                       # optimal soft shape from the grid search
STEPS, BATCH, LR = 600, 320, 2e-3
TRAIN_EBNO = (2.5, 4.0); TS_FRAC, TS_SHIFT, TS_AMAX = 0.30, (0.8, 1.5), 6
RHO_TS, W_REG, W_LO, W_HI, GCLIP = 2.0, 5e-3, 0.5, 1.6, 1.0
VAL_EVERY, VAL_SNR, VAL_TRIALS = 50, 4.5, 100_000
IS_SNRS, IS_TRIALS, IS_SHIFT = (3.5, 4.0, 4.5, 5.0, 5.5, 6.0), 200_000, 1.2

spec = importlib.util.spec_from_file_location("rescue_distill", RD)
rd = importlib.util.module_from_spec(spec); sys.modules["rescue_distill"] = rd
spec.loader.exec_module(rd)
rd.N_PUNCTURED = 128; rd.SHORT_LO, rd.SHORT_HI = 512, 640; rd.N_SHORTENED = 128; rd.Z_LIFT = 64
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

is_sets, focus = load_sets(), load_sets(TS_AMAX)
K, Kf = len(is_sets), len(focus)
A = torch.zeros(K, N, device=DEV)
for k, v in enumerate(is_sets): A[k, v] = 1.0
a_size = A.sum(1)
Af = torch.zeros(Kf, N, device=DEV)
for k, v in enumerate(focus): Af[k, v] = 1.0
ts_bit = (Af.sum(0) > 0).float(); sup_w = 1.0 + (RHO_TS - 1.0) * ts_bit
it_w = torch.linspace(0.3, 1.0, 15, device=DEV); it_w = it_w / it_w.sum()

def new_model(mode):
    m = rd.NeuralMSDecoder(H, num_iter=15, gamma=GAMMA, beta=BETA, shared_weights=False).to(DEV)
    m.cn_update_mode = mode
    return m

@torch.no_grad()
def is_fer(model, ebno, trials, clip, shift=IS_SHIFT, batch=6144, seed=222):
    old = rd.QMS_MAX; rd.QMS_MAX = clip                    # decode at requested clip
    try:
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
    finally:
        rd.QMS_MAX = old

# MS baseline at eval clip
ms = new_model("hard"); ms.eval()
for p in ms.parameters(): p.requires_grad_(False)
ms_val, _ = is_fer(ms, VAL_SNR, VAL_TRIALS, EVAL_CLIP, seed=7)
print(f"# train@clip{TRAIN_CLIP} eval@clip{EVAL_CLIP} | gamma={GAMMA} beta={BETA} | "
      f"K={K} | MS@{EVAL_CLIP} valFER@{VAL_SNR}={ms_val:.3e}", flush=True)

model = new_model("soft"); model.train()
opt = torch.optim.Adam([model.w_vn, model.w_cn], lr=LR)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
g = torch.Generator(device=DEV).manual_seed(1)
e_lo, e_hi = TRAIN_EBNO; s_lo, s_hi = TS_SHIFT; nsh = int(BATCH*TS_FRAC)
best_f, best_vn, best_cn, best_step = float("inf"), None, None, -1
t0 = time.time()
for step in range(STEPS):
    rd.QMS_MAX = TRAIN_CLIP                                # <-- train forwards at +/-20
    model.cn_update_mode = "soft"; model.train()
    ebno = float(torch.empty(1).uniform_(e_lo, e_hi).item())
    sig = rd.awgn_sigma(ebno, RATE); s2 = sig*sig
    y = 1.0 + sig*torch.randn(BATCH, N, device=DEV, generator=g)
    if nsh:
        ks = torch.randint(0, Kf, (nsh,), device=DEV, generator=g)
        sh = torch.empty(nsh, 1, device=DEV).uniform_(s_lo, s_hi)
        y[-nsh:] = y[-nsh:] - sh*Af[ks]
    llr = rd.apply_puncture(2.0*y/s2)
    posts = model(llr, return_all_posteriors=True)
    perbit = torch.nn.functional.softplus(-posts) * sup_w[None,None,:]
    loss = (perbit[:,:,tx_t].mean(dim=(1,2)) * it_w).sum() \
           + W_REG*((model.w_vn-1.0).pow(2).mean() + (model.w_cn-1.0).pow(2).mean())
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([model.w_vn, model.w_cn], GCLIP); opt.step(); sch.step()
    with torch.no_grad():
        model.w_vn.clamp_(W_LO, W_HI); model.w_cn.clamp_(W_LO, W_HI)
    if step % VAL_EVERY == 0 or step == STEPS-1:
        model.cn_update_mode = "hard"; model.eval()
        vf, _ = is_fer(model, VAL_SNR, VAL_TRIALS, EVAL_CLIP, seed=7)   # select on +/-7.5
        marker = ""
        if vf < best_f:
            best_f, best_step = vf, step
            best_vn = model.w_vn.detach().clone(); best_cn = model.w_cn.detach().clone(); marker = " *best*"
        print(f"  step {step:4d} loss={loss.item():.4f} val@7.5FER={vf:.3e} "
              f"w[{float(model.w_vn.min()):.2f},{float(model.w_vn.max()):.2f}] ({time.time()-t0:.0f}s){marker}",
              flush=True)

with torch.no_grad():
    model.w_vn.copy_(best_vn); model.w_cn.copy_(best_cn)
model.cn_update_mode = "hard"; model.eval()
for p in model.parameters(): p.requires_grad_(False)
np.savez(OUT, w_vn=model.w_vn.cpu().numpy().astype(np.float32), w_cn=model.w_cn.cpu().numpy().astype(np.float32),
         cn_idx=model.cn_idx.cpu().numpy(), vn_idx=model.vn_idx.cpu().numpy(),
         num_iter=np.int32(15), gamma=np.float32(GAMMA), beta=np.float32(BETA))
print(f"# best step {best_step} (val@7.5 {best_f:.3e}); saved {OUT.name} "
      f"w_vn mu={float(model.w_vn.mean()):.4f} sd={float(model.w_vn.std()):.4f}", flush=True)

print(f"\n# ===== IS floor @ clip {EVAL_CLIP}: NNMS(trained@20) vs MS (200k/pt) =====", flush=True)
print(f"{'SNR':>5} {'MS@7.5':>11} {'NNMS20@7.5':>11} {'gain':>6}  verdict", flush=True)
wins = 0
for e in IS_SNRS:
    f_ms, _ = is_fer(ms, e, IS_TRIALS, EVAL_CLIP)
    f_tr, es = is_fer(model, e, IS_TRIALS, EVAL_CLIP)
    gain = f_ms/f_tr if f_tr>0 else float("inf"); win = f_tr < f_ms; wins += int(win)
    print(f"{e:5.1f} {f_ms:11.3e} {f_tr:11.3e} {gain:5.2f}x  {'BEATS MS' if win else 'no'} (ess {100*es:.2f}%)", flush=True)
print(f"\n# NNMS(trained@20) beats MS@7.5 at {wins}/{len(IS_SNRS)} floor SNRs", flush=True)
print("DONE", flush=True)
