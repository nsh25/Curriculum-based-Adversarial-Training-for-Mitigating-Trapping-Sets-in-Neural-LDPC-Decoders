"""Instrumented RL-NMS / SAC-NMS / Adv-NMS training, WARM-STARTED from our best
IS-developed weights (model5_isdriven2.npz, ~4.9x MS) instead of MS(w=1), so the
RL/SAC/Adv fine-tune ON TOP of the strong IS-sampled base. Trains + evals at the
deployment clip +/-7.5 with HARD trap shifts (residual signal so the reward/loss
does not saturate). Logs reward/loss, BER+FER over training, weight stats, and
final weights to MATLAB .mat (matlab_logs/*_warm.mat). Plot with plot_training_logs.m
(point it at the *_warm.mat files).
"""
from __future__ import annotations
import importlib.util, math, re, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.io import savemat

ROOT = Path(__file__).resolve().parent
RD = ROOT / "rescue_distill.py"
GRAPH = ROOT / "BaseGraph/5G_LDPC_R0.50_n_dec1280_n1024_k512_z64_s513_640.graph"
TRAP  = ROOT / "BaseGraph/5G_LDPC_R0.50_n_dec1280_n1024_k512_z64_s513_640.found.trap"
ODIR  = ROOT / "weights"
OUTDIR = ODIR / "matlab_logs"; OUTDIR.mkdir(parents=True, exist_ok=True)
BASE_NPZ = ODIR / "model5_isdriven2.npz"           # best IS-developed weights (~4.9x MS)

CLIP = 7.5                                          # train + eval at deployment clip
GAMMA, BETA = 0.05, 5.0
TRAIN_EBNO = (3.5, 5.0); TS_SHIFT = (1.2, 2.8); MARGIN_SCALE = 4.0   # hard -> residual signal
VAL_SNR, VAL_TRIALS = 4.5, 150_000
IS_SNRS, IS_TRIALS = (3.5, 4.0, 4.5, 5.0, 5.5, 6.0), 200_000

spec = importlib.util.spec_from_file_location("rescue_distill", RD)
rd = importlib.util.module_from_spec(spec); sys.modules["rescue_distill"] = rd
spec.loader.exec_module(rd)
rd.N_PUNCTURED = 128; rd.SHORT_LO, rd.SHORT_HI = 512, 640; rd.N_SHORTENED = 128; rd.Z_LIFT = 64
rd.QMS_MAX = CLIP
DEV = rd.DEVICE
H = rd.load_H(GRAPH); M, N = H.shape
RATE = 512 / (N - 256)
tx = np.ones(N, bool); tx[:128] = False; tx[512:640] = False
tx_t = torch.as_tensor(tx, device=DEV); bad = set(np.nonzero(~tx)[0].tolist()); n_tx = int(tx.sum())

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
ts_bit = (Af.sum(0) > 0).float()
it_w = torch.linspace(0.3, 1.0, 15, device=DEV); it_w = it_w / it_w.sum()

def new_model(mode):
    m = rd.NeuralMSDecoder(H, num_iter=15, gamma=GAMMA, beta=BETA, shared_weights=False).to(DEV)
    m.cn_update_mode = mode; return m

def load_base(model):
    z = np.load(BASE_NPZ); Mc = model.num_cns
    zkey = z["vn_idx"].astype(np.int64)*Mc + z["cn_idx"].astype(np.int64)
    mkey = model.vn_idx.cpu().numpy().astype(np.int64)*Mc + model.cn_idx.cpu().numpy().astype(np.int64)
    oz = np.argsort(zkey); pos = oz[np.searchsorted(zkey, mkey, sorter=oz)]
    assert np.array_equal(zkey[pos], mkey)
    return (torch.from_numpy(np.ascontiguousarray(z["w_vn"][:, pos])).to(DEV),
            torch.from_numpy(np.ascontiguousarray(z["w_cn"][:, pos])).to(DEV))

@torch.no_grad()
def is_ber_fer(model, ebno, trials, shift=1.2, batch=6144, seed=222):
    g = torch.Generator(device=DEV).manual_seed(seed)
    sig = rd.awgn_sigma(ebno, RATE); s2 = sig*sig; amp = shift; logK = math.log(K)
    nb = max(1, math.ceil(trials/batch)); actual = nb*batch
    wf = torch.zeros((), device=DEV, dtype=torch.float64); wb = torch.zeros((), device=DEV, dtype=torch.float64)
    w2 = torch.zeros((), device=DEV, dtype=torch.float64)
    for _ in range(nb):
        ks = torch.randint(0, K, (batch,), device=DEV, generator=g)
        nz = sig*torch.randn(batch, N, device=DEV, generator=g) - amp*A[ks]
        hard = (model(rd.apply_puncture(2.0*(1.0+nz)/s2)) < 0.0)[:, tx_t]
        ffail = hard.any(1).float(); berr = hard.sum(1).float()/n_tx
        expo = (-amp*(nz@A.T))/s2 - a_size[None,:]*(amp*amp)/(2.0*s2)
        w = torch.exp(logK - torch.logsumexp(expo,1))
        wf += (w*ffail).double().sum(); wb += (w*berr).double().sum(); w2 += ((w*ffail)**2).double().sum()
    fer = float(wf)/actual; ber = float(wb)/actual
    ess = (float(wf)**2/float(w2))/actual if float(w2)>0 else 0
    return fer, ber, ess

def wstats(m):
    return (float(m.w_vn.mean()), float(m.w_vn.std()), float(m.w_cn.mean()), float(m.w_cn.std()))

def sample(B, gen, frac=0.5):
    ebno = float(torch.empty(1).uniform_(*TRAIN_EBNO).item())
    sig = rd.awgn_sigma(ebno, RATE); s2 = sig*sig
    y = 1.0 + sig*torch.randn(B, N, device=DEV, generator=gen)
    nsh = int(B*frac); sup = torch.zeros(B, N, device=DEV)
    if nsh:
        ks = torch.randint(0, Kf, (nsh,), device=DEV, generator=gen)
        sh = torch.empty(nsh, 1, device=DEV).uniform_(*TS_SHIFT)
        y[-nsh:] = y[-nsh:] - sh*Af[ks]; sup[-nsh:] = Af[ks]
    return y, s2, sup

def train_es_logged(steps, pairs, sigma, lr, seed, label, base_vn, base_cn, val_every=10):
    m = new_model("hard"); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    em = (ts_bit[m.vn_idx] > 0).long(); idx_vn = em; idx_cn = em + 2
    theta = torch.zeros(15, 4, device=DEV); g = torch.Generator(device=DEV).manual_seed(seed); TH = 0.7
    def set_w(th):
        mult = torch.exp(th.clamp(-TH, TH))
        with torch.no_grad(): m.w_vn.copy_(base_vn*mult[:, idx_vn]); m.w_cn.copy_(base_cn*mult[:, idx_cn])
    @torch.no_grad()
    def reward(th):
        set_w(th)
        y, s2, sup = sample(512, g, 0.6); post = m(rd.apply_puncture(2.0*y/s2))
        msk = sup > 0
        return float(torch.tanh(post[msk]/MARGIN_SCALE).mean().item()) if msk.any() else 0.0
    L = dict(step=[], reward=[], val_step=[], val_fer=[], val_ber=[], wvn_mean=[], wvn_std=[], wcn_mean=[], wcn_std=[])
    best_f, best_th = float("inf"), theta.clone(); t0 = time.time()
    for step in range(steps):
        grad = torch.zeros_like(theta)
        for _ in range(pairs):
            xi = sigma*torch.randn(15, 4, device=DEV, generator=g)
            grad += (reward(theta+xi) - reward(theta-xi))*xi
        theta = (theta + lr*grad/(2*pairs*sigma)).clamp(-TH, TH)
        L["step"].append(step); L["reward"].append(reward(theta))
        if step % val_every == 0 or step == steps-1:
            set_w(theta); m.cn_update_mode = "hard"; m.eval()
            fer, ber, _ = is_ber_fer(m, VAL_SNR, VAL_TRIALS, seed=7)
            mv, sv, mc, sc = wstats(m)
            L["val_step"].append(step); L["val_fer"].append(fer); L["val_ber"].append(ber)
            L["wvn_mean"].append(mv); L["wvn_std"].append(sv); L["wcn_mean"].append(mc); L["wcn_std"].append(sc)
            if fer < best_f: best_f, best_th = fer, theta.clone()
            print(f"  [{label}] step {step:3d} reward={L['reward'][-1]:+.4f} FER={fer:.3e} BER={ber:.3e} ({time.time()-t0:.0f}s)", flush=True)
    set_w(best_th)
    return m, L

def train_adv_logged(steps, lr, eps, seed, label, base_vn, base_cn, val_every=20):
    m = new_model("soft")
    with torch.no_grad(): m.w_vn.copy_(base_vn); m.w_cn.copy_(base_cn)
    m.train()
    opt = torch.optim.Adam([m.w_vn, m.w_cn], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    g = torch.Generator(device=DEV).manual_seed(seed); sup_w = 1.0 + 1.0*ts_bit
    L = dict(step=[], loss=[], val_step=[], val_fer=[], val_ber=[], wvn_mean=[], wvn_std=[], wcn_mean=[], wcn_std=[])
    best_f, best_vn, best_cn = float("inf"), None, None; t0 = time.time()
    for step in range(steps):
        m.cn_update_mode = "soft"; m.train()
        y, s2, sup = sample(320, g, 0.4); base = 2.0*y/s2
        adv = base.detach().clone().requires_grad_(True)
        p0 = m(rd.apply_puncture(adv))
        l0 = (torch.nn.functional.softplus(-p0)*sup_w[None,:])[:, tx_t].mean()
        gadv, = torch.autograd.grad(l0, adv)
        adv_in = base - eps*gadv.sign()*(sup > 0).float()
        posts = m(rd.apply_puncture(adv_in), return_all_posteriors=True)
        perbit = torch.nn.functional.softplus(-posts)*sup_w[None,None,:]
        loss = (perbit[:,:,tx_t].mean(dim=(1,2))*it_w).sum() + 2e-3*((m.w_vn-base_vn).pow(2).mean()+(m.w_cn-base_cn).pow(2).mean())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([m.w_vn, m.w_cn], 1.0); opt.step(); sch.step()
        with torch.no_grad(): m.w_vn.clamp_(0.4, 1.8); m.w_cn.clamp_(0.4, 1.8)
        L["step"].append(step); L["loss"].append(float(loss.item()))
        if step % val_every == 0 or step == steps-1:
            m.cn_update_mode = "hard"; m.eval()
            fer, ber, _ = is_ber_fer(m, VAL_SNR, VAL_TRIALS, seed=7)
            mv, sv, mc, sc = wstats(m)
            L["val_step"].append(step); L["val_fer"].append(fer); L["val_ber"].append(ber)
            L["wvn_mean"].append(mv); L["wvn_std"].append(sv); L["wcn_mean"].append(mc); L["wcn_std"].append(sc)
            if fer < best_f: best_f = fer; best_vn = m.w_vn.detach().clone(); best_cn = m.w_cn.detach().clone()
            print(f"  [{label}] step {step:3d} loss={loss.item():.4f} FER={fer:.3e} BER={ber:.3e} ({time.time()-t0:.0f}s)", flush=True)
    with torch.no_grad(): m.w_vn.copy_(best_vn); m.w_cn.copy_(best_cn)
    m.cn_update_mode = "hard"; m.eval()
    return m, L

def finalize(m, L, label, fname):
    ebno = np.array(IS_SNRS, float); fer = np.zeros_like(ebno); ber = np.zeros_like(ebno)
    for i, e in enumerate(ebno): fer[i], ber[i], _ = is_ber_fer(m, float(e), IS_TRIALS)
    out = {k: np.asarray(v, float) for k, v in L.items()}
    out.update(dict(method=label, warm_start_base="model5_isdriven2.npz", ebno_db=ebno,
                    fer_curve=fer, ber_curve=ber,
                    w_vn_final=m.w_vn.detach().cpu().numpy().astype(np.float32),
                    w_cn_final=m.w_cn.detach().cpu().numpy().astype(np.float32),
                    cn_idx=m.cn_idx.cpu().numpy(), vn_idx=m.vn_idx.cpu().numpy(),
                    num_iter=15, gamma=GAMMA, beta=BETA, clip=CLIP))
    savemat(str(OUTDIR/fname), out, do_compression=True)
    print(f"  saved {fname}: final FER {fer[2]:.3e} @4.5dB", flush=True)
    return out

print(f"# WARM-START from {BASE_NPZ.name} | train+eval @clip{CLIP} | out -> {OUTDIR}", flush=True)
ms = new_model("hard"); ms.eval()
for p in ms.parameters(): p.requires_grad_(False)
probe = new_model("hard"); b_vn, b_cn = load_base(probe)
with torch.no_grad(): probe.w_vn.copy_(b_vn); probe.w_cn.copy_(b_cn)
probe.eval()
ms_f = np.array([is_ber_fer(ms, e, IS_TRIALS)[0] for e in IS_SNRS])
ms_b = np.array([is_ber_fer(ms, e, IS_TRIALS)[1] for e in IS_SNRS])
base_f = np.array([is_ber_fer(probe, e, IS_TRIALS)[0] for e in IS_SNRS])
base_b = np.array([is_ber_fer(probe, e, IS_TRIALS)[1] for e in IS_SNRS])
print(f"# MS@4.5={ms_f[2]:.3e}  base(IS-driven2)@4.5={base_f[2]:.3e}", flush=True)

print("\n### RL-NMS (warm) ###", flush=True)
rl, Lrl = train_es_logged(120, 8, 0.05, 0.15, 0, "RL", b_vn, b_cn); orl = finalize(rl, Lrl, "RL-NMS-warm", "rlnms_warm.mat")
print("\n### SAC-NMS (warm) ###", flush=True)
sac, Lsac = train_es_logged(120, 12, 0.08, 0.10, 3, "SAC", b_vn, b_cn); osac = finalize(sac, Lsac, "SAC-NMS-warm", "sacnms_warm.mat")
print("\n### Adv-NMS (warm) ###", flush=True)
adv, Ladv = train_adv_logged(300, 1e-3, 1.5, 1, "Adv", b_vn, b_cn); oadv = finalize(adv, Ladv, "Adv-NMS-warm", "advnms_warm.mat")

comb = dict(ebno_db=np.array(IS_SNRS, float), MS_fer=ms_f, MS_ber=ms_b, base_fer=base_f, base_ber=base_b,
            warm_start_base="model5_isdriven2.npz")
for tag, o in [("RL", orl), ("SAC", osac), ("Adv", oadv)]:
    comb[f"{tag}_fer_curve"] = o["fer_curve"]; comb[f"{tag}_ber_curve"] = o["ber_curve"]
    comb[f"{tag}_train_step"] = o["step"]; comb[f"{tag}_metric"] = o.get("reward", o.get("loss"))
    comb[f"{tag}_metric_name"] = "reward" if "reward" in o else "loss"
    comb[f"{tag}_val_step"] = o["val_step"]; comb[f"{tag}_val_fer"] = o["val_fer"]; comb[f"{tag}_val_ber"] = o["val_ber"]
    comb[f"{tag}_wvn_mean"] = o["wvn_mean"]; comb[f"{tag}_wvn_std"] = o["wvn_std"]
savemat(str(OUTDIR/"training_logs_warm_all.mat"), comb, do_compression=True)
print(f"\n# saved *_warm.mat + training_logs_warm_all.mat in {OUTDIR}", flush=True)
print("DONE", flush=True)
