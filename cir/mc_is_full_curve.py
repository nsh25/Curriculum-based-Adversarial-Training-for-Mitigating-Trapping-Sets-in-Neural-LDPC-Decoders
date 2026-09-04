"""Full BER/FER curves for MS, NNMS, and IS-NMS (the IS-driven method that beats
NNMS), at the deployment clip +/-7.5:
  * MC  (Monte-Carlo)          Eb/N0 = 1.0 : 0.5 : 4.0  (waterfall, feasible)
  * IS  (TS-mixture)           Eb/N0 = 4.0 : 0.5 : 8.0  (error floor)
They overlap at 4.0 dB (MC vs IS cross-check). Saves everything to MATLAB .mat
(matlab_logs/full_curve_mc_is.mat) + plot_full_curve.m.
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
NNMS_NPZ = ODIR / "model5_nnms_train20_eval75.npz"
BEATNNMS_NPZ = ODIR / "model5_beatnnms.npz"      # hard-focal method that beats NNMS
ISNMS_NPZ = ODIR / "model5_isdriven2.npz"        # IS-sample-trained method that beats NNMS

CLIP = 7.5; GAMMA, BETA = 0.05, 5.0
MC_EBNO = np.arange(1.0, 3.0 + 1e-9, 0.5)      # waterfall (MC reliable to ~3 dB)
IS_EBNO = np.arange(3.0, 8.0 + 1e-9, 0.5)      # floor (IS from 3 dB up to 8 dB)
MC_TARGET_FE, MC_CAP, MC_BATCH = 300, 4_000_000, 8192
IS_TRIALS, IS_SHIFT = 300_000, 1.2

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

def load_sets():
    pat = re.compile(r"\((\d+),\s*(\d+)\)\s*(.*)"); out = []
    for ln in TRAP.read_text().splitlines():
        m = pat.match(ln.strip())
        if not m: continue
        a, b = int(m.group(1)), int(m.group(2)); v = [int(x)-1 for x in m.group(3).split()]
        if b <= 2 and 3 <= a <= 10 and not any(x in bad for x in v): out.append(v)
    return out
is_sets = load_sets(); K = len(is_sets)
A = torch.zeros(K, N, device=DEV)
for k, v in enumerate(is_sets): A[k, v] = 1.0
a_size = A.sum(1)

def new_model():
    m = rd.NeuralMSDecoder(H, num_iter=15, gamma=GAMMA, beta=BETA, shared_weights=False).to(DEV)
    m.cn_update_mode = "hard"; m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def load_into(model, path):
    p = str(path); Mc = model.num_cns
    if p.endswith(".mat"):
        from scipy.io import loadmat
        z = loadmat(p); wv_all, wc_all = z["w_vn_final"], z["w_cn_final"]
    else:
        z = np.load(p); wv_all, wc_all = z["w_vn"], z["w_cn"]
    zvn = z["vn_idx"].ravel().astype(np.int64); zcn = z["cn_idx"].ravel().astype(np.int64)
    zkey = zvn*Mc + zcn
    mkey = model.vn_idx.cpu().numpy().astype(np.int64)*Mc + model.cn_idx.cpu().numpy().astype(np.int64)
    oz = np.argsort(zkey); pos = oz[np.searchsorted(zkey, mkey, sorter=oz)]
    assert np.array_equal(zkey[pos], mkey)
    wv, wc = wv_all[:, pos], wc_all[:, pos]
    if wv.shape[0] < 15: wv = np.pad(wv, ((0,15-wv.shape[0]),(0,0)), mode="edge"); wc = np.pad(wc, ((0,15-wc.shape[0]),(0,0)), mode="edge")
    with torch.no_grad():
        model.w_vn.copy_(torch.from_numpy(np.ascontiguousarray(wv)).to(DEV))
        model.w_cn.copy_(torch.from_numpy(np.ascontiguousarray(wc)).to(DEV))
    return model

@torch.no_grad()
def mc(model, ebno, target_fe=MC_TARGET_FE, cap=MC_CAP, batch=MC_BATCH, seed=0):
    rd.QMS_MAX = CLIP
    g = torch.Generator(device=DEV).manual_seed(seed)
    sig = rd.awgn_sigma(ebno, RATE); s2 = sig*sig
    fe = be = done = 0
    while done < cap:
        y = 1.0 + sig*torch.randn(batch, N, device=DEV, generator=g)
        hard = (model(rd.apply_puncture(2.0*y/s2)) < 0.0)[:, tx_t]
        fe += int(hard.any(1).sum().item()); be += int(hard.sum().item()); done += batch
        if fe >= target_fe: break
    return fe/done, be/(done*n_tx), done, fe

@torch.no_grad()
def is_est(model, ebno, trials=IS_TRIALS, shift=IS_SHIFT, batch=8192, seed=222):
    rd.QMS_MAX = CLIP
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

DECODERS = [("MS", None), ("NNMS", NNMS_NPZ),
            ("beatNNMS", BEATNNMS_NPZ),          # hard-focal (beats NNMS)
            ("IS_NMS", ISNMS_NPZ),               # IS-sample trained (beats NNMS)
            ("RL_warm", OUTDIR/"rlnms_warm.mat"),   # RL-NMS warm-started from IS-NMS
            ("SAC_warm", OUTDIR/"sacnms_warm.mat"), # SAC-NMS warm-started from IS-NMS
            ("Adv_warm", OUTDIR/"advnms_warm.mat")] # Adv-NMS warm-started from IS-NMS
out = dict(mc_ebno=MC_EBNO, is_ebno=IS_EBNO)
print(f"# full curve @clip{CLIP} | MC {MC_EBNO[0]}:{MC_EBNO[-1]}dB, IS {IS_EBNO[0]}:{IS_EBNO[-1]}dB", flush=True)
for name, npz in DECODERS:
    m = new_model()
    if npz is not None: load_into(m, npz)
    t0 = time.time()
    mc_fer = []; mc_ber = []
    print(f"\n=== {name} : MC (waterfall) ===", flush=True)
    for e in MC_EBNO:
        f, b, fr, fe = mc(m, float(e))
        mc_fer.append(f); mc_ber.append(b)
        print(f"  {e:.1f} dB  FER={f:.3e} BER={b:.3e} ({fe}FE/{fr:,})", flush=True)
    is_fer = []; is_ber = []; is_ess = []
    print(f"=== {name} : IS (floor) ===", flush=True)
    for e in IS_EBNO:
        f, b, es = is_est(m, float(e))
        is_fer.append(f); is_ber.append(b); is_ess.append(es)
        print(f"  {e:.1f} dB  FER={f:.3e} BER={b:.3e} (ess {100*es:.2f}%)", flush=True)
    out[f"{name}_mc_fer"] = np.array(mc_fer); out[f"{name}_mc_ber"] = np.array(mc_ber)
    out[f"{name}_is_fer"] = np.array(is_fer); out[f"{name}_is_ber"] = np.array(is_ber)
    out[f"{name}_is_ess"] = np.array(is_ess)
    print(f"  {name} done ({time.time()-t0:.0f}s)", flush=True)

savemat(str(OUTDIR/"full_curve_mc_is.mat"), out, do_compression=True)
print(f"\n# saved full_curve_mc_is.mat in {OUTDIR}", flush=True)
print("DONE", flush=True)
