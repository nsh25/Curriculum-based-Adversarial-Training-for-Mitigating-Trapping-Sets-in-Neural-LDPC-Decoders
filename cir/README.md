# CIR — Trapping-Set-Aware Neural Min-Sum LDPC Decoding

Low-error-floor decoding of 5G-NR LDPC codes with a **quantized neural normalized
min-sum (Q-NMS)** decoder whose per-edge, per-iteration weights are trained by a
family of trapping-set-aware methods. The core proposed method is a **gradient
(FGSM-style) adversarial attack** on the trapping-set bits with robust min–max
retraining (**Adv-NMS**); **RL-NMS** and **SAC-NMS** then improve it by optimizing
the *exact* hard 5-bit deployment decoder with derivative-free policy gradients.

On the importance-sampled error floor the proposed decoders cut the frame-error
rate by up to **4.9×** over plain min-sum (4.5–5.5 dB), reaching
FER ≈ 3.5×10⁻⁷ at 5 dB, with **no loss of waterfall coding gain**.

> Full derivations, results tables, and the complexity analysis are in
> [`docs/METHODS.md`](docs/METHODS.md), [`docs/PAPER_RL_SAC_math_and_stats.md`](docs/PAPER_RL_SAC_math_and_stats.md),
> and [`docs/complexity_section.tex`](docs/complexity_section.tex).

---

## Install

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9. A CUDA GPU is used automatically if available
(`torch.cuda`); everything also runs on CPU (slower). Tested with
`numpy 2.1`, `torch 2.11`, `scipy`.

## Quickstart

Every script uses **repo-relative paths** (`Path(__file__).parent`), so run them
from anywhere inside the repo. Pretrained weights ship in `weights/`, so you can
reproduce the evaluation curves without retraining.

```bash
# 1. Reproduce the stitched MC + IS floor curves for all 7 decoders
#    (uses the shipped pretrained weights) -> weights/matlab_logs/full_curve_mc_is.mat
python mc_is_full_curve.py

# 2. (optional) Retrain the full pipeline from scratch, in order:
python train_nnms_clip20_eval75.py        # NNMS  gradient baseline
python train_beat_nnms.py                 # beat-NNMS  focal hard-frame fine-tune
python train_is_driven2.py                # IS-NMS  direct IS-FER minimization (warm base)
python train_rl_sac_adv_logged_warm.py    # Adv-NMS (proposed) + RL/SAC improvements

# 3. (optional) Re-run the structural trapping-set search over all base graphs
python run_all_basegraph_traps.py         # writes BaseGraph/*.found.trap
```

Figures are produced in MATLAB from the `.mat` logs:
`weights/matlab_logs/plot_full_curve.m`, `plot_training_logs.m`,
`plot_weight_histograms.m`.

## Repository layout

```
cir/
├── rescue_distill.py            # core: NeuralMSDecoder, channel/LLR, IS helpers
├── tstools/                     # trapping-set library (Tanner graph, decoder, search)
│   ├── tanner.py  decoder.py  trap_search.py  estimate.py  gpu.py
├── train_nnms_clip20_eval75.py  # NNMS   (gradient BCE, train@±20, select IS@±7.5)
├── train_beat_nnms.py           # beat-NNMS (focal hard-frame fine-tune)
├── train_is_driven2.py          # IS-NMS  (minimizes truncated-IS soft-FER)  -> warm base
├── train_rl_sac_adv_logged_warm.py  # Adv-NMS (proposed) + RL-NMS + SAC-NMS (warm-started)
├── mc_is_full_curve.py          # evaluation: MC waterfall + IS floor for all 7 decoders
├── run_all_basegraph_traps.py   # structural trapping-set search over all base graphs
├── BaseGraph/                   # parity-check matrices (*.graph) + trapping sets (*.trap)
├── weights/                     # pretrained weights (*.npz) + training logs (matlab_logs/*.mat)
└── docs/                        # METHODS.md (full theory), paper math+stats, complexity (LaTeX)
```

## Method pipeline

```
MS (w=1)
 └─ NNMS ─ beat-NNMS ─ IS-NMS ─┬─ Adv-NMS (proposed: FGSM attack on TS LLRs + min–max defense)
                               ├─ RL-NMS  (policy gradient on the exact hard 5-bit decoder)
                               └─ SAC-NMS (max-entropy: wider Gaussian policy)
```

All three proposed methods share the IS-NMS warm start and are selected by
IS-estimated floor FER at 4.5 dB on the exact quantized decoder — so the
comparison isolates the training principle, not the starting point.

## Results (IS floor, hard 5-bit ±7.5 decoder, N=1280, R=1/2)

| FER gain over MS | 4.5 dB | 5.0 dB | 5.5 dB |
|---|---|---|---|
| NNMS | 3.22× | 3.48× | 3.25× |
| IS-NMS | 4.66× | 4.85× | 4.42× |
| **Adv-NMS** | **4.68×** | **4.87×** | **4.42×** |
| **SAC-NMS** | 4.68× | 4.87× | 4.42× |
| **RL-NMS** | 4.71× | 4.77× | 4.32× |

Full curves and BER in [`docs/PAPER_RL_SAC_math_and_stats.md`](docs/PAPER_RL_SAC_math_and_stats.md).

## License

MIT — see [`LICENSE`](LICENSE).
