# Trapping-Set-Aware Neural Min-Sum LDPC Decoding

Low-error-floor decoding of 5G-NR LDPC codes with a **quantized neural normalized
min-sum (Q-NMS)** decoder whose per-edge, per-iteration weights are learned by a
family of trapping-set-aware methods:

**Proposed method (core): gradient-based adversarial attack training (Adv-NMS)**
— a signed-gradient (FGSM-style) attacker perturbs the channel LLRs on the
trapping-set bits, and the decoder weights are retrained to survive the attack
(a min–max robust optimization). **RL-NMS and SAC-NMS are then applied to
improve this method**: the gradient attack/defense must run through a
differentiable surrogate of the decoder, while RL and SAC optimize the *exact*
hard quantized deployment decoder under the same trapping-set stress —
recovering the robustness the surrogate gradient cannot see.

| Method | Script | Output weights | Role |
|---|---|---|---|
| NNMS | `train_nnms_clip20_eval75.py` | `model5_nnms_train20_eval75.npz` | Gradient BCE baseline at clip ±20, IS-selected at clip ±7.5 |
| beat-NNMS | `train_beat_nnms.py` | `model5_beatnnms.npz` | Focal reweighting of hard frames, anchored to NNMS |
| IS-NMS | `train_is_driven2.py` | `model5_isdriven2.npz` | Direct minimization of the **importance-sampled FER** (warm-start base for the proposed methods) |
| **Adv-NMS (warm)** | `train_rl_sac_adv_logged_warm.py` | `matlab_logs/advnms_warm.mat` | **Proposed: gradient (FGSM) adversarial attack** on the TS-restricted channel LLRs + robust retraining |
| **RL-NMS (warm)** | `train_rl_sac_adv_logged_warm.py` | `matlab_logs/rlnms_warm.mat` | **RL improvement of the adversarial method**: policy-gradient search on the true quantized decoder |
| **SAC-NMS (warm)** | `train_rl_sac_adv_logged_warm.py` | `matlab_logs/sacnms_warm.mat` | **Maximum-entropy (SAC-style) improvement**: wider stochastic exploration of the weight policy |

Adv / RL / SAC are all *warm-started* from the IS-NMS weights, so they fine-tune
on top of the strongest gradient-trained base. Evaluation and the final curves
come from `mc_is_full_curve.py` (Monte-Carlo waterfall + importance-sampled
floor) and the MATLAB scripts in `result/.../matlab_logs/`.

---

## 1. System model

### 1.1 Code and rate

Working code (1280 example): 5G-NR LDPC, decode length $N = 1280$, $M = 640$
checks, lifting $z = 64$, information length $k = 512$.
The first $128$ bits ($2z$) are **punctured** (never transmitted) and bits
$[512, 640)$ are **shortened** (known zeros), so the transmitted length is
$n_{tx} = N - 256 = 1024$ and the transmitted rate is

$$
R \;=\; \frac{k}{N - 256} \;=\; \frac{512}{1024} \;=\; 0.5 .
$$

(The 896 code used elsewhere in the repo is the same construction with
$N=896$, $k=256$, $z=32$, $R=1/3$.)

### 1.2 Channel and LLRs

All-zero codeword (valid by code linearity and channel symmetry), BPSK mapping
$0 \mapsto +1$. Received samples

$$
y_v = 1 + n_v,\qquad n_v \sim \mathcal N(0, \sigma^2),\qquad
\sigma^2 = \frac{1}{2 R \cdot 10^{E_b/N_0\,[\mathrm{dB}]/10}} .
$$

Channel LLRs (true AWGN scaling):

$$
\lambda_v \;=\; \frac{2 y_v}{\sigma^2}.
$$

Then puncturing/shortening is applied to the LLR vector:
punctured bits get $\lambda_v = 0$ (erasure), shortened bits get
$\lambda_v = +\Lambda$ (perfectly known zero, saturated positive).

### 1.3 Message quantization (Q-NMS)

Every message (channel LLR, V→C, C→V) passes through a uniform quantizer

$$
Q(x) \;=\; \operatorname{clamp}\!\Big(\Delta \cdot \Big\lfloor \tfrac{x}{\Delta} \Big\rceil,\; -\Lambda,\; +\Lambda\Big),
\qquad \Delta = 0.5,
$$

with **deployment clip $\Lambda = 7.5$** (5-bit). During gradient training the
quantizer uses the straight-through estimator (STE):

$$
Q_{\text{STE}}(x) = x + \big(Q(x) - x\big)_{\text{stop-grad}},
\qquad \frac{\partial Q_{\text{STE}}}{\partial x} \equiv 1 .
$$

NNMS is *trained* with forwards at $\Lambda = 20$ (no saturation floor, clean
gradients) but *validated and deployed* at $\Lambda = 7.5$.

---

## 2. Neural normalized min-sum decoder

Tanner graph with edge set $E$; edge $e = (v, c)$. Trainable weights
$w^{(\ell)}_{vn}[e]$ and $w^{(\ell)}_{cn}[e]$ — one pair **per edge per
iteration**, $\ell = 1,\dots,L$ with $L = 15$ iterations (shape $[15 \times |E|]$,
$|E| = 4288$ for the 1280 code).

Per iteration $\ell$:

**VN update** (extrinsic sum, weighted, quantized):

$$
u_{v \to c} \;=\; Q\!\Big( w^{(\ell)}_{vn}[e] \cdot \Big( \lambda_v + \sum_{c' \in \mathcal N(v) \setminus c} m_{c' \to v} \Big) \Big)
$$

**CN update** (min-sum with normalization weight, quantized):

$$
m_{c \to v} \;=\; Q\!\Big( w^{(\ell)}_{cn}[e] \cdot
\Big( \prod_{v' \in \mathcal N(c) \setminus v} \operatorname{sign}(u_{v' \to c}) \Big)
\cdot \min_{v' \in \mathcal N(c) \setminus v} \big| u_{v' \to c} \big| \Big)
$$

**Posterior** after iteration $\ell$:

$$
\Lambda^{(\ell)}_v = \lambda_v + \sum_{c \in \mathcal N(v)} m_{c \to v},
\qquad \hat x_v = \mathbb 1\big[\Lambda^{(L)}_v < 0\big].
$$

Plain MS is the special case $w \equiv 1$.

### 2.1 Differentiable surrogate (soft CN) used for training

The hard min and sign are non-differentiable, so training forwards use:

- **Soft-min magnitude** (temperature $\beta = 5$), excluding the target edge:

$$
\widetilde{\min_{v'}}\, |u_{v'}|
\;=\; -\frac{1}{\beta} \log \sum_{v' \in \mathcal N(c)\setminus v} e^{-\beta |u_{v' \to c}|},
$$

  clipped to be non-negative. In "hybrid" mode the *forward value* is the exact
  hard min and the gradient flows through the soft-min (STE:
  $\text{soft} + (\text{hard} - \text{soft})_{\text{stop-grad}}$).

- **Soft sign** via $\tanh(\gamma\, u)$ with $\gamma = 0.05$ for the
  sign-parity gradient path.

Evaluation always uses the hard min-sum forward (`cn_update_mode="hard"`).

---

## 3. Trapping sets and the error floor

An $(a, b)$ **trapping set (TS)** $T \subset \{1..N\}$ is a set of $a$ variable
nodes inducing $b$ odd-degree (unsatisfied) checks in the subgraph. Small
$(a, b)$ with $b \le 2$ dominate the min-sum error floor.

TSs are found structurally (`tstools`, `run_all_basegraph_traps.py`) and stored
in `BaseGraph/*.found.trap` / `*.dominant.trap`. Training and IS use the pool

$$
\mathcal T = \{T_1, \dots, T_K\},\qquad
b \le 2,\; 3 \le a \le 10,\; T_k \cap \{\text{punctured/shortened}\} = \emptyset .
$$

Indicator vectors $\mathbf 1_{T_k} \in \{0,1\}^N$ (rows of matrix $A$),
$a_k = |T_k|$. The union support defines the per-bit TS mask
$\text{ts}_v = \mathbb 1[v \in \bigcup_k T_k]$ (over the "focus" pool $a \le 6$).

---

## 4. Importance sampling (IS) of the floor

Direct MC cannot reach FER $\sim 10^{-8}$. We sample noise from a
**mean-shifted mixture** centered on the trapping sets:

$$
q(\mathbf n) \;=\; \frac{1}{K} \sum_{k=1}^{K}
\mathcal N\!\big(\mathbf n;\; -s\, \mathbf 1_{T_k},\; \sigma^2 I\big),
$$

i.e. pick $T_k$ uniformly, then draw $\mathbf n = \sigma \mathbf g - s\,\mathbf 1_{T_k}$
(shift $s = 1.2$ by default, pushing the TS bits toward the decision boundary
since the transmitted symbol is $+1$).

The unbiased estimator uses likelihood ratios $w(\mathbf n) = p(\mathbf n)/q(\mathbf n)$
with $p = \mathcal N(0, \sigma^2 I)$. Because only the TS coordinates are
shifted,

$$
w(\mathbf n) \;=\;
\frac{K}{\displaystyle \sum_{k=1}^{K} \exp\!\Big( -\frac{s\, \mathbf n^{\!\top} \mathbf 1_{T_k}}{\sigma^2} \;-\; \frac{a_k s^2}{2\sigma^2} \Big)} ,
$$

computed stably via log-sum-exp. Then over $N_t$ trials

$$
\widehat{\text{FER}} = \frac{1}{N_t} \sum_{i} w_i\, F_i,
\qquad
\widehat{\text{BER}} = \frac{1}{N_t} \sum_{i} w_i\, \frac{B_i}{n_{tx}},
$$

where $F_i = \mathbb 1[\text{any tx bit wrong}]$ and $B_i$ = number of wrong tx
bits. Reliability is monitored by the (failure-restricted) effective sample
size

$$
\text{ESS} = \frac{\big(\sum_i w_i F_i\big)^2}{\sum_i (w_i F_i)^2} \Big/ N_t .
$$

---

## 5. Training objectives

All methods train only $\{w_{vn}, w_{cn}\}$ (the graph and schedule are fixed)
and share two ingredients:

- **Iteration weighting**: multi-loss over all $L$ posteriors with linearly
  increasing weights $\omega_\ell \propto \text{linspace}(0.3, 1.0)$,
  $\sum_\ell \omega_\ell = 1$.
- **Soft bit-error loss**: $\operatorname{softplus}(-\Lambda^{(\ell)}_v) = \log(1 + e^{-\Lambda^{(\ell)}_v})$,
  the BCE of the posterior against the all-zero codeword.

Model selection during training is always **by IS-estimated FER at 4.5 dB at
the deployment clip ±7.5** (not by training loss).

### 5.1 NNMS — gradient baseline (`train_nnms_clip20_eval75.py`)

Batch mixes plain AWGN at $E_b/N_0 \sim \mathcal U(2.5, 4.0)$ dB and a fraction
$\rho_{TS} = 30\%$ of frames with a TS **bias injection**: pick $T \in$ focus
pool, subtract $s \sim \mathcal U(0.8, 1.5)$ on the TS bits,
$y \leftarrow y - s\,\mathbf 1_T$. Loss:

$$
\mathcal L_{\text{NNMS}}
= \sum_{\ell=1}^{L} \omega_\ell \,
\mathbb E_{b, v \in tx} \Big[ \rho_v\, \operatorname{softplus}\!\big(-\Lambda^{(\ell)}_v\big) \Big]
\;+\; \eta \big( \| w_{vn} - 1 \|_2^2 + \| w_{cn} - 1 \|_2^2 \big),
$$

with per-bit upweighting $\rho_v = 2$ on TS-support bits ($\rho_v = 1$ else),
$\eta = 5\cdot 10^{-3}$, Adam lr $2\cdot 10^{-3}$ cosine-annealed, gradient
clip 1.0, weight clamp $w \in [0.5, 1.6]$. Train forwards run at clip ±20;
validation/selection at ±7.5.

### 5.2 beat-NNMS — focal hard-example fine-tune (`train_beat_nnms.py`)

Warm start from NNMS. The problem: at the deployment clip most frames are
already decoded, so the mean BCE gradient vanishes. Fix: **focal frame
weighting** by the worst (most-negative) posterior margin

$$
m_b = \max_{v \in tx} \big( -\Lambda^{(L)}_v \big) ,
\qquad
f_b = \frac{ \sigma(m_b)^{\kappa} }{ \overline{ \sigma(m)^{\kappa} } }
\quad (\text{detached, mean-normalized}),
$$

so frames near/after failure dominate the gradient. Loss:

$$
\mathcal L_{\text{beat}}
= \sum_\ell \omega_\ell\, \mathbb E_b \Big[ f_b \cdot \mathbb E_{v \in tx} \big[ \rho_v \operatorname{softplus}(-\Lambda^{(\ell)}_v) \big] \Big]
+ \eta_a \big( \| w - w^{\text{NNMS}} \|_2^2 \big),
$$

anchored to the NNMS weights instead of to 1 (build on NNMS, don't forget it).

### 5.3 IS-NMS — direct IS-FER minimization (`train_is_driven2.py`)

Warm start from beat-NNMS. Train **on the IS distribution itself**: every step
draws $\sim 12{,}000$ TS-shifted samples ($s \sim \mathcal U(0.9, 1.4)$,
$E_b/N_0 \sim \mathcal U(4.0, 5.5)$ dB) and minimizes the *importance-weighted
soft frame-error rate*:

- Per-sample soft failure via a smooth-max margin (temperatures $T = T_s = 2$):

$$
m_i = T \log \sum_{v \in tx} e^{-\Lambda^{(L)}_{i,v} / T},
\qquad
\widehat F_i = \sigma\!\big( m_i / T_s \big)
$$

- **Truncated, globally-normalized IS weights** (this is what fixed v1's
  ESS $\approx 0.3\%$ divergence):

$$
\tilde w_i = \min\big( w_i,\; \tau \cdot \bar w \big),\quad \tau = 20,
\qquad
\hat w_i = \frac{\tilde w_i}{\sum_j \tilde w_j}
\quad \text{(normalized over the full 12k-sample step, not per micro-batch)} .
$$

- Objective:

$$
\mathcal L_{\text{IS}}
= \sum_i \hat w_i\, \widehat F_i
\;+\; \lambda_{\text{bce}} \sum_\ell \omega_\ell \sum_i \hat w_i\, \mathbb E_{v \in tx}\big[\operatorname{softplus}(-\Lambda^{(\ell)}_{i,v})\big]
\;+\; \eta_a \| w - w^{\text{beat}} \|_2^2 ,
$$

with $\lambda_{\text{bce}} = 0.3$, $\eta_a = 3\cdot10^{-3}$, lr $2\cdot10^{-4}$,
clamp $[0.4, 1.8]$. The first term is a differentiable estimate of the floor
FER itself — the training loss *is* the deployment metric.

### 5.4 Adv-NMS — the proposed gradient adversarial-attack method (warm)

(`train_rl_sac_adv_logged_warm.py`, `train_adv_logged`)

This is the core proposed method: treat the trapping-set failure mechanism as
an **adversary** and train the decoder against the *worst-case* gradient
perturbation of the channel LLRs. Warm start from IS-NMS, soft-CN surrogate
for gradients, 300 Adam steps (lr $10^{-3}$ cosine). Formally we solve the
robust (min–max) problem

$$
\min_{w} \; \mathbb E_{\mathbf y, T} \Big[ \max_{\|\boldsymbol\delta\|_\infty \le \epsilon,\ \operatorname{supp}(\boldsymbol\delta) \subseteq T} \mathcal L\big( w;\ \boldsymbol\lambda(\mathbf y) + \boldsymbol\delta \big) \Big],
$$

where the perturbation budget is confined to the trapping-set support — the
attacker may only push the bits that actually cause the error floor. Each
training step alternates the two players:

**1) Inner attack (signed gradient / FGSM, TS-restricted).** Sample a batch
(40% TS-shifted with $s \sim \mathcal U(1.2, 2.8)$,
$E_b/N_0 \sim \mathcal U(3.5, 5.0)$ dB), form $\boldsymbol\lambda = 2\mathbf y / \sigma^2$,
and compute the attack loss on the final posterior

$$
\mathcal L_0(\boldsymbol\lambda)
= \mathbb E_{v \in tx} \big[ \rho_v\, \operatorname{softplus}\!\big( -\Lambda^{(L)}_v(\boldsymbol\lambda) \big) \big],
\qquad \rho_v = 1 + \text{ts}_v .
$$

The inner maximization over the $\ell_\infty$ ball has the closed-form
one-step solution (Fast Gradient Sign Method), projected onto the TS support:

$$
\boldsymbol\lambda_{\text{adv}}
= \boldsymbol\lambda \;-\; \epsilon \cdot \operatorname{sign}\!\big( \nabla_{\boldsymbol\lambda} \mathcal L_0 \big) \odot \mathbf 1[\text{TS support}],
\qquad \epsilon = 1.5 .
$$

This is exactly the direction that drives the decoder deepest into the
trapping-set failure mode with an LLR budget of $\epsilon$ per bit.

**2) Outer defense (weight update).** Retrain the weights on the attacked
input with the multi-iteration TS-weighted BCE plus an anchor to the IS-NMS
base:

$$
\mathcal L_{\text{Adv}}
= \sum_{\ell=1}^{L} \omega_\ell\,
\mathbb E_{b, v \in tx} \Big[ \rho_v \operatorname{softplus}\!\big( -\Lambda^{(\ell)}_v(\boldsymbol\lambda_{\text{adv}}) \big) \Big]
\;+\; \eta_a \big( \| w - w^{\text{IS}} \|_2^2 \big),
\qquad \eta_a = 2\cdot10^{-3},
$$

with gradient clip 1.0 and clamp $w \in [0.4, 1.8]$. Validation/selection:
hard decoder, IS-FER at 4.5 dB.

**Limitation that motivates RL/SAC.** Both the attack gradient
$\nabla_{\boldsymbol\lambda} \mathcal L_0$ and the defense gradient
$\nabla_w \mathcal L_{\text{Adv}}$ must flow through the *differentiable
surrogate* of §2.1 (soft-min $\beta$, $\tanh\gamma$ sign, STE quantizer). The
deployed decoder is hard and 5-bit quantized, so the surrogate gradient is
biased precisely in the saturated/quantized regime where the floor lives.
The next two methods close this gap.

### 5.5 RL-NMS — reinforcement learning applied to improve the adversarial method (warm)

(`train_rl_sac_adv_logged_warm.py`, `train_es_logged`)

RL-NMS keeps the **same adversarial stress environment** as Adv-NMS — frames
attacked on the trapping-set bits — but replaces the surrogate-gradient
defense with **derivative-free policy optimization of the exact deployment
decoder** (hard min-sum, 5-bit quantized, clip ±7.5). No gradient ever needs
to pass through the decoder, so there is no surrogate bias.

**MDP formulation.** One decoding episode = one environment step:

- **State** $s$: a batch of attacked channel-LLR frames. The attack here is
  the *hard* TS displacement $y \leftarrow y - s\,\mathbf 1_T$ with
  $s \sim \mathcal U(1.2, 2.8)$ on 60% of frames (the black-box counterpart of
  the FGSM step — it needs no gradient, so it stresses the true quantized
  decoder), $E_b/N_0 \sim \mathcal U(3.5, 5.0)$ dB.
- **Action** $a = \theta \in \mathbb R^{15 \times 4}$: a TS-conditioned
  log-multiplier field applied on top of the adversarially-motivated base
  weights (IS-NMS):

$$
w^{(\ell)}_{vn}[e] = w^{\text{IS}, (\ell)}_{vn}[e] \cdot e^{\theta_{\ell,\, g_{vn}(e)}},
\qquad
w^{(\ell)}_{cn}[e] = w^{\text{IS}, (\ell)}_{cn}[e] \cdot e^{\theta_{\ell,\, g_{cn}(e)}},
$$

  where $g(e) \in \{$vn/non-TS, vn/TS, cn/non-TS, cn/TS$\}$ groups each edge
  by whether its variable node lies on a trapping set, and $\theta$ is clamped
  to $[-0.7, 0.7]$. Four numbers per iteration — the policy acts exactly on
  the TS/non-TS split that the adversary exploits.
- **Reward**: mean saturated posterior margin on the **attacked TS bits**
  (the bits the adversary targeted), from the true hard decoder:

$$
R(\theta) = \mathbb E_{(b,v) \in \text{attacked TS}} \Big[ \tanh\!\Big( \frac{\Lambda^{(L)}_{b,v}(\theta)}{4} \Big) \Big] .
$$

  Positive posterior = correct bit; $\tanh(\cdot/4)$ keeps residual gradient
  signal without saturating — this is the negative of the adversary's goal,
  so maximizing $R$ is the defense move of the same min–max game.

**Policy and objective.** The stochastic policy is Gaussian in parameter
space, $\pi_\theta = \mathcal N(\theta, \sigma_e^2 I)$, and we maximize the
smoothed return

$$
J(\theta) = \mathbb E_{\tilde\theta \sim \pi_\theta} \big[ R(\tilde\theta) \big]
= \mathbb E_{\xi \sim \mathcal N(0, I)} \big[ R(\theta + \sigma_e \xi) \big] .
$$

**Policy gradient.** By the log-likelihood-ratio (REINFORCE / score-function)
identity,

$$
\nabla_\theta J(\theta)
= \mathbb E_{\tilde\theta \sim \pi_\theta} \Big[ R(\tilde\theta)\, \nabla_\theta \log \pi_\theta(\tilde\theta) \Big]
= \frac{1}{\sigma_e} \, \mathbb E_{\xi} \big[ R(\theta + \sigma_e \xi)\, \xi \big],
$$

estimated with **antithetic pairs** (the $\pm\xi$ construction cancels the
baseline and halves the variance):

$$
\hat g = \frac{1}{2 P \sigma_e} \sum_{p=1}^{P} \big[ R(\theta + \sigma_e \xi_p) - R(\theta - \sigma_e \xi_p) \big]\, \xi_p,
\qquad \xi_p \sim \mathcal N(0, I),
$$

$$
\theta \leftarrow \operatorname{clamp}\big( \theta + \alpha\, \hat g,\ \pm 0.7 \big).
$$

RL-NMS runs 120 policy-gradient steps with population $P = 8$ pairs,
exploration $\sigma_e = 0.05$, step $\alpha = 0.15$; every 10 steps the
current deterministic policy (mean $\theta$) is evaluated by IS-FER at
4.5 dB and the best checkpoint is kept.

### 5.6 SAC-NMS — maximum-entropy (soft actor-critic style) improvement (warm)

Same environment, action space, and reward as RL-NMS. SAC-NMS follows the
**maximum-entropy RL** principle behind Soft Actor-Critic: instead of a
(near-)deterministic greedy policy, keep the policy deliberately stochastic
and optimize return *plus* policy entropy,

$$
J_{\text{soft}}(\theta)
= \mathbb E_{\tilde\theta \sim \pi_\theta} \big[ R(\tilde\theta) \big]
\;+\; \alpha_H\, \mathcal H\!\big( \pi_\theta \big),
\qquad
\mathcal H\big( \mathcal N(\theta, \sigma_e^2 I) \big) = \tfrac{d}{2} \log\big( 2\pi e\, \sigma_e^2 \big),
$$

so the entropy bonus is controlled by the exploration scale $\sigma_e$ of the
Gaussian policy. In our parameter-space setting the entropy term is constant
in $\theta$ for fixed $\sigma_e$, so maximum-entropy optimization reduces to
**running the same score-function policy gradient with a wider exploration
distribution and a more conservative step**, paid for with a larger
population to keep the gradient variance in check:

| | population $P$ | exploration $\sigma_e$ (entropy) | step $\alpha$ |
|---|---|---|---|
| RL-NMS (greedy) | 8 pairs | 0.05 | 0.15 |
| **SAC-NMS (max-entropy)** | **12 pairs** | **0.08** | **0.10** |

The wider $\sigma_e$ explores flatter, more robust regions of the weight
landscape (the soft objective smooths $R$ over a larger neighborhood —
$J$ is the Gaussian convolution $R * \mathcal N(0, \sigma_e^2 I)$), which is
exactly the property we want against an adversary: a policy that survives
$\pm\sigma_e$ parameter perturbations also survives larger input
perturbations. Both variants are selected by the same IS-FER@4.5 dB
criterion, and both fine-tune the adversarially-motivated IS-NMS base — i.e.
they are *improvements applied to the proposed adversarial method*, not
independent baselines.

---

## 6. Evaluation: stitched MC + IS curves (`mc_is_full_curve.py`)

- **Waterfall (MC)**: $E_b/N_0 = 1.0 : 0.5 : 3.0$ dB, standard Monte-Carlo
  until 300 frame errors (cap 4M frames).
- **Floor (IS)**: $E_b/N_0 = 3.0 : 0.5 : 8.0$ dB, the estimator of §4 with
  300k trials, shift $s = 1.2$, ESS reported per point.
- The 3.0 dB point exists in both — an MC↔IS cross-check.

All seven decoders (MS, NNMS, beat-NNMS, IS-NMS, RL/SAC/Adv-warm) are run with
the same hard quantized decoder at clip ±7.5; results go to
`matlab_logs/full_curve_mc_is.mat`.

MATLAB plotting:

- `plot_full_curve.m` — stitched FER/BER vs $E_b/N_0$ (all 7 curves; RL/SAC/Adv
  sit nearly on top of IS-NMS in the floor, hence the distinct open markers).
- `plot_training_logs.m` — reward/loss, validation FER/BER, and weight
  statistics over training for RL/SAC/Adv.
- `plot_weight_histograms.m` — VN/CN weight histograms for all methods
  (regenerate the data with `export_weights_hist_mat.py`).

---

## 7. Method relationships (pipeline)

```
MS (w = 1)
  └─ NNMS          gradient BCE, train@±20, select by IS@±7.5
       └─ beat-NNMS   focal hard-frame fine-tune, anchored to NNMS
            └─ IS-NMS    minimizes truncated-IS soft-FER directly
                 │
                 └─ Adv-NMS (warm)   PROPOSED: gradient (FGSM) attack on TS
                     │               LLRs + min-max robust retraining
                     │               (defense gradient uses the soft surrogate)
                     │
                     ├─ RL-NMS  (warm)  RL improvement: policy gradient on the
                     │                  TRUE hard quantized decoder, same TS
                     │                  attack environment (P=8, σ=0.05, α=0.15)
                     └─ SAC-NMS (warm)  max-entropy improvement: wider Gaussian
                                        policy, smoother robust optimum
                                        (P=12, σ=0.08, α=0.10)
```

The logic of the proposed chain: the **gradient method is the adversarial
attack** (FGSM on TS-restricted LLRs, §5.4); since its defense update is
limited by the differentiable surrogate of the quantized decoder, **RL (§5.5)
and SAC (§5.6) are applied to improve it** — they optimize the exact
deployment decoder under the same trapping-set attack, using policy gradients
that need no decoder differentiability. All three share the IS-NMS warm start
and the IS-FER selection metric, so the comparison isolates the training
principle, not the starting point.

Key design invariants across all methods:

1. **The deployment decoder is never approximated at eval time** — hard
   min-sum, 5-bit quantization, clip ±7.5.
2. **Selection metric = IS-estimated floor FER**, not training loss.
3. **Trapping sets drive everything**: the training-noise mixture, the IS
   proposal, the per-bit loss weights, the adversary's perturbation support,
   and the RL/SAC policy parameterization (TS vs non-TS edge groups).

---

## 8. File map

```
BaseGraph/*.graph                  parity-check matrices (protograph, lifted)
BaseGraph/*.found.trap             structural trapping-set lists (a,b + VN indices)
BaseGraph/*.dominant.trap          dominant subsets (a>=3, b<=2)
run_all_basegraph_traps.py         TS search over all base graphs
train_nnms_clip20_eval75.py        NNMS
train_beat_nnms.py                 beat-NNMS (v1) / train_beat_nnms_v2.py (v2)
train_is_driven.py / _driven2.py   IS-NMS (v1 diverged; v2 = ESS-fixed, used)
train_rl_sac_adv_logged_warm.py    Adv-NMS (proposed) + RL/SAC improvements (warm-started)
mc_is_full_curve.py                MC+IS curves for all 7 decoders
result/results_today_2026-08-03/1280_R0.50/
  model5_*.npz                     learned weights (w_vn, w_cn [15 x E])
  matlab_logs/*.mat                training logs, warm weights, full curves
  matlab_logs/plot_*.m             MATLAB figures
  export_weights_hist_mat.py       weight-histogram data export
```
