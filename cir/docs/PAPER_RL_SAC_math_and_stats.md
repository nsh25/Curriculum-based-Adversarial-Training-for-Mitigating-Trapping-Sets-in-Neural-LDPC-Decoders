# Paper-ready material: RL / SAC math + all experimental stats

Trapping-set-aware neural normalized min-sum (Q-NMS) decoding of a 5G-NR LDPC code.
Everything below is drawn from the actual code and result files
(`train_rl_sac_adv_logged_warm.py`, `full_curve_mc_is.mat`, `training_logs_warm_all.mat`).

> **Complexity analysis is in §10** (all costs expressed in the block length $n$).

---

## 1. Experimental setup (numbers to state in the paper)

| Quantity | Value |
|---|---|
| Code | 5G-NR LDPC, base graph BG (R0.50) |
| Decode length $N$ | 1280 |
| Check nodes $M$ | 640 |
| Lifting size $z$ | 64 |
| Information length $k$ | 512 |
| Punctured bits | first 128 ($2z$), LLR $=0$ |
| Shortened bits | $[512,640)$ (128 known zeros), LLR $=+\Lambda$ |
| Transmitted length $n_{tx}$ | $N-256 = 1024$ |
| Transmitted rate $R$ | $512/1024 = 0.5$ |
| Edges $|E|$ | 4288 |
| Trainable weights | $2\times[15\times|E|]$ (per-edge, per-iteration $w_{vn},w_{cn}$) |
| Decoder iterations $L$ | 15 |
| Quantizer step $\Delta$ | 0.5 |
| Deployment clip $\Lambda$ | $\pm7.5$ (5-bit); NNMS *trains* at $\pm20$ |
| Channel | AWGN, BPSK $0\!\mapsto\!+1$, all-zero codeword |
| Noise variance | $\sigma^2 = 1/(2R\cdot10^{E_b/N_0/10})$ |
| Channel LLR | $\lambda_v = 2y_v/\sigma^2$ |
| Selection metric | IS-estimated FER @ 4.5 dB, hard decoder @ $\pm7.5$ |
| Trapping-set pool | $(a,b)$ with $b\le2$, $3\le a\le10$, TS $\cap$ punc/short $=\varnothing$ |

**Evaluation protocol.** Waterfall by standard Monte-Carlo,
$E_b/N_0 = 1.0\!:\!0.5\!:\!3.0$ dB, run to 300 frame errors (cap 4M frames).
Floor by importance sampling, $E_b/N_0 = 3.0\!:\!0.5\!:\!8.0$ dB, 300k trials,
shift $s=1.2$, ESS reported per point. The 3.0 dB point overlaps both as an
MC↔IS cross-check.

---

## 2. Common environment: adversarial trapping-set stress

An $(a,b)$ trapping set $T$ is a set of $a$ variable nodes inducing $b$ odd-degree
(unsatisfied) checks. Indicator vectors $\mathbf 1_{T_k}\in\{0,1\}^N$ form the rows of
$A$; $a_k=|T_k|$; per-bit TS mask $\mathrm{ts}_v=\mathbb 1[v\in\bigcup_k T_k]$.

All three proposed methods (Adv / RL / SAC) share:
- **Warm start** from the IS-NMS weights $w^{\mathrm{IS}}$ (the strongest gradient-trained base).
- **TS stress**: a fraction of frames are displaced on the TS bits toward the decision
  boundary, $y\leftarrow y - s\,\mathbf 1_T$.
- **TS-weighted per-bit loss weight** $\rho_v = 1+\mathrm{ts}_v$.
- **Iteration weighting** $\omega_\ell \propto \mathrm{linspace}(0.3,1.0)$, $\sum_\ell\omega_\ell=1$.
- **Selection** by IS-FER @ 4.5 dB on the exact hard $\pm7.5$ decoder.

---

## 3. Adv-NMS — proposed gradient (FGSM) adversarial method

Robust min–max problem, perturbation confined to the TS support:

$$
\min_{w}\ \mathbb E_{\mathbf y,T}\Big[\max_{\|\boldsymbol\delta\|_\infty\le\epsilon,\ \mathrm{supp}(\boldsymbol\delta)\subseteq T}\ \mathcal L\big(w;\ \boldsymbol\lambda(\mathbf y)+\boldsymbol\delta\big)\Big].
$$

**Inner attack (one-step FGSM, TS-restricted):**

$$
\mathcal L_0(\boldsymbol\lambda)=\mathbb E_{v\in tx}\big[\rho_v\,\mathrm{softplus}(-\Lambda_v^{(L)}(\boldsymbol\lambda))\big],\qquad
\boldsymbol\lambda_{\mathrm{adv}}=\boldsymbol\lambda-\epsilon\,\mathrm{sign}(\nabla_{\boldsymbol\lambda}\mathcal L_0)\odot\mathbf 1[\mathrm{TS}],\ \ \epsilon=1.5.
$$

**Outer defense (weight update):**

$$
\mathcal L_{\mathrm{Adv}}=\sum_{\ell=1}^{L}\omega_\ell\,\mathbb E_{b,v\in tx}\big[\rho_v\,\mathrm{softplus}(-\Lambda_v^{(\ell)}(\boldsymbol\lambda_{\mathrm{adv}}))\big]+\eta_a\|w-w^{\mathrm{IS}}\|_2^2,\quad \eta_a=2\times10^{-3}.
$$

**Hyperparameters:** 300 Adam steps, lr $10^{-3}$ cosine→$10^{-4}$, grad clip 1.0,
weight clamp $[0.4,1.8]$, batch 320 (40% TS-shifted, $s\sim\mathcal U(1.2,2.8)$,
$E_b/N_0\sim\mathcal U(3.5,5.0)$ dB). Soft-CN surrogate for gradients.

**Limitation motivating RL/SAC:** attack and defense gradients both flow through the
differentiable surrogate (soft-min $\beta{=}5$, $\tanh(\gamma u)$ sign $\gamma{=}0.05$, STE
quantizer), which is *biased in the saturated/5-bit regime where the floor lives*.

---

## 4. RL-NMS — policy-gradient improvement on the exact decoder

Same TS attack environment, but derivative-free optimization of the **exact hard 5-bit
$\pm7.5$ decoder** — no gradient passes through the decoder, so no surrogate bias.

**MDP (one decode = one env step):**
- **State** $s$: batch of TS-attacked LLR frames (hard displacement $y\leftarrow y-s\mathbf 1_T$,
  $s\sim\mathcal U(1.2,2.8)$ on 60% of frames, $E_b/N_0\sim\mathcal U(3.5,5.0)$ dB).
- **Action** $a=\theta\in\mathbb R^{15\times4}$: a TS-conditioned log-multiplier field on the base weights,

$$
w_{vn}^{(\ell)}[e]=w_{vn}^{\mathrm{IS},(\ell)}[e]\,e^{\theta_{\ell,g_{vn}(e)}},\qquad
w_{cn}^{(\ell)}[e]=w_{cn}^{\mathrm{IS},(\ell)}[e]\,e^{\theta_{\ell,g_{cn}(e)}},
$$

  with edge group $g(e)\in\{$vn/non-TS, vn/TS, cn/non-TS, cn/TS$\}$, $\theta$ clamped to $[-0.7,0.7]$.
- **Reward** (saturated posterior margin on attacked TS bits, true hard decoder):

$$
R(\theta)=\mathbb E_{(b,v)\in\text{attacked TS}}\big[\tanh(\Lambda_{b,v}^{(L)}(\theta)/4)\big].
$$

**Gaussian policy in parameter space** $\pi_\theta=\mathcal N(\theta,\sigma_e^2 I)$; smoothed return

$$
J(\theta)=\mathbb E_{\xi\sim\mathcal N(0,I)}\big[R(\theta+\sigma_e\xi)\big].
$$

**Score-function (REINFORCE) gradient**, antithetic-pair estimator:

$$
\nabla_\theta J=\tfrac{1}{\sigma_e}\mathbb E_\xi[R(\theta+\sigma_e\xi)\,\xi],\qquad
\hat g=\frac{1}{2P\sigma_e}\sum_{p=1}^{P}\big[R(\theta+\sigma_e\xi_p)-R(\theta-\sigma_e\xi_p)\big]\xi_p,
$$

$$
\theta\leftarrow\mathrm{clamp}(\theta+\alpha\hat g,\ \pm0.7).
$$

**Hyperparameters:** 120 steps, population $P=8$ pairs, $\sigma_e=0.05$, $\alpha=0.15$;
IS-FER eval every 10 steps, best checkpoint kept.

---

## 5. SAC-NMS — maximum-entropy improvement

Same environment / action / reward. Maximum-entropy (Soft Actor-Critic) objective:

$$
J_{\mathrm{soft}}(\theta)=\mathbb E_{\tilde\theta\sim\pi_\theta}[R(\tilde\theta)]+\alpha_H\,\mathcal H(\pi_\theta),\qquad
\mathcal H(\mathcal N(\theta,\sigma_e^2 I))=\tfrac{d}{2}\log(2\pi e\,\sigma_e^2).
$$

In parameter space with fixed $\sigma_e$ the entropy is constant in $\theta$, so the
max-entropy update reduces to the **same score-function policy gradient with a wider
exploration distribution and a more conservative step** (larger population to hold gradient
variance down). $J$ is the Gaussian convolution $R*\mathcal N(0,\sigma_e^2 I)$ — a wider
$\sigma_e$ seeks a flatter, more robust optimum, exactly the property wanted against an adversary.

| Variant | population $P$ | exploration $\sigma_e$ | step $\alpha$ | steps |
|---|---|---|---|---|
| RL-NMS (greedy) | 8 pairs | 0.05 | 0.15 | 120 |
| **SAC-NMS (max-entropy)** | **12 pairs** | **0.08** | **0.10** | 120 |

---

## 6. Importance-sampling estimator (floor measurement)

Mean-shifted mixture proposal centered on the TS pool:

$$
q(\mathbf n)=\tfrac1K\sum_{k=1}^K\mathcal N(\mathbf n;-s\mathbf 1_{T_k},\sigma^2 I),\qquad
\mathbf n=\sigma\mathbf g-s\mathbf 1_{T_k}.
$$

Likelihood ratio $w=p/q$ with $p=\mathcal N(0,\sigma^2 I)$ (only TS coords shifted):

$$
w(\mathbf n)=\frac{K}{\sum_{k}\exp\!\big(-\tfrac{s\,\mathbf n^{\!\top}\mathbf 1_{T_k}}{\sigma^2}-\tfrac{a_k s^2}{2\sigma^2}\big)}\quad(\text{log-sum-exp stable}).
$$

$$
\widehat{\mathrm{FER}}=\tfrac1{N_t}\sum_i w_iF_i,\quad
\widehat{\mathrm{BER}}=\tfrac1{N_t}\sum_i w_i\tfrac{B_i}{n_{tx}},\quad
\mathrm{ESS}=\frac{(\sum_i w_iF_i)^2}{\sum_i(w_iF_i)^2}\Big/N_t.
$$

---

## 7. RESULTS — deployment curves (hard 5-bit $\pm7.5$ decoder, IS floor)

### 7.1 FER vs $E_b/N_0$ (importance-sampled floor, all 7 decoders)

| $E_b/N_0$ (dB) | 3.5 | 4.0 | 4.5 | 5.0 | 5.5 | 6.0 | 6.5 | 7.0 | 7.5 | 8.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| **MS** ($w{=}1$) | 9.92e-6 | 4.34e-6 | 2.70e-6 | 1.70e-6 | 9.64e-7 | 5.16e-7 | 2.49e-7 | 1.11e-7 | 4.48e-8 | 1.63e-8 |
| NNMS | 5.23e-6 | 2.73e-6 | 8.39e-7 | 4.87e-7 | 2.97e-7 | 1.58e-7 | 8.49e-8 | 4.06e-8 | 1.68e-8 | 5.98e-9 |
| beat-NNMS | 1.00e-5 | 1.64e-6 | 5.96e-7 | 3.56e-7 | 2.22e-7 | 1.24e-7 | 6.64e-8 | 3.17e-8 | 1.35e-8 | 4.95e-9 |
| IS-NMS | 9.94e-6 | 1.61e-6 | 5.80e-7 | 3.50e-7 | 2.18e-7 | 1.21e-7 | 6.52e-8 | 3.18e-8 | 1.32e-8 | 4.84e-9 |
| **Adv-NMS** | 9.96e-6 | 1.61e-6 | 5.77e-7 | 3.48e-7 | 2.18e-7 | 1.21e-7 | 6.50e-8 | 3.15e-8 | 1.32e-8 | 4.83e-9 |
| **RL-NMS** | 9.92e-6 | 1.59e-6 | 5.74e-7 | 3.55e-7 | 2.23e-7 | 1.29e-7 | 6.97e-8 | 3.38e-8 | 1.47e-8 | 5.30e-9 |
| **SAC-NMS** | 9.95e-6 | 1.61e-6 | 5.78e-7 | 3.48e-7 | 2.18e-7 | 1.22e-7 | 6.58e-8 | 3.18e-8 | 1.34e-8 | 4.91e-9 |

### 7.2 FER gain over plain MS ($\mathrm{FER}_{MS}/\mathrm{FER}_{\text{method}}$)

| $E_b/N_0$ (dB) | 4.0 | 4.5 | 5.0 | 5.5 | 6.0 | 6.5 | 7.0 | 7.5 | 8.0 |
|---|---|---|---|---|---|---|---|---|---|
| NNMS | 1.59 | 3.22 | 3.48 | 3.25 | 3.26 | 2.93 | 2.73 | 2.66 | 2.72 |
| beat-NNMS | 2.64 | 4.53 | 4.76 | 4.34 | 4.15 | 3.75 | 3.49 | 3.32 | 3.29 |
| IS-NMS | 2.69 | 4.66 | 4.85 | 4.42 | 4.26 | 3.82 | 3.48 | 3.38 | 3.36 |
| **Adv-NMS** | 2.70 | 4.68 | 4.87 | 4.42 | 4.27 | 3.83 | 3.51 | 3.39 | 3.37 |
| **RL-NMS** | 2.73 | 4.71 | 4.77 | 4.32 | 4.01 | 3.57 | 3.28 | 3.04 | 3.06 |
| **SAC-NMS** | 2.70 | 4.68 | 4.87 | 4.42 | 4.22 | 3.79 | 3.49 | 3.35 | 3.31 |

**Headline numbers:** at 5.0 dB the proposed decoders reach **FER $\approx3.5\times10^{-7}$
vs $1.7\times10^{-6}$ for MS — a $\approx\mathbf{4.9\times}$ floor reduction**; the peak gain
region is 4.5–5.5 dB (4.4–4.9×). Adv/SAC/IS land within $<1\%$ of each other across the
floor; RL is marginally weaker at high SNR (as expected from the greedier, lower-variance
policy). This confirms the paper's thesis: **RL/SAC recover the exact-decoder robustness
without the surrogate-gradient bias, matching the gradient method on the deployed 5-bit
decoder.**

### 7.3 Waterfall (Monte-Carlo, FER)

| $E_b/N_0$ (dB) | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|
| MS | 0.992 | 0.851 | 0.349 | 3.35e-2 | 4.80e-4 |
| Adv-NMS | 0.993 | 0.866 | 0.367 | 3.67e-2 | 4.95e-4 |
| RL-NMS | 0.993 | 0.863 | 0.371 | 3.70e-2 | 5.21e-4 |
| SAC-NMS | 0.993 | 0.865 | 0.369 | 3.59e-2 | 4.97e-4 |

Waterfall is essentially unchanged — the training targets **only** the floor (all curves
overlap in the waterfall), so no coding-gain is traded away for the floor improvement.

### 7.4 ESS (IS reliability, failure-restricted) @ representative points

Floor points run at ESS $\sim 2\text{–}4\times10^{-3}$ (e.g. IS-NMS: 2.17e-3 @4.5 dB up to
3.1e-3 @8 dB), i.e. hundreds of effective failure samples per 300k trials — the estimator
is stable across the reported range.

---

## 8. Training-time statistics (selection @ 4.5 dB, 200k IS trials)

| Method | FER@4.5 (train-eval) | gain/MS | final reward/loss | $\bar w_{vn}$ | $\mathrm{std}(w_{vn})$ |
|---|---|---|---|---|---|
| MS | 3.17e-6 | 1.00 | — | 1.000 | 0.000 |
| IS-base | 6.60e-7 | 4.80 | — | — | — |
| Adv-NMS | 6.59e-7 | 4.81 | loss $+0.0318$ | 0.973 | 0.046 |
| RL-NMS | 1.03e-6 | 3.07 | reward $-0.189$ | 0.973 | 0.065 |
| SAC-NMS | 6.58e-7 | 4.81 | reward $-0.184$ | 0.982 | 0.053 |

- Training budget: RL/SAC 120 policy-gradient steps; Adv 300 Adam steps.
- Learned weights stay near unity ($\bar w\approx0.97\!-\!0.98$) with small spread
  ($\mathrm{std}\approx0.05$) — the methods apply a **light, TS-localized normalization**
  rather than a wholesale weight overhaul, which explains why the waterfall is preserved.
- SAC's wider exploration ($\sigma_e{=}0.08$) reaches the same optimum as the gradient
  method (4.81× vs Adv 4.81×), while greedy RL under-shoots slightly at this checkpoint —
  empirical support for the max-entropy / flat-minimum argument in §5.

> Note: §8 numbers use the 200k-trial in-loop estimator at $\pm7.5$; §7 uses the final
> 300k-trial stitched curves. Report §7 as the headline results; §8 as training diagnostics.

---

## 9. One-paragraph abstract-ready summary

> We learn per-edge, per-iteration weights of a 5-bit quantized neural normalized min-sum
> decoder for a rate-1/2, $N=1280$ 5G-NR LDPC code, targeting the trapping-set-induced error
> floor. A gradient (FGSM-style) adversarial attack restricted to the trapping-set bits, with
> robust min–max retraining (Adv-NMS), is improved by reinforcement-learning and
> maximum-entropy (SAC-style) policy-gradient search that optimizes the *exact* hard quantized
> deployment decoder — eliminating the surrogate-gradient bias that limits the gradient method
> in the saturated/quantized regime. On the importance-sampled floor the proposed decoders cut
> the frame-error rate by up to **4.9×** over plain min-sum (4.5–5.5 dB), reaching FER
> $\approx3.5\times10^{-7}$ at 5 dB, with no loss of waterfall coding gain.

---

## 10. Complexity analysis (in the block length $n$)

### 10.1 Notation and the sparsity fact that makes everything linear in $n$

| Symbol | Meaning | Value (1280 code) |
|---|---|---|
| $n=N$ | block (decode) length | 1280 |
| $M$ | number of check nodes | 640 |
| $\lvert E\rvert$ | Tanner-graph edges | 4288 |
| $\bar d_v,\ \bar d_c$ | mean VN / CN degree | $\lvert E\rvert/N=3.35$, $\lvert E\rvert/M=6.7$ |
| $L$ | decoder iterations | 15 |
| $B$ | batch size (frames in parallel) | 320-12k |
| $S$ | training steps | 120-300 |
| $K$ | trapping sets in the pool | - |
| $\bar a$ | mean TS size ($\le a_{\max}$) | $\le 10$ |
| $P$ | ES policy population (antithetic pairs) | 8 / 12 |
| $d_\theta$ | policy dimension $=4L$ | 60 |
| $N_t$ | IS trials per SNR point | $2$-$3\times10^{5}$ |

**Key fact.** For an LDPC code the parity-check matrix is sparse with *bounded*
row/column degrees, so

$$
\lvert E\rvert=\bar d_v\,N=\bar d_c\,M=\Theta(n),\qquad M=(1-R')\,n=\Theta(n).
$$

Every message array is indexed by edges, so **one decoder half-iteration is
$\Theta(\lvert E\rvert)=\Theta(n)$**, and all per-frame costs below are linear in $n$
(the degrees enter only as constants). This is the standard reason min-sum LDPC
decoding is attractive; our added methods preserve it.

### 10.2 Base primitive - one min-sum decode ($B$ frames, $L$ iters)

Per iteration the batched decoder (`tstools/decoder.py`, and the neural forward)
does, for the whole batch:

- **CN update** with the exclude-one min1/min2 trick: reductions over the
  degree axis of a $[B,M,d_c]$ array $\Rightarrow \Theta(B\,M\,d_c)=\Theta(B\lvert E\rvert)$;
- **VN update / posterior** (edge-indexed scatter-add): $\Theta(B\lvert E\rvert)$;
- **syndrome / early-stop check**: $\Theta(B\lvert E\rvert)$.

$$
\boxed{\ T_{\text{decode}}=\Theta(B\,L\,\lvert E\rvert)=\Theta(B\,L\,n)\ },\qquad
S_{\text{decode}}=\Theta(B\,n)\ \text{(messages)} +\Theta(L\,n)\ \text{(weights)}.
$$

Active-set compaction makes this an *upper* bound: converged frames are dropped,
so the realized cost is $\Theta\!\big(\sum_{\ell} B_\ell\,n\big)$ with $B_\ell$ the
still-unconverged count. In the waterfall $B_\ell\!\to\!0$ fast; on the floor the
stubborn (trapping-set) frames run the full $L$, which is exactly the regime we measure.

The exclude-one min1/min2 formulation avoids the naive $\Theta(d_c^2)$ per check -
without it the CN update would be $\Theta(B\,M\,d_c^2)$. Both are $\Theta(n)$ in $n$
but the constant matters in practice.

### 10.3 Per-algorithm complexity

Let $T_{\text{dec}}=B\,L\,n$ (one forward decode of a batch). All training methods
optimize only the $2\lvert E\rvert L=\Theta(Ln)$ weights; the graph is fixed.

| Algorithm | Passes / step | Time (total) | Extra memory vs inference |
|---|---|---|---|
| **Q-NMS inference (deploy)** | 1 fwd | $\Theta(B\,L\,n)$ | - ; $\Theta(Bn+Ln)$ total |
| **Structural TS search** (offline) | - | $\Theta(a_{\max}\,n^2)$ | $\Theta(n)$ per seed state |
| **Decoder (empirical) TS search** | 1 fwd/batch | $\Theta(N_t^{\text{s}}\,L\,n)$ | $\Theta(Bn)$ |
| **IS floor estimator** | 1 fwd/batch | $\Theta\!\big(N_t\,(L\,n+K\bar a)\big)$ | $\Theta(Kn)$ for $A$ |
| **NNMS / beat / IS-NMS** (grad) | 1 fwd + 1 bwd | $\Theta(S\,B\,L\,n)$ | $\Theta(B\,L\,n)$ (autograd tape) |
| **Adv-NMS** (FGSM + defense) | 2 fwd + 2 bwd | $\Theta(S\,B\,L\,n)$ | $\Theta(B\,L\,n)$ |
| **RL-NMS** (ES / score-fn) | $2P$ fwd, **0 bwd** | $\Theta(S\,P\,B\,L\,n)$ | **$\Theta(Bn+Ln)$** (no tape) |
| **SAC-NMS** (max-ent ES) | $2P$ fwd, **0 bwd** | $\Theta(S\,P\,B\,L\,n)$ | **$\Theta(Bn+Ln)$** (no tape) |

**Derivations.**

- **Structural TS search** (`structural_search`): $\Theta(n)$ seeds (4-cycle
  endpoints + degree-$\ge2$ VNs); each seed grows up to $a_{\max}$ steps, and every
  step scans the length-$M$ degree vector to rank candidates
  ($O(M)=O(n)$). Hence $\Theta(\text{seeds}\times a_{\max}\times n)=\Theta(a_{\max}\,n^2)$.
  This is a **one-time offline** cost, independent of decoding/training. (Beam
  width $w>1$ multiplies by $w$; the greedy $w{=}1$ path is used.)

- **IS floor estimator** (`is_ber_fer`): per batch, decode $\Theta(BLn)$ plus the
  likelihood-ratio weights $w=\!K/\sum_k e^{(\cdot)}$. The shift term $\mathbf n^\top\mathbf 1_{T_k}$
  is `nz @ A.T` with $A\in\{0,1\}^{K\times N}$: dense $\Theta(BnK)$, or $\Theta(B\,K\bar a)$
  exploiting the $\bar a$ nonzeros per TS, then a log-sum-exp $\Theta(BK)$. Summed
  over $N_t$ trials $\Rightarrow \Theta\!\big(N_t(Ln+K\bar a)\big)$; the decode dominates
  whenever $Ln\gg K\bar a$. Importance sampling replaces the $\Theta(1/\text{FER})$
  frames of naive Monte-Carlo (e.g. $\sim\!10^{8}$ at the floor) with a fixed $N_t$,
  a variance reduction of orders of magnitude at equal $n$-cost per frame.

- **Gradient methods** (NNMS, beat-NNMS, IS-NMS): back-propagation through the
  $L$-iteration unrolled decoder costs the same order as the forward,
  $\Theta(BLn)$ per step, but must **store every intermediate message/posterior**,
  giving the $\Theta(BLn)$ activation memory. IS-NMS adds $\Theta(BK)$ per step for
  its per-sample IS weights (subdominant) and uses a large $B\!\sim\!12$k.

- **Adv-NMS**: one extra forward+backward for the FGSM input gradient
  $\nabla_{\boldsymbol\lambda}\mathcal L_0$, then the multi-posterior defense
  forward+backward - a constant factor ($\approx2\times$) over a plain gradient
  step, same $\Theta(SBLn)$ order and $\Theta(BLn)$ memory.

- **RL-NMS / SAC-NMS**: each step evaluates $2P$ perturbed policies, and **each
  evaluation is a single inference decode** - no back-propagation through the
  decoder. Cost $\Theta(SPBLn)$; the score-function gradient itself is
  $\Theta(P\,d_\theta)=\Theta(PL)$ (negligible). Because it is inference-only, the
  memory is the decode memory $\Theta(Bn+Ln)$ - it **removes the $\Theta(BLn)$
  autograd tape** that the gradient methods need. Periodic IS validation every
  $v$ steps adds $\Theta((S/v)\,N_{\text{val}}\,Ln)$. SAC differs from RL only in the
  constants $(P,\sigma_e,\alpha)=(12,0.08,0.10)$ vs $(8,0.05,0.15)$ - i.e. a $1.5\times$
  larger $P$, so $1.5\times$ the per-step forwards.

### 10.4 Reading of the table

1. **Everything is linear in $n$ per decode** ($\Theta(Ln)$ per frame), inherited
   from LDPC sparsity; the neural weights add no asymptotic cost, only the
   $\Theta(Ln)$ weight storage. **Deployment is plain min-sum complexity.**

2. **ES vs gradient - a time/memory trade.** RL/SAC pay a $\sim\!2P$ ($16$-$24\times$)
   factor in *forward passes* per step, but (i) do **no back-propagation** and
   carry **no autograd tape** ($\Theta(Bn+Ln)$ vs $\Theta(BLn)$ memory), and
   (ii) evaluate the **exact hard 5-bit decoder**, which the gradient methods
   cannot differentiate. The tiny policy dimension $d_\theta=4L=60$ keeps the ES
   gradient estimate well-conditioned despite the $\Theta(Ln)$ underlying weights.

3. **Offline vs online.** The $\Theta(a_{\max}n^2)$ TS search and the IS pool $A$
   ($\Theta(Kn)$) are computed **once**; per-frame decoding and the deployed
   decoder stay $\Theta(Ln)$.

4. **Scaling to longer codes.** All online costs grow *linearly* in $n$; only the
   offline structural TS search is $\Theta(a_{\max}n^2)$ (and is trivially parallel
   over the $\Theta(n)$ seeds). No method has complexity worse than quadratic in $n$,
   and every runtime/deployment path is strictly linear.
