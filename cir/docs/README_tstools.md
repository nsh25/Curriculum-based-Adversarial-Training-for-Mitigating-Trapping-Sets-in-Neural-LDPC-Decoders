# tstools — trapping-set search + importance sampling for 5G LDPC codes

A small, dependency-light (NumPy only) toolkit to

1. **search trapping sets** in a lifted LDPC Tanner graph,
2. estimate each set's **error-floor contribution by importance sampling (IS)**,
3. **verify the estimator against Monte-Carlo (MC)**.

It reads the base graphs already in `BaseGraph/` (the `.graph` files produced by
`qc_txt_to_graph.py` from the 3GPP QC shift matrices).

## What is a trapping set?

For an LDPC code with Tanner graph *G*, an **(a, b) trapping set** *T* is a set of
`a` variable nodes whose induced subgraph has `b` odd-degree (unsatisfied) check
nodes. Small `a` with small `b` — especially *elementary* sets, where every
unsatisfied check has degree 1 — are the sub-graphs an iterative BP decoder gets
stuck in, and they dominate the **error floor**. `b` is exactly the syndrome
weight when precisely the variables of *T* are in error.

## Files

| file | purpose |
|------|---------|
| `tstools/tanner.py`      | load `.graph`, build H / edge structures, syndrome, `(a,b)` signature, 4-cycles |
| `tstools/decoder.py`     | vectorised, batched BP decoder (scaled **min-sum** default, or **SPA**), active-set compaction |
| `tstools/trap_search.py` | `structural_search` (short-cycle seeded expansion) + `decoder_search` (empirical); `.trap` read/write |
| `tstools/estimate.py`    | `monte_carlo`, `importance_sampling` (set-targeted **and** global mean-translation), `ebn0_to_sigma` |
| `run_trap_analysis.py`   | end-to-end driver (search → IS floor → MC verification) |
| `selftest.py`            | fast sanity checks |

## Channel / decoder model

All-zero codeword over BPSK-AWGN (valid for any linear code with a symmetric
decoder, so no generator matrix is needed):

```
transmit x = +1        receive y = x + n,  n ~ N(0, sigma^2)
sigma^2 = 1 / (2 * R * Eb/N0)          LLR = 2*y / sigma^2
```

Punctured coordinates (the first `2z` columns in 5G) can be flagged; their LLR is
forced to 0.

### Fixed-point (quantized) decoding

The decoder can run in fixed point via `quant_step` / `quant_clip`. The **channel
LLRs and every check→var / var→check message** are uniformly quantized to a
mid-tread grid and saturated at `±quant_clip`:

```
q(x) = clip( round(x / step) * step , -clip, +clip )
```

The default in `run_trap_analysis.py` is **step 0.5, clip ±7.5**, i.e. the
alphabet `{-7.5, -7.0, …, +7.5}` — 31 levels, **5-bit signed**, the kind of grid
a hardware min-sum core uses. `quantize()` is exported for direct use. Effect on
the R=0.5 code at Eb/N0 = 1.5 dB:

```
FER float             = 2.47e-1
FER fixed (0.5, 7.5)  = 3.07e-1     # quantization loss, as expected
```

The search → IS → MC-verify flow all honour the setting; at Eb/N0 = 0 dB the
fixed-point restricted-MC and set-IS still agree (`1.15e-3` vs `8.33e-4`). Pass
`--quant-step 0` to fall back to the float reference decoder, or e.g.
`--quant-step 0.25` for 6-bit.

## The two estimators

**Monte-Carlo** draws noise from the true density and counts decoder failures.
Correct, but hopeless in the floor: a `1e-9` event needs ~`1e11` trials.

**Importance sampling** draws noise from a biased density `g` (a mean shift `mu`)
that makes failures common, then de-biases every sample by the likelihood ratio

```
w = f(n)/g(n) = exp( (||mu||^2 - 2 n·mu) / (2 sigma^2) )
FER_hat = mean( w · 1[failure] )
```

which is unbiased for *any* `mu`. The shift is **set-targeted**: `mu = -shift·sigma`
on the trapping set's variable nodes only (`shift ≈ 2.5–3σ`, enough to drive `y`
negative there). This low-dimensional shift is the whole point — a *global* mean
translation over all ~600 coordinates makes `‖mu‖² ≈ 160σ²`, so the likelihood
ratio spans dozens of orders of magnitude and the estimator collapses. IS for
LDPC error floors **must** bias only the trapping-set subspace.

### Verifying IS against MC

You cannot compare set-IS to MC in the deep floor — MC sees nothing there. The
honest check uses the *same event* (`T ⊆ residual error support`) at a **low
SNR** where plain MC still catches it:

* **restricted MC** counts failures whose error support contains `T`;
* **set-IS(→T)** estimates the identical probability by importance sampling.

They must agree within error bars. Measured on the R=0.5 code at Eb/N0 = 0 dB:

```
MC(->T) = 1.97e-3  (59 events, rel 13%)
IS(->T) = 1.25e-3  (26 eff-events, rel 20%)   -> |MC-IS| < 2-sigma band : agree
```

Once validated there, the same IS with a stronger shift reaches the floor
(e.g. ~1e-7 at 3.5 dB) where MC would need ~1e9 trials.

## Run it

```bash
# uses C:\Users\moham\anaconda3\python.exe on this machine
python run_trap_analysis.py \
    BaseGraph/5G_LDPC_R0.50_n_dec640_n512_k256_z32_s257_320.graph \
    --a-max 10 --b-max 4 --top 5 \
    --ebn0 3.5 --verify-ebn0 1.5 \
    --is-trials 40000 --mc-trials 60000
```

Add `--puncture2z --z 32` to model 5G puncturing, and `--decoder-search` to also
harvest trapping sets empirically from BP failures. Results are written next to
the graph as `<name>.found.trap` in the same `(a, b) v1 v2 ...` (1-based) format
as the existing `.trap` files.

## Notes / caveats

* The BP decoder is pure NumPy (batched). It manages ~hundreds of 640-bit words
  per second — fine for search, IS, and moderate MC; not a production C kernel.
* Not every structural `(a,b)` set is decoder-harmful. `decoder_search` and the
  set-targeted IS reveal which sets are *absorbing* (the decoder settles onto
  exactly `T`).
* IS variance is only small when the proposal matches the dominant failure mode:
  narrow set shifts belong in the floor, the global shift near the waterfall.
