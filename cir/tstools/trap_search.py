"""Trapping-set search for LDPC Tanner graphs.

An (a, b) trapping set T is a set of ``a`` variable nodes whose induced
subgraph in the Tanner graph has exactly ``b`` odd-degree (unsatisfied) check
nodes. Small ``a`` with small ``b`` (especially elementary sets, where every
unsatisfied check has degree 1) are the structures that dominate the LDPC
error floor.

Two complementary finders are provided:

* ``structural_search`` – deterministic graph search. Seeds from short cycles
  and greedily grows each seed, at every step adding the variable node that
  removes the most unsatisfied checks, recording every (a, b) with b <= b_max.

* ``decoder_search`` – empirical. Runs the BP decoder in the error-floor
  regime (optionally mean-shift biased to hit failures fast) and records the
  support of each residual error pattern; those supports *are* the trapping
  sets the decoder actually gets stuck in.

Both return canonicalised, de-duplicated ``TrapSet`` objects, and this module
reads/writes the project's ``(a, b) v1 v2 ...`` ``.trap`` files (1-based VNs).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .decoder import decode
from .tanner import Tanner


@dataclass(frozen=True)
class TrapSet:
    vns: tuple[int, ...]          # sorted 0-based variable nodes
    a: int
    b: int
    source: str = ""

    @staticmethod
    def make(t: Tanner, vns, source: str = "") -> "TrapSet":
        s = tuple(sorted(set(int(v) for v in vns)))
        a, b = t.trap_signature(s)
        return TrapSet(vns=s, a=a, b=b, source=source)


# ---------------------------------------------------------------- structural
def _is_trivial(t: Tanner, S: set[int]) -> bool:
    """Reject sets made only of degree-1 variable nodes (parity columns).

    A lone degree-1 VN is a formal (1, 1) 'trapping set' but the decoder never
    gets stuck on it, so it is noise for error-floor purposes.
    """
    return all(len(t.vn_to_cn[v]) <= 1 for v in S)


def _candidate_scores(t: Tanner, S: set[int], deg: np.ndarray) -> dict[int, int]:
    """Score neighbour VNs by how much they reduce the odd-check count b."""
    odd = np.nonzero(deg & 1)[0]
    cand: dict[int, int] = {}
    for c in odd:
        for v in t.cn_to_vn[c]:
            v = int(v)
            if v in S:
                continue
            nb = t.vn_to_cn[v]
            # net Δb if we add v: -(odd nbrs) + (even nbrs)
            net = int((deg[nb] & 1).sum()) - int((~(deg[nb] & 1).astype(bool)).sum())
            cand[v] = max(cand.get(v, -(10 ** 9)), net)
    return cand


def _grow_from(
    t: Tanner, seed, a_max: int, b_max: int, a_min: int, beam: int = 1,
) -> list[TrapSet]:
    """Expand seed(s): greedy (beam=1) or beam-search branching (beam>1).

    At every size, keep up to ``beam`` partial sets ranked by lowest current b
    (then best next-step score). Records every induced (a,b) seen along the way.
    """
    found: list[TrapSet] = []
    # state = (frozenset S, deg copy)
    S0 = set(int(v) for v in seed)
    deg0 = np.zeros(t.M, dtype=np.int64)
    for v in S0:
        deg0[t.vn_to_cn[v]] += 1
    frontier: list[tuple[frozenset[int], np.ndarray]] = [(frozenset(S0), deg0)]
    seen_partial: set[frozenset[int]] = set()

    while frontier:
        nxt: list[tuple[int, int, frozenset[int], np.ndarray, int]] = []
        # rank key for keeping beam: (b, -best_cand_score, ...)
        for S_f, deg in frontier:
            if S_f in seen_partial:
                continue
            seen_partial.add(S_f)
            S = set(S_f)
            b = int((deg & 1).sum())
            if b <= b_max and len(S) >= a_min and not _is_trivial(t, S):
                found.append(TrapSet.make(t, S, source="structural"))
            if len(S) >= a_max or b == 0:
                continue
            cand = _candidate_scores(t, S, deg)
            if not cand:
                continue
            # try the top-`beam` additions from this partial set
            ranked = sorted(cand.items(), key=lambda kv: -kv[1])[: max(1, beam)]
            for v, score in ranked:
                deg2 = deg.copy()
                deg2[t.vn_to_cn[v]] += 1
                S2 = frozenset(S | {v})
                b2 = int((deg2 & 1).sum())
                nxt.append((b2, -score, S2, deg2, v))
        if not nxt:
            break
        # keep global beam of best partials
        nxt.sort(key=lambda x: (x[0], x[1]))
        frontier = []
        kept: set[frozenset[int]] = set()
        for b2, _neg, S2, deg2, _v in nxt:
            if S2 in kept:
                continue
            kept.add(S2)
            frontier.append((S2, deg2))
            if len(frontier) >= max(1, beam):
                break
    return found


def structural_search(
    t: Tanner,
    a_max: int = 12,
    b_max: int = 4,
    a_min: int = 1,
    max_seeds: int | None = None,
    seed_pairs: bool = True,
    beam: int = 1,
) -> list[TrapSet]:
    """Deterministic short-cycle-seeded trapping-set search.

    Records every induced (a, b) with ``a_min <= a <= a_max`` and ``b <= b_max``
    encountered while growing each seed. ``beam>1`` explores multiple expansions
    per seed (deeper coverage of near-dominant sets). Trivial degree-1-only sets
    are dropped. Seeds are 4-cycle endpoints plus each degree>=2 variable node.
    """
    seeds: list[tuple[int, ...]] = []
    if seed_pairs:
        seeds.extend(t.four_cycles())            # endpoints of 4-cycles
    dv = np.array([len(t.vn_to_cn[v]) for v in range(t.N)])
    for v in np.argsort(dv):                      # low-degree VNs first
        if dv[v] >= 2:                            # skip trivial parity columns
            seeds.append((int(v),))
    if max_seeds is not None:
        seeds = seeds[:max_seeds]

    best: dict[tuple[int, ...], TrapSet] = {}
    for seed in seeds:
        for ts in _grow_from(t, seed, a_max, b_max, a_min, beam=beam):
            if ts.a == 0:
                continue
            best[ts.vns] = ts
    return _rank(list(best.values()))


# ---------------------------------------------------------------- decoder
def decoder_search(
    t: Tanner,
    sigma: float,
    trials: int,
    batch: int = 2000,
    max_iter: int = 50,
    rule: str = "minsum",
    alpha: float = 1.0,
    a_max: int = 30,
    b_max: int = 8,
    rng: np.random.Generator | None = None,
    punctured_mask: np.ndarray | None = None,
    quant_step: float | None = None,
    quant_clip: float = 7.5,
) -> list[TrapSet]:
    """Collect residual-error supports from BP failures in the floor regime."""
    rng = rng or np.random.default_rng(0)
    sigma2 = sigma * sigma
    found: dict[tuple[int, ...], TrapSet] = {}
    done = 0
    while done < trials:
        B = min(batch, trials - done)
        # all-zero -> transmit +1; y = 1 + noise
        y = 1.0 + rng.normal(0.0, sigma, size=(B, t.N))
        Lch = 2.0 * y / sigma2
        if punctured_mask is not None:
            Lch[:, punctured_mask] = 0.0
        res = decode(t, Lch, max_iter=max_iter, rule=rule, alpha=alpha,
                     quant_step=quant_step, quant_clip=quant_clip)
        bad = ~res["converged"]
        for row in np.nonzero(bad)[0]:
            supp = np.nonzero(res["bits"][row])[0]     # error pattern support
            if 0 < supp.size <= a_max:
                ts = TrapSet.make(t, supp, source="decoder")
                if ts.b <= b_max:
                    found[ts.vns] = ts
        done += B
    return _rank(list(found.values()))


# ---------------------------------------------------------------- ranking/io
def _rank(sets: list[TrapSet]) -> list[TrapSet]:
    # smaller a first, then smaller b: closest to the dominant floor structures
    return sorted(sets, key=lambda s: (s.a, s.b, s.vns))


def write_trap(path: str | Path, sets: list[TrapSet]) -> None:
    with open(path, "w") as f:
        for s in sets:
            vns = " ".join(str(v + 1) for v in s.vns)   # back to 1-based
            f.write(f"({s.a}, {s.b}) {vns}\n")


def read_trap(t: Tanner | None, path: str | Path) -> list[TrapSet]:
    sets: list[TrapSet] = []
    pat = re.compile(r"\((\d+),\s*(\d+)\)\s*(.*)")
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = pat.match(line)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        vns = tuple(sorted(int(x) - 1 for x in m.group(3).split()))
        sets.append(TrapSet(vns=vns, a=a, b=b, source="file"))
    return sets
