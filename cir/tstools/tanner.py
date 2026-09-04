"""Tanner graph container for LDPC codes.

Loads the project's ``.graph`` format:

    line 0 : "<N> <M>"          # N = #variable nodes, M = #check nodes
    line c : v1 v2 ... vk       # 1-based variable nodes touching check c

and exposes the structures the rest of the toolkit needs: adjacency lists,
padded edge-index arrays for a vectorised belief-propagation decoder, and a
few helpers (syndrome, odd-degree-check count of a variable-node set, short
cycle enumeration).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Tanner:
    N: int                       # number of variable nodes (columns of H)
    M: int                       # number of check nodes    (rows of H)
    cn_to_vn: list[np.ndarray]   # per check: 0-based VN indices
    vn_to_cn: list[np.ndarray]   # per VN:    0-based CN indices

    # ---- edge / message layout (filled in __post_init__) ----------------
    E: int = 0
    edge_cn: np.ndarray = None   # [E] check of each edge
    edge_vn: np.ndarray = None   # [E] variable of each edge
    cn_edges: np.ndarray = None  # [M, dc_max] edge ids per check, -1 padded
    cn_mask: np.ndarray = None   # [M, dc_max] True where real edge
    vn_edges: np.ndarray = None  # [N, dv_max] edge ids per variable, -1 padded
    cn_ptr: np.ndarray = None    # [M] start edge index of each check (edges grouped by check)

    def __post_init__(self) -> None:
        edge_cn, edge_vn = [], []
        # deterministic edge ordering: grouped by check, so cn_edges rows are
        # contiguous ranges — handy and cache friendly.
        for c, vns in enumerate(self.cn_to_vn):
            for v in vns:
                edge_cn.append(c)
                edge_vn.append(int(v))
        self.edge_cn = np.asarray(edge_cn, dtype=np.int32)
        self.edge_vn = np.asarray(edge_vn, dtype=np.int32)
        self.E = self.edge_cn.size

        # per-check edge lists (contiguous because of the ordering above)
        dc = np.array([len(v) for v in self.cn_to_vn])
        dc_max = int(dc.max())
        self.cn_edges = np.full((self.M, dc_max), -1, dtype=np.int32)
        self.cn_mask = np.zeros((self.M, dc_max), dtype=bool)
        self.cn_ptr = np.zeros(self.M, dtype=np.int64)
        e = 0
        for c in range(self.M):
            self.cn_ptr[c] = e
            k = len(self.cn_to_vn[c])
            self.cn_edges[c, :k] = np.arange(e, e + k)
            self.cn_mask[c, :k] = True
            e += k

        # per-variable edge lists
        vn_edge_ids: list[list[int]] = [[] for _ in range(self.N)]
        for eid, v in enumerate(self.edge_vn):
            vn_edge_ids[v].append(eid)
        dv_max = max((len(x) for x in vn_edge_ids), default=1)
        self.vn_edges = np.full((self.N, dv_max), -1, dtype=np.int32)
        for v, ids in enumerate(vn_edge_ids):
            self.vn_edges[v, :len(ids)] = ids

    # ------------------------------------------------------------------ IO
    @classmethod
    def from_graph(cls, path: str | Path) -> "Tanner":
        path = Path(path)
        lines = path.read_text().strip().splitlines()
        N, M = (int(x) for x in lines[0].split())
        cn_to_vn: list[np.ndarray] = []
        for line in lines[1:]:
            vs = [int(x) for x in line.split()]
            # file is 1-based; keep only positive entries, store 0-based
            cn_to_vn.append(np.array([v - 1 for v in vs if v > 0], dtype=np.int32))
        if len(cn_to_vn) != M:
            raise ValueError(f"{path}: header says M={M} but found {len(cn_to_vn)} checks")
        vn_to_cn: list[list[int]] = [[] for _ in range(N)]
        for c, vns in enumerate(cn_to_vn):
            for v in vns:
                vn_to_cn[v].append(c)
        vn_arr = [np.array(x, dtype=np.int32) for x in vn_to_cn]
        return cls(N=N, M=M, cn_to_vn=cn_to_vn, vn_to_cn=vn_arr)

    # -------------------------------------------------------------- helpers
    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        """Parity syndrome of a hard bit vector (or batch [B,N]).

        Vectorised: bits are scattered onto edges (grouped by check), then
        segment-summed with ``reduceat`` at the per-check boundaries.
        """
        bits = np.atleast_2d(bits).astype(np.int64)
        per_edge = bits[:, self.edge_vn]                      # [B, E]
        seg = np.add.reduceat(per_edge, self.cn_ptr, axis=1)  # [B, M]
        return (seg & 1).astype(np.int8)

    def odd_checks(self, vn_set) -> np.ndarray:
        """Checks with odd degree inside the variable-node set ``vn_set``.

        These are the *unsatisfied* checks when exactly those variables are in
        error; ``len`` of the result is the ``b`` of an (a, b) trapping set.
        """
        vn_set = np.asarray(list(vn_set), dtype=np.int64)
        deg = np.zeros(self.M, dtype=np.int64)
        for v in vn_set:
            deg[self.vn_to_cn[v]] += 1
        return np.nonzero(deg & 1)[0]

    def trap_signature(self, vn_set) -> tuple[int, int]:
        """(a, b) = (#variables, #odd-degree checks) for a candidate set."""
        s = set(int(v) for v in vn_set)
        return len(s), int(self.odd_checks(s).size)

    def four_cycles(self) -> list[tuple[int, int]]:
        """Variable-node pairs that share >=2 checks (endpoints of 4-cycles)."""
        pairs: set[tuple[int, int]] = set()
        for c, vns in enumerate(self.cn_to_vn):
            pass  # 4-cycles come from *pairs of checks*; use VN co-membership
        # A 4-cycle is v-c-w-c'-v: v,w share two checks c,c'. Count shared checks.
        from collections import defaultdict
        shared: dict[tuple[int, int], int] = defaultdict(int)
        for c, vns in enumerate(self.cn_to_vn):
            vl = sorted(int(v) for v in vns)
            for i in range(len(vl)):
                for j in range(i + 1, len(vl)):
                    shared[(vl[i], vl[j])] += 1
        for (v, w), cnt in shared.items():
            if cnt >= 2:
                pairs.add((v, w))
        return sorted(pairs)


def code_rate(t: Tanner, punctured: int = 0) -> float:
    """Design rate estimate k/n_tx using rank(H) ~ M and puncturing."""
    k = t.N - t.M
    n_tx = t.N - punctured
    return k / n_tx if n_tx > 0 else float("nan")
