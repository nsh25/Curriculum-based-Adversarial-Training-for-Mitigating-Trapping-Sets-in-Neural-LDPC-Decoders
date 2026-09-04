from pathlib import Path
import re
import numpy as np

bg = Path(__file__).resolve().parent

codes = [
    ("5G_LDPC_R0.33_n_dec896_n768_k256_z32_s257_320.txt", 32),
    ("5G_LDPC_R0.50_n_dec1280_n1024_k512_z64_s513_640.txt", 64),
    ("5G_LDPC_R0.50_n_dec640_n512_k256_z32_s257_320.txt", 32),
    ("5G_LDPC_R0.73_n_dec2304_n2112_k1536_z72_s1537_1584.txt", 72),
    ("5G_LDPC_R0.73_n_dec480_n352_k256_z32_s257_320.txt", 32),
    ("802_11n_N648_R56_z27.txt", 27),
    ("wman_N0576_R34_z24.txt", 24),
    ("MACKAY_N96_K48.txt", 1),
    ("BCH_63_51.txt", 1),
    ("Polar_64_48.txt", 1),
]


def read_shifts(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(x) for x in re.split(r"[\s,]+", line) if x != ""])
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged matrix in {path}: {widths}")
    return np.array(rows, dtype=int)


def expand_qc(S: np.ndarray, z: int):
    Mb, Nb = S.shape
    M, N = Mb * z, Nb * z
    cn_to_vns = [[] for _ in range(M)]
    for i in range(Mb):
        for j in range(Nb):
            sh = int(S[i, j])
            if sh < 0:
                continue
            # MATLAB circshift(eye(z), [0, sh]): VN_local = (CN_local + sh) % z
            for r in range(z):
                c = (r + sh) % z
                cn_to_vns[i * z + r].append(j * z + c + 1)  # 1-based VNs
    return N, M, cn_to_vns


def write_graph(path: Path, N: int, M: int, cn_to_vns):
    with path.open("w") as f:
        f.write(f"{N} {M}\n")
        for vns in cn_to_vns:
            f.write(" ".join(map(str, sorted(vns))) + "\n")


def main():
    for name, z in codes:
        src = bg / name
        if not src.exists():
            print("MISSING", name)
            continue
        S = read_shifts(src)
        N, M, adj = expand_qc(S, z)
        out = bg / f"{src.stem}.graph"
        write_graph(out, N, M, adj)
        deg = [len(a) for a in adj]
        print(f"OK {src.stem}: z={z} N={N} M={M} rowdeg=[{min(deg)},{max(deg)}]")

    ref = bg.parent / "480" / "5G_LDPC_R0.73_n_dec480_n352_k256_z32_s257_320.graph"
    new = bg / "5G_LDPC_R0.73_n_dec480_n352_k256_z32_s257_320.graph"
    if ref.exists() and new.exists():

        def edges(text: str):
            lines = text.splitlines()
            e = set()
            for ci, line in enumerate(lines[1:]):
                for v in map(int, line.split()):
                    if v > 0:
                        e.add((ci + 1, v))
            return e

        ea, eb = edges(ref.read_text()), edges(new.read_text())
        print(
            "480 compare:",
            "header",
            ref.read_text().splitlines()[0],
            "vs",
            new.read_text().splitlines()[0],
            "edges_equal",
            ea == eb,
            "symdiff",
            len(ea ^ eb),
        )


if __name__ == "__main__":
    main()
