"""Structural trap search for every .graph in BaseGraph/.

Writes for each code:
  <stem>.found.trap       — all sets (a_min..a_max, b<=b_max)
  <stem>.dominant.trap    — dominant subset (a>=a_dom_min, b<=b_dom_max)

  python run_all_basegraph_traps.py
"""
import functools, time, traceback
print = functools.partial(print, flush=True)
from collections import Counter
from pathlib import Path

from tstools import Tanner, structural_search, write_trap

ROOT = Path("BaseGraph")
A_MIN, A_MAX, B_MAX, BEAM = 1, 16, 4, 4
# dominant filter (matches dominant_trap.py defaults)
A_DOM_MIN, B_DOM_MAX = 3, 2


def main():
    graphs = sorted(ROOT.glob("*.graph"))
    print(f"# BaseGraph: {len(graphs)} PCM files")
    print(f"# structural: a={A_MIN}..{A_MAX}, b<={B_MAX}, beam={BEAM}")
    print(f"# dominant:   a>={A_DOM_MIN}, b<={B_DOM_MAX}")
    print()

    for g in graphs:
        print("=" * 72)
        print(f"# {g.name}")
        t0 = time.time()
        try:
            t = Tanner.from_graph(g)
            print(f"#   N={t.N} M={t.M} E={t.E}")
            sets = structural_search(t, a_max=A_MAX, b_max=B_MAX,
                                     a_min=A_MIN, beam=BEAM)
            out_all = g.with_suffix(".found.trap")
            write_trap(out_all, sets)
            hist = Counter((s.a, s.b) for s in sets)
            print(f"#   found {len(sets)} sets -> {out_all.name}  "
                  f"[{time.time()-t0:.1f}s]")
            for ab in sorted(hist):
                print(f"#     ({ab[0]:2d},{ab[1]}): {hist[ab]}")

            dom = [s for s in sets if s.a >= A_DOM_MIN and s.b <= B_DOM_MAX]
            dom.sort(key=lambda s: (s.a, s.b, s.vns))
            dhist = Counter((s.a, s.b) for s in dom)
            out_dom = Path(str(g).replace(".graph", ".dominant.trap"))
            write_trap(out_dom, dom)
            print(f"#   DOMINANT {len(dom)} sets in {len(dhist)} classes "
                  f"-> {out_dom.name}")
            for ab in sorted(dhist):
                print(f"#     ({ab[0]:2d},{ab[1]}): {dhist[ab]}")
            if dom:
                a0, b0 = dom[0].a, dom[0].b
                print(f"#   >>> smallest class ({a0},{b0}) x{dhist[(a0,b0)]}  "
                      f"eg VNs(1-based) {[v+1 for v in dom[0].vns]}")
        except Exception as ex:
            print(f"#   FAILED: {ex}")
            traceback.print_exc()
        print()
    print("ALLDONE")


if __name__ == "__main__":
    main()
