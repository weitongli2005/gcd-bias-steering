"""Offline analysis on saved option logprobs.

Computes, per condition (ambig / disambig / ALL):
  FREE   : argmax over {S, U, A}          -> stereo_rate, unknown_rate, accuracy
  FORCED : argmax over {S, A} only        -> forced_stereo_rate, forced_accuracy
  DECOMP : dStereo = composition + preference
  BOOT   : 95% CI by resampling ITEMS (not pairs)
Also emits a re-ranking table (standard metric vs abstention-controlled metric).
"""
import argparse, json, math, os, random, sys

SIDE = {"ambig", "disambig", "ALL"}


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [v / s for v in e]


def per_item(scores, items):
    """returns list of dicts with probabilities and the four argmax outcomes"""
    out = []
    for sc, it in zip(scores, items):
        p = softmax(sc)
        s, u, a = it["s_idx"], it["u_idx"], it["a_idx"]
        ps, pu, pa = p[s], p[u], p[a]
        free = max(range(3), key=lambda i: p[i])
        forced = s if ps >= pa else a
        out.append(dict(eid=it["eid"], cc=it["cc"], label=it["label"],
                        ps=ps, pu=pu, pa=pa,
                        free=("S" if free == s else "U" if free == u else "A"),
                        forced=("S" if forced == s else "A"),
                        stereo_idx=s))
    return out


def agg(rows):
    n = len(rows)
    if n == 0:
        return None
    free = [r["free"] for r in rows]; forced = [r["forced"] for r in rows]
    fS = sum(1 for x in free if x == "S") / n
    fU = sum(1 for x in free if x == "U") / n
    fA = sum(1 for x in free if x == "A") / n
    answered = fS + fA
    # forced-choice probability P(S)
    gS = sum(1 for x in forced if x == "S") / n
    # free-dpref: preference among items the model actually answered with a group
    dpref_free = (fS / answered) if answered else float("nan")
    # accuracy: free uses argmax over 3 (label index), forced cannot pick unknown
    acc_free = sum(1 for r in rows
                   if (r["free"] == "S" and r["stereo_idx"] == r["label"])
                   or (r["free"] == "A" and r["stereo_idx"] != r["label"] and r["label"] != None
                       and r["free"] != "U" and r["label"] == 2)
                   or (r["free"] == "U" and r["label"] == 2)) / n
    return dict(n=n, stereo_rate=fS, unknown_rate=fU, anti_rate=fA,
                dpref_free=dpref_free, forced_stereo_rate=gS,
                **{k: v for k, v in [("acc_free", acc_free)]})


def metrics_simple(rows, label_of_unknown):
    """accurate-enough accuracy: correct if free pick equals the labelled index"""
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/a2_multicat2.json")
    ap.add_argument("--out", default="results/a3_report.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    d = json.load(open(args.inp))
    random.seed(0)
    report = {"categories": {}}

    for cat, blk in d["cats"].items():
        items = blk["items"]
        base_rows = per_item(blk["base_scores"], items)
        ste_rows = per_item(blk["steered_scores"], items)
        cat_rep = {}
        for cc in ["ambig", "disambig", "ALL"]:
            if cc == "ALL":
                bi, si = base_rows, ste_rows
            else:
                bi = [r for r in base_rows if r["cc"] == cc]
                si = [r for r in ste_rows if r["cc"] == cc]
            if not bi:
                continue
            b, s = agg(bi), agg(si)
            # decomposition of dStereo = d(1-unk)*dpref_base + (1-unk_new)*d dpref
            comp = -(s["unknown_rate"] - b["unknown_rate"]) * b["dpref_free"]
            pref = (1 - s["unknown_rate"]) * (s["dpref_free"] - b["dpref_free"])
            tot = s["stereo_rate"] - b["stereo_rate"]
            share = (comp / tot) if abs(tot) > 1e-9 else float("nan")
            # bootstrap over items
            boots = []
            n = len(bi)
            for _ in range(args.boot):
                idx = [random.randrange(n) for _ in range(n)]
                bb, ss = agg([bi[i] for i in idx]), agg([si[i] for i in idx])
                boots.append((ss["stereo_rate"] - bb["stereo_rate"],
                              ss["unknown_rate"] - bb["unknown_rate"],
                              ss["forced_stereo_rate"] - bb["forced_stereo_rate"]))
            def ci(k):
                v = sorted(x[k] for x in boots)
                return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]]
            cat_rep[cc] = dict(
                base=b, steered=s,
                d_stereo=tot, d_unknown=s["unknown_rate"] - b["unknown_rate"],
                d_forced_stereo=s["forced_stereo_rate"] - b["forced_stereo_rate"],
                d_dpref=s["dpref_free"] - b["dpref_free"],
                comp=comp, pref=pref, comp_share=share,
                ci_d_stereo=ci(0), ci_d_unknown=ci(1), ci_d_forced=ci(2),
            )
        report["categories"][cat] = cat_rep

    json.dump(report, open(args.out, "w"), indent=1)

    # ---- print ----
    print(f"{'cat':20s} {'side':9s} {'dStereo':>8s} {'dUnknown':>9s} {'dForced':>8s} "
          f"{'comp%':>7s} {'dStereo CI':>20s}")
    for cat, cr in report["categories"].items():
        for cc in ["ambig", "disambig", "ALL"]:
            if cc not in cr:
                continue
            r = cr[cc]
            print(f"{cat:20s} {cc:9s} {r['d_stereo']:+8.3f} {r['d_unknown']:+9.3f} "
                  f"{r['d_forced_stereo']:+8.3f} {r['comp_share']*100:6.0f}% "
                  f"[{r['ci_d_stereo'][0]:+.3f},{r['ci_d_stereo'][1]:+.3f}]")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
