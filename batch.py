"""Multi-category run that SAVES raw option logprobs, so forced-answer /
decomposition / bootstrap can be computed offline without re-running the model."""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a1_lib import LM, score_options, PREFIX
from a1_run import build_items, capture_last, metrics

CATS = ["Age", "Gender_identity", "Race_ethnicity", "Religion", "Nationality"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cats", default=",".join(CATS))
    ap.add_argument("--n", type=int, default=0)          # 0 = all
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="results/a2_multicat.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cats = args.cats.split(",")
    lm = LM(args.model)
    store = {"args": vars(args), "cats": {}}

    for cat in cats:
        t0 = time.time()
        try:
            items = build_items(cat, None)
        except Exception as e:
            print(f"[{cat}] LOAD FAIL {e}", flush=True); continue
        if args.n:
            items = items[::max(1, len(items) // args.n)][:args.n]
        n_amb = sum(1 for it in items if it["cc"] == "ambig")
        print(f"[{cat}] n={len(items)} ambig={n_amb}", flush=True)

        if not items:
            print(f"[{cat}] n=0 - skipped (no matchable items)", flush=True)
            continue
        base = score_options(lm, items, batch_size=args.bs)
        bm, _ = metrics(base, items)
        print(f"  base stereo={bm['stereo_rate']:.3f} unk={bm['unknown_rate']:.3f} "
              f"acc={bm['accuracy']:.3f}", flush=True)

        dis = [it for it in items if it["cc"] == "disambig"]
        acts = capture_last(lm, dis, [args.layer], batch_size=args.bs)
        H = acts[args.layer].view(len(dis), 3, -1).float()
        D = torch.zeros(H.shape[-1])
        for j, it in enumerate(dis):
            D += H[j, it["s_idx"]] - H[j, it["a_idx"]]
        D /= len(dis)
        print(f"  ||d||={D.norm():.2f}", flush=True)

        h = lm.steer(args.layer, D.cpu(), args.alpha, mode="proj")
        ste = score_options(lm, items, batch_size=args.bs)
        h.remove()
        sm, _ = metrics(ste, items)
        print(f"  steered stereo={sm['stereo_rate']:.3f} unk={sm['unknown_rate']:.3f} "
              f"acc={sm['accuracy']:.3f}", flush=True)

        store["cats"][cat] = dict(
            items=[{k: it[k] for k in ("eid", "cc", "pol", "s_idx", "u_idx", "a_idx", "label")} for it in items],
            base_scores=base, steered_scores=ste,
        )
        print(f"  done {time.time()-t0:.0f}s", flush=True)
        json.dump(store, open(args.out, "w"))

    json.dump(store, open(args.out, "w"))
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
