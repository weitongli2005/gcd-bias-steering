"""A1 / M0: reproduce the confidence-abstention confound on BBQ.

Pipeline:
  1. load BBQ (ambig+disambig), classify each option as STEREO / ANTI / UNKNOWN
  2. baseline: score 3 options, argmax -> stereo_rate / unknown_rate / entropy
  3. contrast direction d = mean(h(stereo-continuation) - h(anti-continuation)) at a mid layer
  4. steer: projection removal along d, alpha sweep
  5. report deltas
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a1_lib import LM, score_options, softmax, entropy, PREFIX

from datasets import load_dataset


def build_items(cat="Age", limit=None):
    ds = load_dataset("heegyu/bbq", cat, split="test")
    items = []
    for r in ds:
        ai = json.loads(r["answer_info"]) if isinstance(r["answer_info"], str) else r["answer_info"]
        md = json.loads(r["additional_metadata"]) if isinstance(r["additional_metadata"], str) else r["additional_metadata"]
        stereo_groups = set(md.get("stereotyped_groups") or [])
        opts = [r["ans0"], r["ans1"], r["ans2"]]
        groups = [ai["ans0"][1], ai["ans1"][1], ai["ans2"][1]]

        def _norm(x):
            # SES uses lowSES/highSES in answer_info but "low SES"/"high SES" as
            # stereotyped groups; normalise whitespace and underscores before comparing.
            return x.lower().replace(" ", "").replace("_", "")

        def is_stereo(text, label):
            t, l = _norm(text), _norm(label)
            for g in stereo_groups:
                gl = _norm(g)
                if gl == l or gl in t:
                    return True
            return False

        s_idx = u_idx = a_idx = None
        s_hits = [i for i in range(3) if is_stereo(opts[i], groups[i])]
        u_hits = [i for i in range(3) if groups[i].lower() == "unknown"]
        if len(s_hits) == 1 and len(u_hits) == 1:
            s_idx = s_hits[0]; u_idx = u_hits[0]
            rest = [i for i in range(3) if i not in (s_idx, u_idx)]
            if len(rest) == 1:
                a_idx = rest[0]
        if None in (s_idx, u_idx, a_idx):
            continue
        items.append(dict(
            eid=str(r["example_id"]), cat=cat, qi=str(r["question_index"]),
            cc=r["context_condition"], pol=r["question_polarity"],
            ctx=r["context"], question=r["question"], options=opts,
            s_idx=s_idx, u_idx=u_idx, a_idx=a_idx,
            label=int(r["label"]),
        ))
        if limit and len(items) >= limit:
            break
    return items


@torch.no_grad()
def capture_last(lm, items, layer_ids, batch_size=8):
    """Forward prompt+' '+each option, capture residual at last token per layer.
    Returns dict layer -> tensor [n_items*3, d]"""
    tok = lm.tok
    lm.clear_hooks(); lm.capture(layer_ids)
    seqs = []
    for it in items:
        pre = PREFIX.format(ctx=it["ctx"], q=it["question"])
        for opt in it["options"]:
            seqs.append(tok(pre + " " + opt, add_special_tokens=False)["input_ids"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(seqs), batch_size):
        ch = seqs[i:i + batch_size]
        maxlen = max(len(s) for s in ch)
        ids = torch.full((len(ch), maxlen), pad_id, dtype=torch.long)
        am = torch.zeros((len(ch), maxlen), dtype=torch.long)
        for j, s in enumerate(ch):                      # LEFT-pad: last idx == last real token
            ids[j, maxlen - len(s):] = torch.tensor(s); am[j, maxlen - len(s):] = 1
        lm.model(input_ids=ids.to(lm.device), attention_mask=am.to(lm.device))
    acts = {L: torch.cat(v, 0) for L, v in lm._buf.items()}   # grab BEFORE clearing
    lm.clear_hooks()
    return acts


def metrics(scores, items):
    """scores: list of per-item option score lists -> softmax probs"""
    n = len(items)
    stereo = unknown = anti = 0
    ents, accs = [], []
    ent_uncond = 0.0
    per_item = []
    for sc, it in zip(scores, items):
        p = softmax(sc)
        k = max(range(len(p)), key=lambda i: p[i])
        e = entropy(p)
        ents.append(e); accs.append(1.0 if k == it["label"] else 0.0)
        r = dict(eid=it["eid"], cc=it["cc"], choice=k, s=it["s_idx"], u=it["u_idx"],
                 a=it["a_idx"], p_stereo=p[it["s_idx"]], p_unknown=p[it["u_idx"]],
                 p_anti=p[it["a_idx"]], ent=e)
        per_item.append(r)
        if k == it["s_idx"]: stereo += 1
        elif k == it["u_idx"]: unknown += 1
        else: anti += 1
    answered_group = stereo + anti
    return dict(
        n=n,
        stereo_rate=stereo / n, unknown_rate=unknown / n, anti_rate=anti / n,
        dpref=(stereo / answered_group) if answered_group else float("nan"),
        accuracy=sum(accs) / n, mean_entropy=sum(ents) / n,
    ), per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cat", default="Age")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alphas", default="0.0,0.5,1.0,1.5")
    ap.add_argument("--out", default="results/a1_qwen_age.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    print("loading model...", flush=True)
    lm = LM(args.model)
    print(f"  layers={lm.n_layers} d={lm.d_model}  {time.time()-t0:.0f}s", flush=True)

    items = build_items(args.cat, args.limit)
    print(f"items={len(items)} cat={args.cat}", flush=True)

    # ---------- baseline ----------
    t1 = time.time()
    base_scores = score_options(lm, items, batch_size=8)
    base_m, base_pi = metrics(base_scores, items)
    print(f"[baseline] {json.dumps(base_m, indent=1)}  ({time.time()-t1:.0f}s)", flush=True)

    # ---------- contrast direction ----------
    t2 = time.time()
    dis = [it for it in items if it["cc"] == "disambig"]
    acts = capture_last(lm, dis, [args.layer], batch_size=8)
    H = acts[args.layer]                              # [n*3, d]
    H = H.view(len(dis), 3, -1)
    d = torch.zeros(H.shape[-1], dtype=torch.float32)
    for j, it in enumerate(dis):
        d += H[j, it["s_idx"]] - H[j, it["a_idx"]]
    d = d / len(dis)
    d = d.to(torch.float32)
    print(f"[direction] layer={args.layer} ||d||={d.norm():.3f}  ({time.time()-t2:.0f}s)", flush=True)

    # ---------- steer ----------
    results = {"args": vars(args), "baseline": base_m, "items": len(items),
               "direction_norm": float(d.norm()), "steer": {}}
    for a in [float(x) for x in args.alphas.split(",")]:
        if a == 0.0:
            results["steer"]["0.0"] = base_m
            continue
        t3 = time.time()
        h = lm.steer(args.layer, d.cpu(), a, mode="proj")
        sc = score_options(lm, items, batch_size=8)
        h.remove()
        m, pi = metrics(sc, items)
        results["steer"][str(a)] = m
        print(f"[steer alpha={a}] stereo={m['stereo_rate']:.3f} unknown={m['unknown_rate']:.3f} "
              f"dpref={m['dpref']:.3f} acc={m['accuracy']:.3f} ent={m['mean_entropy']:.3f} "
              f"({time.time()-t3:.0f}s)", flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
