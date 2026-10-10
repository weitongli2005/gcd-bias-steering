"""GCD: Gradient-Constrained Direction design, versus ARCS selection and the baseline.

Diagnosis (measured, not assumed):
  * the published ambiguous score is a product, commit x dpref;
  * an unconstrained contrastive direction lowers it mainly by suppressing answering --
    it flips 20.8% of DISAMBIGUATED items (where abstaining is wrong) to the unknown option;
  * orthogonalising the contrast direction against a *contrastively estimated*
    non-committal axis is a no-op, because the two are collinear (cos 0.997-0.999).

Method.  Design the direction instead of estimating it.  For small perturbations the option
logits are approximately linear in the activation, so we can ask directly for a direction
that raises the anti-stereotype option relative to the stereotype option while leaving the
unknown option alone:

      maximise   w^T g_pref          g_pref = grad(logit_anti - logit_stereo)
      subject to w^T g_unk  = 0      g_unk  = grad(logit_unknown)
      ||w|| = 1

   => w* = (I - g_unk g_unk^T / ||g_unk||^2) g_pref,  normalised.

Three backward passes per item.  The constraint is on the *gradient* of the confounding
channel, not on a contrastively estimated proxy, which is why it can succeed where
projection against an estimated axis failed.
"""
import argparse, itertools, json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a1_lib import LM, score_options, PREFIX
from a1_run import build_items, capture_last
from a3_analyze import per_item, agg

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--tag", default="qwen7b")
ap.add_argument("--cats", default="Age,Religion,Disability_status,Gender_identity,Nationality")
ap.add_argument("--layer", type=int, default=18)
ap.add_argument("--alphas", default="1,2,4,8")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
lm = LM(args.model); tok = lm.tok
blocks = lm.model.model.layers; L = args.layer

def sdis_raw(rows):
    S=sum(1 for r in rows if r["free"]=="S"); A=sum(1 for r in rows if r["free"]=="A")
    return (2*S/(S+A)-1) if (S+A) else float("nan")

def stats(rows, items):
    a=agg(rows)
    dis=[r for r,it in zip(rows,items) if it["cc"]=="disambig"]
    return dict(dpref=a["dpref_free"], unk=a["unknown_rate"], stereo=a["stereo_rate"],
                dis_acc=sum(1 for r in dis if r["free"]=="A")/max(len(dis),1), sdis=sdis_raw(rows))

@torch.no_grad()
def run(D, alpha, items):
    h=lm.steer(L, D.cpu(), alpha, mode="proj")
    sc=score_options(lm,items,batch_size=args.bs); h.remove()
    return stats(per_item(sc,items), items)

def grad_dirs(items, idx):
    """returns (g_pref, g_unk) averaged over the given item indices"""
    dev=lm.device
    g_pref=torch.zeros(lm.model.config.hidden_size, device=dev)
    g_unk=torch.zeros_like(g_pref)
    store={}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.retain_grad(); store["h"]=h; return out
    handle=blocks[L].register_forward_hook(hook)
    for n,j in enumerate(idx):
        it=items[j]
        pre=PREFIX.format(ctx=it["ctx"], q=it["question"])
        for role,opt in [("S",it["options"][it["s_idx"]]),
                         ("A",it["options"][it["a_idx"]]),
                         ("U",it["options"][it["u_idx"]])]:
            enc=tok(pre+" "+opt, return_tensors="pt", add_special_tokens=False,
                    return_offsets_mapping=True)
            ids=enc["input_ids"].to(lm.device); offs=enc["offset_mapping"][0]
            a=next((i for i,(cs,ce) in enumerate(offs) if ce>len(pre)), len(offs)); a=max(a,1)
            out=lm.model(input_ids=ids)
            logits=out.logits.float()
            pos=a-1; tid=int(ids[0,a])
            h_full=store["h"]                 # [1, T, d] -- retain on the full tensor
            h_full.retain_grad()
            coeff = 1.0 if role=="A" else (-1.0 if role=="S" else 0.0)
            logits[0,pos,tid].backward(retain_graph=False)
            g=h_full.grad[0,pos,:].detach().clone()
            if role=="U": g_unk += g/len(idx)
            else:         g_pref += coeff*g/len(idx)
            lm.model.zero_grad(set_to_none=True)
        if (n+1)%50==0: print(f"    grads {n+1}/{len(idx)}", flush=True)
    handle.remove()
    return g_pref, g_unk

R={"args":vars(args),"cats":{}}
AL=[float(x) for x in args.alphas.split(",")]
for cat in args.cats.split(","):
    raw=build_items(cat,None)
    amb=[i for i in raw if i["cc"]=="ambig"]; dis=[i for i in raw if i["cc"]=="disambig"]
    import random; random.seed(0); k=args.n//2
    items=random.sample(amb,min(k,len(amb)))+random.sample(dis,min(k,len(dis)))
    qis=sorted(set(it["qi"] for it in items)); random.seed(1); random.shuffle(qis)
    half=set(qis[:len(qis)//2])
    fidx=[j for j,it in enumerate(items) if it["qi"] in half]
    eidx=[j for j,it in enumerate(items) if it["qi"] not in half]
    fit=[items[j] for j in fidx]; ev=[items[j] for j in eidx]
    for j,it in enumerate(items): it["_j"]=j
    bf=agg(per_item(score_options(lm,fit,batch_size=args.bs),fit))
    be=agg(per_item(score_options(lm,ev,batch_size=args.bs),ev))
    base_ev=stats(per_item(score_options(lm,ev,batch_size=args.bs),ev),ev)

    acts=capture_last(lm,items,[L],batch_size=args.bs)
    H=acts[L].view(len(items),3,-1).float().to(lm.device)
    Dm=torch.zeros(H.shape[-1], device=H.device)
    for j in fidx: Dm += H[j,items[j]["s_idx"]]-H[j,items[j]["a_idx"]]
    Dm/=max(len(fidx),1)

    print(f"  [{cat}] computing gradients on {len(fidx)} fit items", flush=True)
    g_pref,g_unk = grad_dirs(items, fidx)
    gu = g_unk/ (g_unk.norm()+1e-9)
    g_par = (g_pref@gu)*gu
    Dg = g_pref - g_par
    cos_raw = float(F.cosine_similarity(Dm.unsqueeze(0), Dg.unsqueeze(0)))
    cos_unk = float(F.cosine_similarity(Dm.unsqueeze(0), g_unk.unsqueeze(0)))
    Dg = Dg/(Dg.norm()+1e-9)
    Dm = Dm.to(Dg.device)

    rec={}
    for al in AL:
        rec[f"mean_{al:g}"]=dict(kind="mean-contrast",alpha=al,**run(Dm,al,ev))
        rec[f"gcd_{al:g}"] =dict(kind="GCD (ours)",   alpha=al,**run(Dg,al,ev))
    R["cats"][cat]=dict(n=len(items),n_fit=len(fidx),n_ev=len(ev),base=base_ev,
                        cos_mean_vs_gcd=cos_raw, cos_mean_vs_gunk=cos_unk, grid=rec)
    m=rec.get("mean_8"); g=rec.get("gcd_8")
    print(f"     dUnk: mean={m['unk']-be['unknown_rate']:+.3f}  GCD={g['unk']-be['unknown_rate']:+.3f} | "
          f"dpref: mean={m['dpref']-be['dpref_free']:+.3f}  GCD={g['dpref']-be['dpref_free']:+.3f} | "
          f"cos(D_mean, D_gcd)={cos_raw:+.3f}", flush=True)

print("\n=== summary (held-out) ===")
for tag_ in ["mean","gcd"]:
    for al in AL:
        rs=[(c,v["grid"][f"{tag_}_{al:g}"],v["base"]) for c,v in R["cats"].items() if f"{tag_}_{al:g}" in v["grid"]]
        if not rs: continue
        print(f"  {tag_:>4s} a={al:<4g} dUnk={np.mean([r[1]['unk']-r[2]['unk'] for r in rs]):+.4f} "
              f"dpref={np.mean([r[1]['dpref']-r[2]['dpref'] for r in rs]):+.4f} "
              f"dis_acc={np.mean([r[1]['dis_acc'] for r in rs]):.4f} "
              f"(base {np.mean([r[2]['dis_acc'] for r in rs]):.4f})")
json.dump(R,open(args.out,"w"),indent=1,default=str)
print("WROTE",args.out)
