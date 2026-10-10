"""A1 core: BBQ option scoring + contrast direction + steering.
M0 goal: reproduce the confidence/abstention confound.
"""
import json, math, os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = "Context: {ctx}\nQuestion: {q}\nAnswer:"


# ---------------- model ----------------
class LM:
    def __init__(self, path, dtype=torch.bfloat16, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype
        ).to(device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        self.device = device
        self._hooks = []
        self._buf = {}

    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._buf = {}

    # ---- capture residual stream ----
    def capture(self, layer_ids):
        self._buf = {L: [] for L in layer_ids}
        for L in layer_ids:
            def mk(L):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    # left-padding convention -> position -1 is the last REAL token
                    self._buf[L].append(h[:, -1, :].detach().float().cpu())
                return hook
            self._hooks.append(self.model.model.layers[L].register_forward_hook(mk(L)))

    # ---- steering ----
    def steer(self, layer, vec, alpha, mode="proj", sign=1, targets="all"):
        """mode: proj = remove component along vec ; add = subtract alpha*vec"""
        v = (vec.to(self.device).float()) * float(sign)

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hh = h.float()
            vv = v / (v.norm() + 1e-8)
            if mode == "proj":
                coef = (hh @ vv).unsqueeze(-1)
                hh = hh - alpha * coef * vv
            else:
                hh = hh - alpha * vv
            hh = hh.to(h.dtype)
            return (hh,) + out[1:] if isinstance(out, tuple) else hh
        return self.model.model.layers[layer].register_forward_hook(hook)


# ---------------- scoring ----------------
@torch.no_grad()
def score_options(lm, items, batch_size=8, prefix=None):
    """items: list of dict(ctx, question, options=[str,...])
    returns list of np-array-like lists of length-normalized logprobs per option."""
    tok = lm.tok
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        seqs, spans = [], []
        for it in chunk:
            pre = (prefix or PREFIX).format(ctx=it["ctx"], q=it["question"])
            pre_len = len(pre)
            for opt in it["options"]:
                full = pre + " " + opt
                enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                fids = enc["input_ids"]; offs = enc["offset_mapping"]
                # first token that OVERLAPS the option region (BPE merges the leading space
                # into the option token, so its char offset starts *before* pre_len)
                a = next((i for i, (cs, ce) in enumerate(offs) if ce > pre_len), len(offs))
                a = max(a, 1)
                seqs.append(fids)
                spans.append((a, len(fids)))
        maxlen = max(len(s) for s in seqs)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for j, s in enumerate(seqs):
            input_ids[j, :len(s)] = torch.tensor(s)
            attn[j, :len(s)] = 1
        input_ids = input_ids.to(lm.device)
        attn = attn.to(lm.device)
        logits = lm.model(input_ids=input_ids, attention_mask=attn).logits
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for j, (a, b) in enumerate(spans):
            if b <= a:
                out.append(-1e9); continue
            tgt = input_ids[j, a:b]
            lp = logprobs[j, a - 1:b - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append(float(lp.mean()))   # length-normalized
    # regroup
    grouped = []
    k = 0
    for it in items:
        n = len(it["options"])
        grouped.append(out[k:k + n]); k += n
        # note: caller handles softmax; we want RAW sums vs means? use mean then softmax
    return grouped


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def entropy(p):
    return -sum(pv * math.log(pv + 1e-12) for pv in p)
