# Reward Hacking in Activation-Space Bias Steering — Core Code

Core implementation and configuration for the paper:

> **Rewarding Silence: Contrastive Bias Steering Suffers an Answer-Rate Confound,
> and Gradient-Constrained Direction Design (GCD) Avoids It**

We audit activation-space bias steering on BBQ and show that the standard
contrastive recipe improves the published score chiefly by raising the model's
abstention rate on ambiguous items, not by changing its preference. We then
introduce **GCD** (Gradient-Constrained Direction), which *designs* the steering
direction from option-logit gradients instead of estimating it.

---

## Installation

```bash
pip install -r requirements.txt
```

All experiments run on a single 48 GB GPU in bf16. Set `HF_ENDPOINT` if you are
behind a mirror, e.g. `export HF_ENDPOINT=https://hf-mirror.com`.

The scripts accept a local model path or a Hugging Face model ID via `--model`.
Model weights, experiment outputs, and datasets are not included in this
repository. Obtain BBQ and StereoSet from their official releases and follow
their licenses and citation requirements.

---

## Layout

```
a1_lib.py, a1_run.py, a3_analyze.py  shared loading, scoring, and analysis utilities
scoring.py         option scoring and token/space convention
runner.py          one full run and metrics
bbq_faithful.py    faithful BBQ formulas
gcd.py             GCD closed-form direction
batch.py           batch execution over BBQ categories
run_main.py        main reproduction entry point
```

---

## The scoring convention (the easiest thing to get wrong)

Prompts are scored as `pre + " " + option`, and we read the log-probability at
the **first token whose character span overlaps the option region**, because BPE
merges the leading space into the option's first token. The space is **kept**:

```python
full = pre + " " + opt
a = next(i for i, (cs, ce) in enumerate(offsets) if ce > pre_len)
score = mean(logprob(tokens[a:]))
```

This convention is what the paper's Reproduction Details section fixes. If you
strip the space, or score the space token itself, you will not reproduce our
numbers.

**Metric conventions** (`runner.py::metrics`):

- `dpref = stereo / (stereo + anti)`, **no polarity flip**.
- `s_Dis = 2 * (n_biased / n_non_unknown) - 1`; unknown responses are excluded
  from both numerator and denominator.
- `s_Amb = commit * (2 * dpref - 1)` on ambiguous items, with
  `commit = 1 - Pr(U)`.

**GCD** (`gcd.py`): `w* = (I - gU gU^T)(gA - gS) / ||.||`, with three backward
passes **per fit item**; evaluation items need none. Steering is applied as
`h <- h - alpha * v/||v||` via a forward hook on `model.model.layers[L]`, with
layer `L = round(n_layers * 0.64)` (18 for a 28-layer model, 23 for a 36-layer
one).

---

## Reproducing the main result

```bash
python run_main.py --out reproduced.json
```

For the paper's headline numbers (Qwen2.5-7B, layer 18, alpha=8, held-out,
11 categories), run the entry point above and compare the resulting metrics.

|          | delta unknown-rate | dis-acc | delta preference |
|----------|--------------------|---------|------------------|
| base     | --                 | 0.4828  | --               |
| contrastive | **+0.0705**     | 0.4581  | -0.0040          |
| **GCD**  | **-0.0049**        | **0.4932** | -0.0160       |

The paper reports the contrastive recipe moving the unknown rate by 14.3x more
than GCD, with `cos(D_mean, D_GCD) = -0.0163`.

---

## Data

BBQ is used through `heegyu/bbq` (test split, 28,824 items, 11 categories).
**Please obtain BBQ from its official release** and cite Parrish et al. (2022);
we do not redistribute the dataset here. StereoSet is the intrasentence
validation split.

---

## Citation

```bibtex
@inproceedings{anonymous2027rewarding,
  title     = {Rewarding Silence: Contrastive Bias Steering Suffers an
               Answer-Rate Confound, and Gradient-Constrained Direction
               Design ({GCD}) Avoids It},
  author    = {Anonymous},
  booktitle = {Submitted to ACL Rolling Review},
  year      = {2027}
}
```

---

## License

MIT (see `LICENSE`).
