# Have-drafter scaling: local grid (Qwen3-0.6B, Mathlib/Order)

Data-scaling run for a drafter that proposes `have` subgoals, trained and evaluated in CanonicalLean's `generate` format. Ran 2026-10-17 on one RTX 3060 (12 GB) in about 5 h.

## Task

At a tactic step that introduces a new hypothesis or subgoal, the model sees the goal and must propose that subgoal as a `have` tool call. A proposal counts as correct when Canonical's `generate` accepts it and it is definitionally equal (`isDefEq`) to the subgoal the real Mathlib proof introduced.

## Data

- **Source:** `extract_data --drafts` over `Mathlib/Order` (Mathlib at Lean v4.35.0): 13,123 drafts from 335 files.
- **Held out:** 9 files from `Mathlib/Order/Filter`, never trained on: `AtTopBot/BigOperators`, `AtTopBot/Prod`, `Cofinite`, `CountableInter`, `CountableSeparatingOn`, `EventuallyConst`, `Extr`, `Prod`, `Ultrafilter/Basic`. They contain 330 drafts.
- **Training pool:** the 12,793 drafts in the other files. Each point takes the first `n` after one seeded shuffle, so smaller training sets are subsets of larger ones.
- **Example format:** exactly what `generate` sends to the model.
  - User turn: `subgoalPrompt`, i.e. the goal printed with Canonical's `roundtrip` options, plus up to 12 premises.
  - Tools: `subgoalTools`, the `have` tool.
  - Target: the assistant's `have` tool call (`name`, `type`, `clear`).
- **Premises:** the optimal ones, meaning the constants the real proof step used, rendered by `premisesString`. The premise-selection server is not used. 1,272 of the 1,862 Filter drafts have at least one premise.
- **Length:** median about 900 tokens per example, almost all prompt. Examples over 4,096 tokens (about 0.2%) are dropped.

## Training

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-0.6B` (instruct) |
| Method | LoRA r=64, alpha=128, all linear layers, bf16, gradient checkpointing |
| Loss | Cross-entropy on the assistant's tool-call tokens only |
| Chat template | Qwen3 template with `enable_thinking=False`, as Canonical requests |
| Optimizer | AdamW, lr 5e-4, 5% warmup then cosine decay, grad clip 1.0 |
| Batch | Length-bucketed, up to 8,192 padded tokens per step |
| Epochs | 3 |
| Seed | 0 |

The Base model was tried first and dropped: it never learned the `<tool_call>` and `<|im_end|>` tokens, so llama-server never parsed a tool call.

## Evaluation

- **Serving:** llama-server with a q8_0 GGUF of the base model plus the LoRA adapter, using the same flags as Canonical's `startSubgoalDaemon`.
- **Calls:** `extract_data --checkAccuracy` calls `generate` (temperature 0, one attempt) on the same goal and premises the drafts were built from.
- **Universe fix:** this run uses `generate'`, a copy of `generate` that differs only in calling `elabStringAsExpr'`. Canonical's `elabStringAsExpr` rejects types that name the goal's universes, which affects about 18% of drafts ([elabStringAsExpr-bug.md](elabStringAsExpr-bug.md)). On a 150-draft model the fix raised accuracy on trained examples from 60.1% to 76.5%.
- **`train_acc`:** 200 drafts sampled from that point's training set (about 190-200 after duplicates).
- **`heldout_acc`:** all held-out drafts. 323 are checked because 7 duplicate an existing hypothesis, which `draft` also drops.
- **Losses:** mean per-token cross-entropy on the target tokens, over the training set and over the held-out drafts.

## Results

| Training drafts | Steps | Train loss | Held-out loss | Accuracy on trained examples | Accuracy on held-out files |
|---:|---:|---:|---:|---:|---:|
| 123 | 54 | 0.114 | 0.432 | 17.3% (127) | 0.9% (323) |
| 248 | 102 | 0.081 | 0.406 | 30.3% (198) | 2.5% (323) |
| 493 | 204 | 0.076 | 0.350 | 39.9% (198) | 5.6% (323) |
| 989 | 402 | 0.071 | 0.299 | 45.5% (202) | 11.5% (323) |
| 1,984 | 807 | 0.057 | 0.245 | 49.0% (196) | 10.8% (323) |
| 3,973 | 1,596 | 0.048 | 0.219 | 55.2% (192) | 17.6% (323) |

Accuracy cells show the number of checks in parentheses.

### Scaling fit

Held-out loss fits a power law in training-set size D:

```
heldout_loss ≈ 1.24 · D^(-0.209)
```

| D | Predicted | Actual |
|---:|---:|---:|
| 123 | 0.453 | 0.432 |
| 248 | 0.391 | 0.406 |
| 493 | 0.339 | 0.350 |
| 989 | 0.293 | 0.299 |
| 1,984 | 0.253 | 0.245 |
| 3,973 | 0.219 | 0.219 |

## Observations

- Each doubling of the data lowers held-out loss by about 13%, with no flattening up to 4k drafts.
- Held-out accuracy rises 3-6 points per doubling, from 0.9% to 17.6%. With 323 checks the noise is about ±2 points, which covers the dip at 2k.
- Accuracy on trained examples rises but reaches only 55%, short of the 80% target. At 3 epochs the 0.6B model doesn't reproduce training statements exactly, even though train loss keeps falling. Next to test: more epochs (an epoch sweep at fixed D) and larger models.
- Because epochs are fixed, larger D also means more optimizer steps. Part of the gain on trained examples comes from extra optimization rather than extra data.

## Caveats

- **Single run:** one seed, one held-out set, one module (`Order`), one model size.
- **Patched elaboration:** accuracy is measured through `generate'`. Unmodified Canonical scores lower until the `elabStringAsExpr` fix lands.
- **Line endings:** on this Windows machine, Canonical's prompt template is embedded with CRLF line endings, so prompts contain `\r\n`. Linux prompts will differ slightly.
- **One attempt:** evaluation is a single greedy attempt (pass@1). Canonical's default of temperature 0.3 with up to 3 retries is not measured.

## Reproduce

From `tools/training`, with the venv from this run (CUDA torch, transformers, peft):

```
HF_HOME=D:/llama-training/hf .venv/Scripts/python drafter_scaling.py configs/drafter-local.toml
HF_HOME=D:/llama-training/hf .venv/Scripts/python drafter_scaling.py configs/drafter-local.toml --report
```

The whole configuration is [configs/drafter-local.toml](configs/drafter-local.toml). A run skips points already in `results.jsonl` and ends by printing the report. `--report` prints only the report: the data used, how to read the table, the results and the power-law fit. Raw results are in `D:\llama-training\drafter\results.jsonl`, and each point's LoRA adapter and per-draft checks are under `D:\llama-training\drafter\Order\<point>\`.
