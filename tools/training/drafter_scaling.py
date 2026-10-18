#!/usr/bin/env python3
"""Have-drafter scaling experiment: extract drafts, fine-tune, serve, measure defeq accuracy.

    tools/training/.venv/Scripts/python tools/training/drafter_scaling.py CONFIG
    tools/training/.venv/Scripts/python tools/training/drafter_scaling.py CONFIG --report
    # CONFIG (e.g. configs/drafter-local.toml) defines the whole experiment: data, held-out files,
    # models, sizes and hyperparameters. Running it trains and evaluates every (model, size) not yet
    # in <out>/results.jsonl, then prints the report; --report only prints it.
    # Format: Canonical's `generate` chat (subgoalPrompt + subgoalTools from .drafts.jsonl), loss on
    # the assistant's `have` tool call; evaluated by `extract_data --checkAccuracy`, which calls `generate`.
"""
import argparse, json, math, os, random, subprocess, sys, time, tomllib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from premise_loop import LOOP, CONVERT, ensure_llama, jsonl, lean_files, run, server_up

DRAFTER = "http://127.0.0.1:8765"
CONVERT_LORA = CONVERT.with_name("convert_lora_to_gguf.py")

def clear(files, suffixes):
    for lean in files:
        for suffix in suffixes:
            lean.with_suffix(suffix).unlink(missing_ok=True)

def extract(entry, files, *flags):
    clear(files, [".extract.jsonl", ".extract.jsonl.tmp", ".drafts.jsonl" if "--drafts" in flags else ".checks.jsonl"])
    run(["lake", "exe", "extract_data", *flags, str(entry)])

def render(tok, messages, tools, **kw):
    # Canonical sends chat_template_kwargs {"enable_thinking": false}.
    return tok.apply_chat_template(messages, tools=tools, tokenize=False, enable_thinking=False, **kw)

def examples(rows, tok, max_len):
    """(prompt, target) token ids: the `generate` prompt, and the assistant's `have` tool call."""
    out = []
    for row in rows:
        prompt = render(tok, row["messages"][:-1], row["tools"], add_generation_prompt=True)
        full = render(tok, row["messages"], row["tools"])
        assert full.startswith(prompt), "chat template is not prefix-stable"
        p = tok(prompt, add_special_tokens=False).input_ids
        t = tok(full[len(prompt):], add_special_tokens=False).input_ids
        if len(p) + len(t) <= max_len:
            out.append((p, t))
    return out

def batches(exs, batch_tokens):
    """Length-sorted batches of at most `batch_tokens` padded tokens, in random order."""
    exs = sorted(exs, key=lambda e: len(e[0]) + len(e[1]))
    out, cur = [], []
    for e in exs:
        if cur and (len(cur) + 1) * (len(e[0]) + len(e[1])) > batch_tokens:
            out.append(cur); cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    random.shuffle(out)
    return out

def collate(batch, pad, device):
    width = max(len(p) + len(t) for p, t in batch)
    ids = torch.full((len(batch), width), pad)
    labels = torch.full((len(batch), width), -100)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (p, t) in enumerate(batch):
        seq = p + t
        ids[i, :len(seq)] = torch.tensor(seq)
        labels[i, len(p):len(seq)] = torch.tensor(t)
        mask[i, :len(seq)] = 1
    return ids.to(device), labels.to(device), mask.to(device)

def target_loss(model, ids, labels, mask, reduction="mean"):
    """Cross-entropy over target tokens only, without materializing full-vocabulary logits."""
    base = model.get_base_model()
    hidden = base.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    keep = targets != -100
    logits = base.lm_head(hidden[keep]).float()
    return torch.nn.functional.cross_entropy(logits, targets[keep], reduction=reduction)

@torch.no_grad()
def mean_loss(model, exs, pad, batch_tokens):
    model.eval()
    total, n = 0.0, 0
    for b in batches(exs, batch_tokens):
        ids, labels, mask = collate(b, pad, "cuda")
        total += float(target_loss(model, ids, labels, mask, "sum")); n += int((labels[:, 1:] != -100).sum())
    model.train()
    return total / max(n, 1)

def train(args, rows, heldout, out_dir):
    """Fine-tune a LoRA adapter on `rows`; `args` is one grid point (config + model + n)."""
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    exs, held = examples(rows, tok, args.max_len), examples(heldout, tok, args.max_len)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(batches(exs, args.batch_tokens)) / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    print(f"{len(exs)} examples, {steps} steps", flush=True)
    for epoch in range(args.epochs):
        total, n = 0.0, 0
        bs = batches(exs, args.batch_tokens)
        for i, b in enumerate(bs):
            ids, labels, mask = collate(b, pad, "cuda")
            loss = target_loss(model, ids, labels, mask)
            (loss / args.accum).backward()
            total += float(loss); n += 1
            if (i + 1) % args.accum == 0 or i + 1 == len(bs):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad()
        print(f"epoch {epoch + 1} train loss {total / n:.4f}", flush=True)
    loss, held_loss = mean_loss(model, exs, pad, args.batch_tokens), mean_loss(model, held, pad, args.batch_tokens)
    print(f"final train loss {loss:.4f} heldout loss {held_loss:.4f}", flush=True)
    # Adapter only: merging copies the whole model to host memory.
    model.save_pretrained(out_dir / "adapter")
    return len(exs), loss, held_loss

def kill_port(port):
    if os.name == "nt":
        subprocess.run(["powershell", "-NoProfile", "-Command",
            f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force }}"])
    else:
        subprocess.run(f"lsof -ti tcp:{port} -sTCP:LISTEN | xargs -r kill", shell=True)
    while server_up(DRAFTER):
        time.sleep(0.5)

def base_gguf(model, out):
    """q8_0 GGUF of the base model, converted once per model."""
    from huggingface_hub import snapshot_download
    gguf = out / "base" / f"{model.split('/')[-1]}.q8_0.gguf"
    if not gguf.is_file():
        gguf.parent.mkdir(parents=True, exist_ok=True)
        hf = snapshot_download(model, local_files_only=True)
        subprocess.check_call([sys.executable, str(CONVERT), hf, "--outfile", str(gguf), "--outtype", "q8_0"])
    return gguf

def start_drafter(gguf, lora):
    kill_port(8765)
    exe = ensure_llama()
    # As Canonical's startSubgoalDaemon, plus the adapter; --no-mmap instead of --mlock to spare host RAM.
    args = [str(exe), "-m", str(gguf), "--lora", str(lora), "--host", "127.0.0.1", "--port", "8765",
            "--n-gpu-layers", "-1", "--no-mmap", "--ctx-size", "8192", "--reasoning-budget", "0",
            "--parallel", "1", "--cache-ram", "0"]
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen(args, cwd=exe.parent, creationflags=flags, close_fds=True)
    for _ in range(600):
        if server_up(DRAFTER):
            return
        time.sleep(0.5)
    raise SystemExit("drafter server did not come up within 300s")

def split(cfg, files):
    """(training files, held-out files)."""
    held = {(LOOP / f).resolve() for f in cfg["heldout"]}
    files = sorted(files)
    return [f for f in files if f not in held], [f for f in files if f in held]

def key(row):
    """A `--checkKeys` line for ExtractData.lean."""
    return json.dumps([row["goal"], row["type"]], ensure_ascii=False, separators=(",", ":"))

def rel(path):
    return path.relative_to(LOOP).as_posix() if path.is_relative_to(LOOP) else str(path)

def point(cfg, model, n):
    """One grid point: the config's hyperparameters plus a model and a training-set size."""
    return argparse.Namespace(**{k: v for k, v in cfg.items() if k != "models"}, **cfg["models"][model], model=model, n=n)

def experiment(cfg_path, cfg, args, entry, files, out, train_only=False):
    random.seed(args.seed); torch.manual_seed(args.seed)
    train_files, held_files = split(cfg, files)
    pool = [{**row, "file": f} for f in train_files for row in jsonl(f.with_suffix(".drafts.jsonl"))]
    heldout = [row for f in held_files for row in jsonl(f.with_suffix(".drafts.jsonl"))]
    random.shuffle(pool)
    rows = pool[:args.n]
    name = f"{args.model.split('/')[-1]}-n{len(rows)}-e{args.epochs}-lr{args.lr}-s{args.seed}"
    out_dir = out / entry.stem / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if train_only:
        n_train, loss, held_loss = train(args, rows, heldout, out_dir)
        (out_dir / "train.json").write_text(json.dumps({"n_train": n_train, "train_loss": loss, "heldout_loss": held_loss}))
        return
    print(f"== {name}: {len(rows)}/{len(pool)} training drafts, {len(heldout)} held out", flush=True)
    # Train in a child process so its CUDA/torch memory is released before Lean loads Mathlib.
    subprocess.check_call([sys.executable, "-u", __file__, str(cfg_path), "--train-only", args.model, str(args.n)])
    trained = json.loads((out_dir / "train.json").read_text())
    n_train, loss, held_loss = trained["n_train"], trained["train_loss"], trained["heldout_loss"]
    from huggingface_hub import snapshot_download
    lora = out_dir / "adapter.f16.gguf"
    subprocess.check_call([sys.executable, str(CONVERT_LORA), str(out_dir / "adapter"), "--outfile", str(lora),
                           "--outtype", "f16", "--base", snapshot_download(args.model, local_files_only=True)])

    start_drafter(base_gguf(args.model, out), lora)
    seen = {key(r) for r in rows}
    sample = random.Random(args.seed).sample(sorted(rows, key=key), min(args.eval_train, len(rows)))
    keys = out_dir / "check_keys.txt"
    keys.write_text("".join(key(r) + "\n" for r in sample + heldout), encoding="utf-8")
    # Only elaborate the files that hold a checked draft.
    eval_files = sorted({r["file"] for r in sample} | set(held_files))
    eval_list = out_dir / "eval_files.txt"
    eval_list.write_text("".join(str(f) + "\n" for f in eval_files), encoding="utf-8")
    extract(eval_list, eval_files, "--checkAccuracy", f"--checkKeys={keys}")
    checks = {f: jsonl(f.with_suffix(".checks.jsonl")) for f in eval_files}
    on_seen = [c["defeq"] for f in eval_files if f not in held_files for c in checks[f] if key(c) in seen]
    on_held = [c["defeq"] for f in held_files for c in checks[f]]
    (out_dir / "checks.jsonl").write_text("".join(json.dumps(c) + "\n" for f in eval_files for c in checks[f]), encoding="utf-8")
    result = {"entry": rel(entry), "model": args.model, "n": args.n, "n_train": n_train, "epochs": args.epochs,
              "lr": args.lr, "lora_r": args.lora_r, "seed": args.seed, "train_loss": loss, "heldout_loss": held_loss,
              "train_acc": sum(on_seen) / max(1, len(on_seen)), "train_checks": len(on_seen),
              "heldout_acc": sum(on_held) / max(1, len(on_held)), "heldout_checks": len(on_held)}
    kill_port(8765)
    print(json.dumps(result), flush=True)
    with open(out / "results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")

def results(cfg, entry, out):
    """This config's rows of results.jsonl, by (model, n)."""
    want = (rel(entry), cfg["epochs"], cfg["lr"], cfg["lora_r"], cfg["seed"])
    rows = {}
    for r in jsonl(out / "results.jsonl"):
        if (r["entry"], r["epochs"], r["lr"], r.get("lora_r"), r["seed"]) == want and r["model"] in cfg["models"]:
            rows[(r["model"], r.get("n"))] = r
    return rows

def report(cfg_path, cfg, entry, files, out):
    train_files, held_files = split(cfg, files)
    count = lambda fs: sum(len(jsonl(f.with_suffix(".drafts.jsonl"))) for f in fs)
    module = rel(entry).removeprefix(".lake/packages/mathlib/")
    rows = results(cfg, entry, out)
    n_held = count(held_files)
    for model in cfg["models"]:
        print(f"\n{model} on {module}, {cfg['epochs']} epochs, lr {cfg['lr']}, LoRA r{cfg['lora_r']}")
        print(f"  Each row trains on a different number of drafts (out of {count(train_files)} available).")
        print(f"  Every row is tested on the same {n_held} held-out drafts ({len(held_files)} files, never trained on).")
        print("  Accuracy: generate's subgoal is defeq to the real one. [n] = drafts checked; held acc checks fewer")
        print("  than 'held out' because drafts that restate a hypothesis are skipped, as they are in training.\n")
        print(f"  {'trained on':>10}  {'held out':>8}  {'train loss':>10}  {'held loss':>9}  {'train acc':>13}  {'held acc':>13}")
        done = [rows[(model, n)] for n in cfg["sizes"] if (model, n) in rows]
        for r in done:
            print(f"  {r['n_train']:>10}  {n_held:>8}  {r['train_loss']:>10.3f}  {r['heldout_loss']:>9.3f}  "
                  f"{100 * r['train_acc']:>5.1f}% [{r['train_checks']:>4}]  {100 * r['heldout_acc']:>5.1f}% [{r['heldout_checks']:>4}]")
        missing = [n for n in cfg["sizes"] if (model, n) not in rows]
        if missing:
            print(f"  not run yet: {missing}")
        if len(done) >= 2:
            xs = [math.log(r["n_train"]) for r in done]
            ys = [math.log(r["heldout_loss"]) for r in done]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            print(f"\n  held loss ~= {math.exp(my - b * mx):.3f} * (trained on)^{b:.3f}")
    print(f"\nResults: {(out / 'results.jsonl').as_posix()}")

def main():
    p = argparse.ArgumentParser(
        description="Have-drafter scaling experiment. For every model and training-set size in CONFIG:\n"
                    "fine-tune a LoRA drafter on that many drafts, serve it, and measure how often Canonical's\n"
                    "`generate` proposes the real subgoal (defeq), on trained drafts and on held-out files.\n"
                    "Points already in the results are skipped; a report is printed at the end.",
        epilog="examples (from tools/training):\n"
               "  .venv/Scripts/python drafter_scaling.py configs/drafter-local.toml           run the grid, then report\n"
               "  .venv/Scripts/python drafter_scaling.py configs/drafter-local.toml --report  only print the report\n\n"
               "CONFIG keys: entry, heldout, out, sizes, epochs, lr, lora_r, seed, max_len, eval_train,\n"
               "and [models.\"<hf id>\"] with batch_tokens and accum. See configs/drafter-local.toml.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="experiment config (TOML), e.g. configs/drafter-local.toml")
    p.add_argument("--report", action="store_true", help="only print the report for CONFIG's results; don't train")
    p.add_argument("--extract", action="store_true", help="re-extract drafts from CONFIG's entry even if they exist")
    p.add_argument("--train-only", nargs=2, metavar=("MODEL", "N"), help=argparse.SUPPRESS)
    args = p.parse_args()
    cfg_path = Path(args.config)
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    entry = (LOOP / cfg["entry"]).resolve()
    files, out = lean_files(entry), Path(cfg["out"])
    if args.train_only:
        model, n = args.train_only
        return experiment(cfg_path, cfg, point(cfg, model, int(n)), entry, files, out, train_only=True)
    if args.report:
        return report(cfg_path, cfg, entry, files, out)
    if args.extract or not all(f.with_suffix(".extract.jsonl").is_file() for f in files):
        extract(entry, files, "--drafts")
    if not any(f.with_suffix(".drafts.jsonl").is_file() for f in files):
        raise SystemExit("no drafts extracted")
    done = results(cfg, entry, out)
    for model in cfg["models"]:
        for n in cfg["sizes"]:
            if (model, n) not in done:
                experiment(cfg_path, cfg, point(cfg, model, n), entry, files, out)
    report(cfg_path, cfg, entry, files, out)

if __name__ == "__main__":
    main()
