#!/usr/bin/env python3
"""Mine premises, train, reload the premise server, repeat.

    python tools/training/premise_loop.py [ENTRY]
    # ENTRY: .lean file, directory, or .txt of paths (default Mathlib/Order/Defs/LinearOrder.lean)
    # Starts premise-server if it is down. Later rounds swap model and db with PUT /config.
    # Flags: --rounds --epochs --batch-size --n-neg --lr --lr-later --model --out --extract-attempts --load-timeout --save-every
    # Rerun the same command to resume: finished rounds are skipped and training continues from its checkpoint.
    # Training uses every visible CUDA GPU; --batch-size is per GPU.
"""
import argparse, json, os, random, shutil, socket, subprocess, sys, tarfile, time, urllib.error, urllib.request, zipfile
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

LOOP = Path(__file__).resolve().parent
LLAMA = LOOP.parents[1]
CONVERT = LLAMA / "convert_hf_to_gguf.py"
# The Lean-trained HF checkpoints (l3lab/hanwenzhu ...-ml-ne2) are now empty repos, so the
# base model is rebuilt from the premise server's GGUF on top of its architecture repo.
ARCH_MODEL = "sentence-transformers/all-distilroberta-v1"
SERVER = "http://127.0.0.1:8766"
TEMP, MAX_LEN, WARMUP = 0.05, 512, 0.1

def lean_files(entry: Path) -> list[Path]:
    if entry.is_dir():
        return [p for p in entry.rglob("*.lean") if not any(x.startswith(".") for x in p.relative_to(entry).parts)]
    if entry.suffix == ".txt":
        return [(p if (p := Path(s)).is_absolute() else LOOP / p) for line in entry.read_text(encoding="utf-8").splitlines() if (s := line.split("#", 1)[0].strip())]
    return [entry]

def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    raise SystemExit("no CUDA or MPS device; training on CPU is too slow")

def embed(model, tok, texts, device):
    batch = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

def copy_pooling(src, dst: Path):
    src = Path(src)
    if not (src / "modules.json").is_file():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(str(src), allow_patterns=["modules.json", "1_Pooling/*", "2_Normalize/*"]))
    shutil.copy(src / "modules.json", dst / "modules.json")
    for mod in json.loads((src / "modules.json").read_text()):
        path = src / mod.get("path", "")
        if path.is_dir() and mod.get("path"):
            shutil.copytree(path, dst / path.name, dirs_exist_ok=True)

def train(model_name, data, out_dir, epochs, batch_size, n_neg, save_every, lr):
    """Train on every visible CUDA GPU (one process each), else on the one available device."""
    world = torch.cuda.device_count() if pick_device() == "cuda" else 1
    args = (model_name, data, out_dir, epochs, batch_size, n_neg, save_every, lr)
    if world > 1:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            init = f"tcp://127.0.0.1:{s.getsockname()[1]}"
        print(f"training on {world} GPUs", flush=True)
        torch.multiprocessing.spawn(train_worker, (world, init) + args, nprocs=world)
    else:
        train_worker(0, 1, None, *args)
    return out_dir / "hf"

def train_worker(rank, world, init, model_name, data, out_dir, epochs, batch_size, n_neg, save_every, lr):
    kind = device = pick_device()
    if world > 1:
        torch.distributed.init_process_group("nccl" if kind == "cuda" else "gloo", init_method=init, rank=rank, world_size=world)
        if kind == "cuda":
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
    log = print if rank == 0 else (lambda *a, **k: None)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device).train()
    # the pooler is unused by mean pooling, so it never has a gradient on any rank
    params = [p for name, p in model.named_parameters() if not name.startswith("pooler.")]
    examples = [
        (row["goal"], pos, (row["negativePremises"] * n_neg)[:n_neg])
        for row in jsonl(data) if row["premises"] and row["negativePremises"]
        for pos in row["premises"]
    ]
    if not examples:
        raise SystemExit("no premise pairs")
    global_batch = batch_size * world
    steps_per_epoch = -(-len(examples) // global_batch)
    total_steps = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * total_steps), total_steps)
    ckpt, start = out_dir / "checkpoint.pt", 0
    if ckpt.is_file():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        start = state["step"]
        log(f"resume from step {start}/{total_steps}", flush=True)
    for epoch in range(start // steps_per_epoch, epochs):
        # seeded per epoch so a resumed run replays the same batch order
        order = list(range(len(examples)))
        random.Random(epoch).shuffle(order)
        total, steps = 0.0, 0
        for b in range(max(start - epoch * steps_per_epoch, 0), steps_per_epoch):
            batch = [examples[j] for j in order[b * global_batch:(b + 1) * global_batch][rank::world]]
            opt.zero_grad()
            if batch:  # empty only on an epoch's last step, when fewer examples than GPUs remain
                with torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda"):
                    q = embed(model, tok, [g for g, _, _ in batch], device)
                    p = embed(model, tok, [p for _, p, _ in batch], device)
                    n = embed(model, tok, [t for *_, ns in batch for t in ns], device).view(len(batch), n_neg, -1)
                    scores = torch.cat([(q * p).sum(-1, keepdim=True), torch.einsum("bd,bkd->bk", q, n)], 1)
                    loss = F.cross_entropy(scores.float() / TEMP, torch.zeros(len(batch), dtype=torch.long, device=device))
                loss.backward()
                total += float(loss); steps += 1
            if world > 1:
                # one all-reduce per step keeps every replica's update identical
                flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
                torch.distributed.all_reduce(flat)
                flat /= world
                for p, g in zip(params, flat.split([p.numel() for p in params])):
                    p.grad = g.view_as(p)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            step = epoch * steps_per_epoch + b + 1
            if rank == 0 and step % save_every == 0 and step < total_steps:
                part = ckpt.with_name(ckpt.name + ".part")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, part)
                part.replace(ckpt)
                log(f"step {step}/{total_steps} loss {total / max(steps, 1):.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        log(f"epoch {epoch + 1} loss {total / max(steps, 1):.4f}", flush=True)
    if rank == 0:
        out, part = out_dir / "hf", out_dir / "hf.part"
        shutil.rmtree(part, ignore_errors=True)
        model.save_pretrained(part); tok.save_pretrained(part)
        copy_pooling(model_name, part)
        part.rename(out)
        ckpt.unlink(missing_ok=True)
    if world > 1:
        torch.distributed.destroy_process_group()

def to_gguf(hf, gguf: Path):
    part = gguf.with_name(gguf.name + ".part")
    subprocess.check_call([sys.executable, str(CONVERT), str(hf), "--outfile", str(part), "--outtype", "f16"])
    part.replace(gguf)

def start_gguf(model, out: Path) -> Path:
    """GGUF of the starting model, so round 0 mines with the model it trains."""
    if not model:
        return ensure_premise_model()
    gguf = out / "base" / "model.f16.gguf"
    if not gguf.is_file():
        # re-saved like the round outputs: a bare RobertaModel, which convert supports and *ForMaskedLM is not
        hf = gguf.parent / "hf"
        AutoModel.from_pretrained(model).save_pretrained(hf); AutoTokenizer.from_pretrained(model).save_pretrained(hf)
        copy_pooling(model, hf)
        to_gguf(hf, gguf)
    return gguf

def load_config(server, db, gguf, timeout):
    url = server.rstrip("/") + "/config"
    config = {"index": str(Path(db).resolve()), "model": str(Path(gguf).resolve())}
    print(f"PUT /config model={config['model']} index={config['index']}", flush=True)
    req = urllib.request.Request(url, data=json.dumps(config).encode(), headers={"Content-Type": "application/json"}, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"PUT {url} {exc.code}: {exc.read().decode(errors='replace')}") from exc
    if not body.get("ok"):
        raise SystemExit(f"PUT {url} {body}")

LLAMA_TAG = "premise-v12"
PREMISE_GGUF = ("chasenorman/lean-premise-distilroberta-GGUF", "thomas-zhu-lean-premise.f16.gguf")

def cache_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Canonical"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "Canonical"
    return Path.home() / ".cache" / "Canonical"

def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    print(f"download {url}", flush=True)
    part = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, part)
    part.replace(dest)

def ensure_llama() -> Path:
    root = cache_dir() / "llama.cpp" / "release" / LLAMA_TAG
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    found = next(root.rglob(name), None) if root.is_dir() else None
    if found:
        return found
    suffix = "bin-win-vulkan-x64.zip" if os.name == "nt" else "bin-ubuntu-vulkan-x64.tar.gz"
    if sys.platform == "darwin":
        suffix = "bin-macos-arm64.tar.gz"
    archive = root / f"llama-{LLAMA_TAG}-{suffix}"
    download(f"https://github.com/FrederickPu/llama.cpp/releases/download/{LLAMA_TAG}/{archive.name}", archive)
    root.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        zipfile.ZipFile(archive).extractall(root)
    else:
        tarfile.open(archive).extractall(root)
    archive.unlink(missing_ok=True)
    found = next(root.rglob(name), None)
    if not found:
        raise SystemExit(f"{archive.name} did not contain {name}")
    return found

def ensure_premise_model() -> Path:
    repo, filename = PREMISE_GGUF
    dest = cache_dir() / repo / filename
    download(f"https://huggingface.co/{repo}/resolve/main/{filename}", dest)
    return dest

def hf_snapshot(repo: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download
    for attempt in range(10):
        try:
            return Path(snapshot_download(repo, allow_patterns=patterns))
        except Exception as exc:  # the hub drops connections intermittently
            print(f"retry {repo}: {exc!r}"[:200], flush=True)
            time.sleep(3)
    raise SystemExit(f"could not download {repo}")

GGUF_TO_HF = {
    "token_embd": "embeddings.word_embeddings", "position_embd": "embeddings.position_embeddings",
    "token_types": "embeddings.token_type_embeddings", "token_embd_norm": "embeddings.LayerNorm",
    "attn_q": "attention.self.query", "attn_k": "attention.self.key", "attn_v": "attention.self.value",
    "attn_output": "attention.output.dense", "attn_output_norm": "attention.output.LayerNorm",
    "ffn_up": "intermediate.dense", "ffn_down": "output.dense", "layer_output_norm": "output.LayerNorm",
}

def ensure_base_model() -> Path:
    """HF checkpoint of the premise server's GGUF: its weights loaded into `ARCH_MODEL`."""
    out = cache_dir() / "lean-premise-hf"
    if (out / "model.safetensors").is_file():
        return out
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    import re, numpy as np, gguf
    arch = hf_snapshot(ARCH_MODEL, ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"])
    model, tok = AutoModel.from_pretrained(arch), AutoTokenizer.from_pretrained(arch)
    sd = model.state_dict()
    reader = gguf.GGUFReader(ensure_premise_model())
    toks = reader.fields["tokenizer.ggml.tokens"]
    if [bytes(toks.parts[i]).decode("utf-8") for i in toks.data] != [t for t, _ in sorted(tok.get_vocab().items(), key=lambda kv: kv[1])]:
        raise SystemExit(f"premise GGUF vocab differs from {ARCH_MODEL}")
    loaded = set()
    for t in reader.tensors:
        stem, kind = t.name.rsplit(".", 1)
        m = re.fullmatch(r"blk\.(\d+)\.(\w+)", stem)
        key = f"encoder.layer.{m[1]}.{GGUF_TO_HF[m[2]]}.{kind}" if m else f"{GGUF_TO_HF[stem]}.{kind}"
        data = torch.from_numpy(np.array(t.data, dtype=np.float32))
        # convert_hf_to_gguf drops RoBERTa's first `1 + pad_token_id` position rows
        target = sd[key][sd[key].shape[0] - data.numel() // sd[key][0].numel():] if key == "embeddings.position_embeddings.weight" else sd[key]
        target.copy_(data.reshape(target.shape))
        loaded.add(key)
    if missing := {k for k in sd if not k.startswith("pooler.")} - loaded:
        raise SystemExit(f"premise GGUF is missing {sorted(missing)}")
    model.load_state_dict(sd)
    model.save_pretrained(out); tok.save_pretrained(out)
    copy_pooling(arch, out)
    return out

def server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as resp:
            return json.loads(resp.read().decode()).get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False

def start_premise(url: str, log: Path):
    if server_up(url):
        return
    host, port = "127.0.0.1", url.rsplit(":", 1)[-1].split("/")[0]
    exe, model = ensure_llama(), ensure_premise_model()
    # llama.cpp sizes its pool to every core on the node, not the job's share, and its spinning
    # threads would starve extract_data; with the model on the GPU a few threads are plenty
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    threads = str(min(4, cores))
    args = [str(exe), "--premise", "-m", str(model), "--host", host, "--port", port,
            "--pooling", "mean", "--ctx-size", "512", "--threads", threads, "--threads-batch", threads]
    print(f"starting premise server, log {log}", flush=True)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen(args, cwd=exe.parent, creationflags=flags, close_fds=True, stdout=out, stderr=subprocess.STDOUT)
    for _ in range(240):
        if server_up(url):
            return
        time.sleep(0.5)
    raise SystemExit("premise server did not come up within 120s")

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    # extract_data survives the occasional Lean panic, but each one would print a ~100-line backtrace
    env = {"LEAN_BACKTRACE": "0", **os.environ}
    proc, start = subprocess.Popen(cmd, cwd=LOOP, env=env), time.monotonic()
    while proc.poll() is None:
        try:
            proc.wait(600)
        except subprocess.TimeoutExpired:
            print(f"{cmd[0]} still running after {(time.monotonic() - start) / 60:.0f} min", flush=True)
    if proc.returncode:
        raise SystemExit(f"{cmd[0]} failed ({proc.returncode})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("entry", nargs="?", default=str(LOOP / ".lake/packages/mathlib/Mathlib/Order/Defs/LinearOrder.lean"))
    p.add_argument("--server", default=SERVER)
    p.add_argument("--model", help="HF model to start from (default: the premise server's GGUF)")
    # training from a general-purpose model needs larger steps than refining a premise selector
    p.add_argument("--lr", type=float, help="first-round learning rate (default: 1e-4 with --model, else 2e-5)")
    p.add_argument("--lr-later", type=float, default=2e-5, help="learning rate for later rounds, which refine the previous round")
    p.add_argument("--out", default=str(LOOP / "out/training"))
    for name, default in [("rounds", 2), ("epochs", 1), ("batch-size", 16), ("n-neg", 3), ("extract-attempts", 5), ("load-timeout", 600), ("save-every", 1000)]:
        p.add_argument("--" + name, type=int, default=default)
    args = p.parse_args()
    start_premise(args.server, Path(args.out) / "premise-server.log")
    if not (LOOP / ".lake/packages/Canonical/lakefile.lean").is_file():
        print("lake update (mathlib clone is large and quiet)", flush=True)
        run(["lake", "update"])
    entry = Path(args.entry)
    entry = entry.resolve() if entry.is_absolute() else (LOOP / entry).resolve()
    files, model_name = lean_files(entry), args.model or str(ensure_base_model())
    prev = (Path(args.out) / "base" / "premise.db", start_gguf(args.model, Path(args.out)))
    for rnd in range(args.rounds):
        round_dir = Path(args.out) / f"round-{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)
        data, hf, gguf, db = (round_dir / n for n in ("premises.jsonl", "hf", "model.f16.gguf", "premise.db"))
        if gguf.is_file():
            print(f"round {rnd} already trained", flush=True)
        else:
            # mine with the previous round's (or the starting) model; also restores it after a server restart
            load_config(args.server, *prev, args.load_timeout)
            if not data.is_file():
                print(f"extract round {rnd}", flush=True)
                for _ in range(args.extract_attempts):
                    try:
                        run(["lake", "-q", "exe", "extract_data", str(entry)])
                        break
                    except SystemExit:
                        continue
                else:
                    raise SystemExit("extract_data failed")
                rows = [row for lean in files for row in jsonl(lean.with_suffix(".premises.jsonl"))]
                if not rows:
                    raise SystemExit("extract produced no premise rows")
                part = data.with_name(data.name + ".part")
                part.write_text("".join(json.dumps(row) + "\n" for row in rows))
                part.replace(data)
            if not hf.is_dir():
                lr = (args.lr or (1e-4 if args.model else 2e-5)) if rnd == 0 else args.lr_later
                train(model_name, data, round_dir, args.epochs, args.batch_size, args.n_neg, args.save_every, lr)
            # before the gguf exists, so a finished round never leaves stale extraction behind
            for lean in files:
                for suffix in (".extract.jsonl", ".extract.jsonl.tmp", ".premises.jsonl"):
                    lean.with_suffix(suffix).unlink(missing_ok=True)
            to_gguf(hf, gguf)
            for suffix in ("", "-wal", "-shm"):
                db.with_name(db.name + suffix).unlink(missing_ok=True)
        prev, model_name = (db, gguf), str(hf)
    load_config(args.server, *prev, args.load_timeout)

if __name__ == "__main__":
    main()
