#!/usr/bin/env python3
"""Mine premises, train, reload the premise server, repeat.

    python tools/training/premise_loop.py [ENTRY]
    # ENTRY: .lean file, directory, or .txt of paths (default Mathlib/Init.lean)
    # Starts premise-server if it is down. Later rounds swap model and db with PUT /config.
    # Flags: --rounds --epochs --batch-size --n-neg --model --out --extract-attempts --load-timeout
"""
import argparse, json, os, random, shutil, subprocess, sys, tarfile, time, urllib.error, urllib.request, zipfile
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

LOOP = Path(__file__).resolve().parent
LLAMA = LOOP.parents[1]
CONVERT = LLAMA / "convert_hf_to_gguf.py"
BASE_MODEL = "l3lab/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne2"
SERVER = "http://127.0.0.1:8766"
LR, TEMP, MAX_LEN = 2e-5, 0.05, 512

def lean_files(entry: Path) -> list[Path]:
    if entry.is_dir():
        return [p for p in entry.rglob("*.lean") if not any(x.startswith(".") for x in p.relative_to(entry).parts)]
    if entry.suffix == ".txt":
        return [(p if (p := Path(s)).is_absolute() else LOOP / p) for line in entry.read_text().splitlines() if (s := line.split("#", 1)[0].strip())]
    return [entry]

def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

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

def train(model_name, rows, out_dir, epochs, batch_size, n_neg):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = pick_device()
    model.to(device).train()
    examples = [
        (row["goal"], pos, (row["negativePremises"] * n_neg)[:n_neg])
        for row in rows if row["premises"] and row["negativePremises"]
        for pos in row["premises"]
    ]
    if not examples:
        raise SystemExit("no premise pairs")
    random.shuffle(examples)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(epochs):
        random.shuffle(examples)
        total, steps = 0.0, 0
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                q = embed(model, tok, [g for g, _, _ in batch], device)
                p = embed(model, tok, [p for _, p, _ in batch], device)
                n = embed(model, tok, [t for *_, ns in batch for t in ns], device).view(len(batch), n_neg, -1)
                scores = torch.cat([(q * p).sum(-1, keepdim=True), torch.einsum("bd,bkd->bk", q, n)], 1)
                loss = F.cross_entropy(scores.float() / TEMP, torch.zeros(len(batch), dtype=torch.long, device=device))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss); steps += 1
        print(f"epoch {epoch + 1} loss {total / steps:.4f}", flush=True)
    out = out_dir / "hf"
    model.save_pretrained(out); tok.save_pretrained(out)
    copy_pooling(model_name, out)
    return out

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

def server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2) as resp:
            return json.loads(resp.read().decode()).get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False

def start_premise(url: str):
    if server_up(url):
        return
    host, port = "127.0.0.1", url.rsplit(":", 1)[-1].split("/")[0]
    exe, model = ensure_llama(), ensure_premise_model()
    args = [str(exe), "--premise", "-m", str(model), "--host", host, "--port", port,
            "--pooling", "mean", "--ctx-size", "512"]
    print("starting premise server", flush=True)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    subprocess.Popen(args, cwd=exe.parent, creationflags=flags, close_fds=True)
    for _ in range(240):
        if server_up(url):
            return
        time.sleep(0.5)
    raise SystemExit("premise server did not come up within 120s")

def run(cmd):
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=LOOP)
    while proc.poll() is None:
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            print("still running", flush=True)
    if proc.returncode:
        raise SystemExit(f"{cmd[0]} failed ({proc.returncode})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("entry", nargs="?", default=str(LOOP / ".lake/packages/mathlib/Mathlib/Init.lean"))
    p.add_argument("--server", default=SERVER)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--out", default=str(LOOP / "out/training"))
    for name, default in [("rounds", 2), ("epochs", 1), ("batch-size", 16), ("n-neg", 8), ("extract-attempts", 5), ("load-timeout", 600)]:
        p.add_argument("--" + name, type=int, default=default)
    args = p.parse_args()
    start_premise(args.server)
    if not (LOOP / ".lake/packages/Canonical/lakefile.lean").is_file():
        print("lake update (mathlib clone is large and quiet)", flush=True)
        run(["lake", "update"])
    entry = Path(args.entry)
    entry = entry.resolve() if entry.is_absolute() else (LOOP / entry).resolve()
    files, model_name = lean_files(entry), args.model
    for rnd in range(args.rounds):
        print(f"extract round {rnd}", flush=True)
        for _ in range(args.extract_attempts):
            try:
                run(["lake", "exe", "extract_data", str(entry)])
                break
            except SystemExit:
                continue
        else:
            raise SystemExit("extract_data failed")
        rows = [row for lean in files for row in jsonl(lean.with_suffix(".premises.jsonl"))]
        if not rows:
            raise SystemExit("extract produced no premise rows")
        round_dir = Path(args.out) / f"round-{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "premises.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        hf = train(model_name, rows, round_dir, args.epochs, args.batch_size, args.n_neg)
        gguf = round_dir / "model.f16.gguf"
        subprocess.check_call([sys.executable, str(CONVERT), str(hf), "--outfile", str(gguf), "--outtype", "f16"])
        db = round_dir / "premise.db"
        for suffix in ("", "-wal", "-shm"):
            db.with_name(db.name + suffix).unlink(missing_ok=True)
        load_config(args.server, db, gguf, args.load_timeout)
        for lean in files:
            for suffix in (".extract.jsonl", ".extract.jsonl.tmp", ".premises.jsonl"):
                lean.with_suffix(suffix).unlink(missing_ok=True)
        model_name = str(hf)

if __name__ == "__main__":
    main()
