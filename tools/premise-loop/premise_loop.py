#!/usr/bin/env python3
import argparse, json, subprocess, sys, time, urllib.error, urllib.request
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
        return [Path(s) for line in entry.read_text().splitlines() if (s := line.split("#", 1)[0].strip())]
    return [entry]

def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []

def embed(model, tok, texts, device):
    batch = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)

def train(model_name, rows, out_dir, epochs, batch_size, n_neg):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model.to(dtype=torch.bfloat16)
    model.to(device).train()
    examples = [
        (row["goal"], pos, (row["negativePremises"] * n_neg)[:n_neg])
        for row in rows if row["premises"] and row["negativePremises"]
        for pos in row["premises"]
    ]
    if not examples:
        raise SystemExit("no premise pairs")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for epoch in range(epochs):
        total, steps = 0.0, 0
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
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
    return out

def load_model(server, gguf, timeout):
    url = server.rstrip("/") + "/model"
    req = urllib.request.Request(url, data=json.dumps({"path": str(Path(gguf).resolve())}).encode(), headers={"Content-Type": "application/json"}, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"PUT {url} {exc.code}: {exc.read().decode(errors='replace')}") from exc
    if not body.get("ok"):
        raise SystemExit(f"PUT {url} {body}")
    health = server.rstrip("/") + "/health"
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if json.loads(resp.read().decode()).get("status") == "ok":
                    return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1)
    raise SystemExit("premise server not healthy")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("entry", nargs="?", default=str(LOOP / ".lake/packages/mathlib/Mathlib"))
    p.add_argument("--server", default=SERVER)
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--out", default=str(LOOP / "out/premise-loop"))
    for name, default in [("rounds", 2), ("epochs", 1), ("batch-size", 16), ("n-neg", 8), ("extract-attempts", 5), ("load-timeout", 600)]:
        p.add_argument("--" + name, type=int, default=default)
    args = p.parse_args()
    if not (LOOP / ".lake/packages/CanonicalDrafter/lakefile.toml").is_file():
        subprocess.check_call(["lake", "update"], cwd=LOOP)
    entry = Path(args.entry)
    entry = entry.resolve() if entry.is_absolute() else (LOOP / entry).resolve()
    files, model_name = lean_files(entry), args.model
    for rnd in range(args.rounds):
        for _ in range(args.extract_attempts):
            if subprocess.call(["lake", "exe", "CanonicalDrafter/extract_data", str(entry)], cwd=LOOP) == 0:
                break
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
        load_model(args.server, gguf, args.load_timeout)
        for lean in files:
            for suffix in (".extract.jsonl", ".extract.jsonl.tmp", ".premises.jsonl"):
                lean.with_suffix(suffix).unlink(missing_ok=True)
        model_name = str(hf)

if __name__ == "__main__":
    main()
