#!/usr/bin/env python3
"""End-to-end test of llama-server --premise over HTTP, as the Lean client uses it.

    python tools/server/premise/test_regression.py --model MODEL.gguf [--server-bin build/bin/llama-server]

Starts the server on a fresh database, then checks /version, /cache, /select (with local
declarations), packed batches, and swapping the database and model with PUT /config,
including that rejected swaps leave the served pair unchanged. MODEL defaults to $PREMISE_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
URL = ""

MODULE, TOKEN = "Demo.Module", "demo-v1"
DECLARATIONS = [
    {"name": "Demo.zero_add", "decl": "theorem zero_add (n : Nat) : 0 + n = n"},
    {"name": "Demo.add_zero", "decl": "theorem add_zero (n : Nat) : n + 0 = n"},
]
GOAL = "|- n + 0 = n"
TAIL = {"module": "Demo.Tail", "token": "tail-v1", "declarations": [{"name": "Demo.tail", "decl": "theorem tail : True"}]}
LOCAL = {"name": "Demo.local", "decl": "theorem local (n : Nat) : n + 0 = n"}


def request(path: str, body: dict, method: str = "POST"):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.load(res)


def rejected(path: str, body: dict, method: str = "POST") -> bool:
    try:
        request(path, body, method)
    except urllib.error.HTTPError as error:
        return error.code == 400 and "message" in json.load(error).get("error", {})
    return False


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def select(modules: list[str], goal: str, k: int, declarations: list | None = None) -> list[dict]:
    return request("/select", {"modules": modules, "declarations": declarations or [], "goal": goal, "k": k})


def run(index: Path, other: Path, model: Path) -> None:
    check(request("/version", {"modules": [MODULE]}) == [None], "a new database caches nothing")

    check(request("/cache", {"module": MODULE, "token": TOKEN, "declarations": DECLARATIONS}) == {"ok": True}, "/cache")
    check(request("/version", {"modules": [MODULE, "Missing", MODULE]}) == [TOKEN, None, TOKEN],
          "/version answers each requested module in order")
    hits = select([MODULE], GOAL, 2)
    check([h["name"] for h in hits] == ["Demo.add_zero", "Demo.zero_add"], f"/select ranks the goal's premise first: {hits}")

    with_local = select([MODULE], GOAL, 3, [LOCAL])
    check({h["name"] for h in with_local} == {"Demo.zero_add", "Demo.add_zero", "Demo.local"}, "local declarations are searched")
    check(select([MODULE], GOAL, 3, [LOCAL]) == with_local, "repeating a request gives the same answer")

    check(request("/cache", TAIL) == {"ok": True}, "caching a second module")
    check(request("/cache", {"module": MODULE, "token": "demo-v2", "declarations": [{"name": "Demo.new", "decl": "theorem new : True"}]})
          == {"ok": True}, "re-caching a module")
    check([h["name"] for h in select([MODULE], "|- True", 8)] == ["Demo.new"], "re-caching replaces a module's declarations")
    check([h["name"] for h in select(["Demo.Tail"], "|- True", 8)] == ["Demo.tail"], "re-caching leaves other modules")

    packed = [{"name": f"Demo.batch_{i}", "decl": DECLARATIONS[0]["decl"]} for i in range(64)]
    check(request("/cache", {"module": MODULE, "token": "packed-v1", "declarations": packed}) == {"ok": True}, "caching a packed batch")
    hits = select([MODULE], DECLARATIONS[0]["decl"], 64)
    check(len(hits) == 64 and all(abs(h["score"] - 1.0) < 1e-3 for h in hits),
          "texts embedded in one packed batch match the same text embedded alone")

    check(rejected("/cache", {"module": MODULE}), "/cache without a token is rejected")
    check(rejected("/version", {"modules": [1]}), "/version with a non-string module is rejected")

    served = {"index": str(index), "model": str(model)}
    check(request("/config", {"index": str(other), "model": str(model)}, "PUT") == {"ok": True}, "switching databases")
    check(request("/version", {"modules": ["Demo.Tail"]}) == [None], "the other database is served")
    check(request("/config", served, "PUT") == {"ok": True}, "switching back")
    check(request("/config", served, "PUT") == {"ok": True}, "selecting the served pair again")
    check(request("/version", {"modules": ["Demo.Tail"]}) == ["tail-v1"], "the original database is served again")

    for config in [{"index": "", "model": str(model)}, {"index": str(index) + "\0suffix", "model": str(model)},
                   {"index": str(index.parent / "missing" / "premise.db"), "model": str(model)},
                   {"index": str(index), "model": str(model) + ".missing"}, {"index": str(index)}]:
        check(rejected("/config", config, "PUT"), f"invalid config is rejected: {config}")
        check(request("/version", {"modules": ["Demo.Tail"]}) == ["tail-v1"], f"a rejected config changes nothing: {config}")

    copy = model.with_name(model.name + ".copy.gguf")
    shutil.copyfile(model, copy)
    try:
        check(request("/config", {"index": str(other), "model": str(copy)}, "PUT") == {"ok": True}, "swapping database and model")
        check(request("/cache", TAIL) == {"ok": True}, "caching with the swapped model")
        check(request("/config", served, "PUT") == {"ok": True}, "swapping back")
        check(request("/version", {"modules": [MODULE]}) == ["packed-v1"], "swapping models leaves the database")
    finally:
        copy.unlink(missing_ok=True)


def main() -> int:
    global URL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=os.environ.get("PREMISE_MODEL"), help="GGUF embedding model")
    ap.add_argument("--server-bin", type=Path, default=REPO_ROOT / "build" / "bin" / "llama-server")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    if not args.model or not args.model.is_file():
        sys.exit("pass --model or set PREMISE_MODEL to a GGUF embedding model")
    URL = f"http://127.0.0.1:{args.port}"

    with tempfile.TemporaryDirectory(prefix="premise-test-") as tmp:
        index, other, log = Path(tmp) / "premise.db", Path(tmp) / "other.db", Path(tmp) / "server.log"
        with open(log, "w") as out:
            server = subprocess.Popen([str(args.server_bin), "--premise", "--model", str(args.model), "--pooling", "mean",
                                       "--index-db", str(index), "--host", "127.0.0.1", "--port", str(args.port)],
                                      stdout=out, stderr=subprocess.STDOUT)
        try:
            for _ in range(600):
                if server.poll() is not None:
                    sys.exit(f"llama-server exited with {server.returncode}:\n{log.read_text()}")
                try:
                    urllib.request.urlopen(URL + "/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.2)
            run(index, other, args.model)
        except AssertionError as error:
            print(f"FAIL: {error}", file=sys.stderr)
            return 1
        finally:
            server.terminate()
            server.wait(timeout=10)
    print("premise mode OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
