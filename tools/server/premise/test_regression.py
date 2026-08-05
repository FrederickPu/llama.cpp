"""
Demo and regression test for llama-server --premise.

Premise mode is embedding-only retrieval used by CanonicalDrafter to rank Lean
premises. It does not generate text. The intended flow is:

  1. Start: llama-server --premise --model ... --index-vecs ... --index-names ...
  2. Ask /version whether one Lean module is already cached.
  3. Send that module's declarations to /cache when the version token is missing or stale.
  4. Call /select with imports, local declarations, a goal, and k.

Run with --print-guide to see the equivalent manual commands and JSON bodies.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parents[3]
LLAMA_DIR = REPO_ROOT
BUILD_DIR = REPO_ROOT / "build"
BUILD_BIN_DIR = BUILD_DIR / "bin"

CMAKE_CONFIGURE_ARGS = [
    "-DLLAMA_BUILD_COMMON=ON",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_LIBRESSL=ON",
    "-DPremiseServer=ON",
]

DEFAULT_MODEL = Path("D:/hparam_outputs/thomas-zhu-lean-premise.f16.gguf")
DEFAULT_INDEX_VECS = BUILD_DIR / "premise-demo-vectors.bin"
DEFAULT_INDEX_NAMES = BUILD_DIR / "premise-demo-names.bin"
THOMAS_ZHU_MODEL_ID = "l3lab/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne2"
THOMAS_ZHU_MODEL_REVISION = "v4.30.0"

BASE_URL = "http://127.0.0.1:8081"
DEMO_MODULE = "Demo.Module"
DEMO_TOKEN = "premise-server-demo-v1"
DEMO_DECLARATIONS = [
    {"name": "Demo.zero_add", "decl": "theorem zero_add (n : Nat) : 0 + n = n"},
    {"name": "Demo.add_zero", "decl": "theorem add_zero (n : Nat) : n + 0 = n"},
]
DEMO_GOAL = "|- n + 0 = n"


def die(message: str) -> None:
    raise SystemExit(message)


def find_exe(name: str, build_dir: Path = BUILD_DIR) -> Path | None:
    pattern = f"**/{name}.exe" if os.name == "nt" else f"**/{name}"
    matches = sorted(build_dir.glob(pattern))
    return matches[0] if matches else None


def configure_build() -> None:
    if os.name == "nt":
        vsdev = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat")
        if vsdev.exists():
            args = " ".join(CMAKE_CONFIGURE_ARGS)
            command = f'call "{vsdev}" -arch=x64 && cmake -S "{LLAMA_DIR}" -B "{BUILD_DIR}" -G "NMake Makefiles" {args}'
            subprocess.check_call(["cmd.exe", "/c", command], cwd=REPO_ROOT)
            return
    subprocess.check_call(["cmake", "-S", str(LLAMA_DIR), "-B", str(BUILD_DIR), *CMAKE_CONFIGURE_ARGS], cwd=REPO_ROOT)


def build_target(target: str) -> None:
    configure_build()
    if os.name == "nt":
        vsdev = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat")
        if vsdev.exists():
            command = f'call "{vsdev}" -arch=x64 && cmake --build "{BUILD_DIR}" --target {target}'
            subprocess.check_call(["cmd.exe", "/c", command], cwd=REPO_ROOT)
            return
    subprocess.check_call(["cmake", "--build", str(BUILD_DIR), "--target", target], cwd=REPO_ROOT)


def find_llama_server(server_bin: Path | None) -> Path:
    if server_bin:
        return server_bin
    server = find_exe("llama-server")
    if server:
        return server

    print("llama-server not found; building target llama-server...", flush=True)
    build_target("llama-server")
    server = find_exe("llama-server")
    if not server:
        die("Failed to locate llama-server after building it")
    return server


def premise_server_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(find_llama_server(args.server_bin)),
        "--premise",
        "--host", args.host,
        "--port", str(args.port),
        "--model", str(args.model),
        "--pooling", args.pooling,
        "--ctx-size", str(args.ctx_size),
        "--index-vecs", str(args.index_vecs),
        "--index-names", str(args.index_names),
    ]
    command.extend(args.server_arg or [])
    return command


class ManagedPremiseServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def __enter__(self) -> "ManagedPremiseServer":
        if not self.args.model.exists():
            die(
                f"Premise GGUF does not exist: {self.args.model}\n"
                f"Expected Thomas Zhu model: {THOMAS_ZHU_MODEL_ID} revision {THOMAS_ZHU_MODEL_REVISION}"
            )
        self.args.index_vecs.parent.mkdir(parents=True, exist_ok=True)
        self.args.index_names.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if BUILD_BIN_DIR.exists():
            env["PATH"] = str(BUILD_BIN_DIR) + os.pathsep + env.get("PATH", "")

        log_path = self.args.server_log or (BUILD_DIR / "premise-server-regression.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

        command = premise_server_command(self.args)
        print("Starting llama-server --premise:", " ".join(command), flush=True)

        creationflags = 0
        if os.name == "nt":
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
            creationflags |= subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=self.log_file,
            stderr=self.log_file,
            creationflags=creationflags,
        )
        self.wait_until_ready()
        return self

    def wait_until_ready(self) -> None:
        deadline = time.time() + self.args.startup_timeout
        health_url = f"http://{self.args.host}:{self.args.port}/health"

        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                die(f"llama-server exited early with code {self.process.returncode}; see {self.log_file.name}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        print(f"premise mode ready at {health_url}", flush=True)
                        return
            except Exception:
                time.sleep(0.5)

        die(f"Timed out waiting for llama-server --premise; see {self.log_file.name}")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process and self.process.poll() is None:
            print("Stopping llama-server", flush=True)
            if os.name == "nt":
                try:
                    os.kill(self.process.pid, signal.CTRL_C_EVENT)
                except OSError:
                    self.process.terminate()
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_file:
            self.log_file.close()


def post_json(path: str, body: dict, timeout: int = 120):
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def cache_body() -> dict:
    return {
        "module": DEMO_MODULE,
        "imports": [],
        "declarations": DEMO_DECLARATIONS,
        "token": DEMO_TOKEN,
    }


def select_body(k: int = 2) -> dict:
    return {
        "imports": [DEMO_MODULE],
        "declarations": [],
        "goal": DEMO_GOAL,
        "k": k,
    }


def dump_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def run_tests(args: argparse.Namespace) -> bool:
    print(f"Using premise mode at {BASE_URL}", flush=True)

    print(f"1. Checking /version for uncached module {DEMO_MODULE!r}", flush=True)
    before = post_json("/version", {"module": DEMO_MODULE})
    if before is not None and not isinstance(before, str):
        print(f"[FAIL] expected null or a version token string, got {before!r}", file=sys.stderr)
        return False
    if before is not None:
        print(f"   module already has token {before!r}; refreshing it with {DEMO_TOKEN!r}", flush=True)

    print(f"2. Sending {len(DEMO_DECLARATIONS)} declarations to /cache", flush=True)
    cache_result = post_json("/cache", cache_body())
    if cache_result != {"ok": True}:
        print(f"[FAIL] /cache returned {cache_result!r}", file=sys.stderr)
        return False

    print("3. Verifying /version returns the cached token", flush=True)
    after = post_json("/version", {"module": DEMO_MODULE})
    if after != DEMO_TOKEN:
        print(f"[FAIL] expected cached module token, got {after!r}", file=sys.stderr)
        return False

    print(f"4. Asking /select for premises relevant to goal {DEMO_GOAL!r}", flush=True)
    suggestions = post_json("/select", select_body())
    if len(suggestions) != 2 or not all("name" in item and "score" in item for item in suggestions):
        print(f"[FAIL] /select returned malformed suggestions: {suggestions!r}", file=sys.stderr)
        return False

    print("premise mode demo OK. Suggestions:", flush=True)
    print(json.dumps(suggestions, indent=2), flush=True)
    return True


def print_usage_guide(args: argparse.Namespace) -> None:
    server_exe = args.server_bin or find_exe("llama-server") or (BUILD_BIN_DIR / ("llama-server.exe" if os.name == "nt" else "llama-server"))
    version_request = dump_json({"module": DEMO_MODULE})
    cache_request = dump_json(cache_body())
    select_request = dump_json(select_body())
    line_continue = "^" if os.name == "nt" else "\\"
    guide = f"""Manual equivalent of this regression runner
==========================================

Model
-----

This demo expects a GGUF embedding model compatible with Lean premise retrieval.
The default is:

  Hugging Face: {THOMAS_ZHU_MODEL_ID}
  Revision:     {THOMAS_ZHU_MODEL_REVISION}
  Local GGUF:   {args.model}

Download or convert the model separately, then pass its GGUF path with --model.

Build
-----

   cmake -S {LLAMA_DIR} -B {BUILD_DIR} {' '.join(CMAKE_CONFIGURE_ARGS)}
   cmake --build {BUILD_DIR} --target llama-server

Start
-----

The cache is a pair of files rewritten together on /cache:
  --index-vecs  : FAISS IndexFlatIP embeddings
  --index-names : module graph + declaration names in FAISS row order
Pretty-printed declaration strings are sent by Lean only to compute embeddings;
they are not stored in either file.

   {server_exe} --premise --host {args.host} --port {args.port} --model {args.model} {line_continue}
       --pooling {args.pooling} --ctx-size {args.ctx_size} {line_continue}
       --index-vecs {args.index_vecs} {line_continue}
       --index-names {args.index_names}

API
---

1. Check whether the module token is already cached in this process.

   POST http://{args.host}:{args.port}/version
   Content-Type: application/json

{textwrap.indent(version_request, "   ")}

   Expected before /cache: null

2. Cache or refresh declarations for that one module (declarations are
   batch-embedded in a single forward pass).

   POST http://{args.host}:{args.port}/cache
   Content-Type: application/json

{textwrap.indent(cache_request, "   ")}

   Expected response: {{"ok":true}}

3. Confirm the module token now matches.

   POST http://{args.host}:{args.port}/version
   Content-Type: application/json

{textwrap.indent(version_request, "   ")}

   Expected after /cache: "{DEMO_TOKEN}"

4. Select premises for a goal.

   The imports list names modules that were populated by /cache. The
   declarations field is for local, one-off candidates that should be considered
   for this request but not stored as a module cache entry.

   POST http://{args.host}:{args.port}/select
   Content-Type: application/json

{textwrap.indent(select_request, "   ")}

   Expected response shape:

   [{{"name":"Demo.add_zero","score":0.87}}, ...]

Notes
-----

  - /version is an in-memory module freshness check; it resets on restart.
  - --index-vecs stores embeddings; --index-names stores module/name identity.
    Both are updated together. Declaration strings are not persisted.
  - Lean decides which declarations belong to each module and sends fully
    qualified names through /cache; the server does not infer module membership.
  - /version and /cache are one module per request.
  - /select can include unsaved local declarations in the request body.
  - --url runs the same demo against an existing premise-mode server and will
    refresh {DEMO_MODULE} in that process.
"""
    print(guide.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start llama-server --premise and demonstrate /version, /cache, and /select.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Common flows:
          python tools/server/premise/test_regression.py --print-guide
          python tools/server/premise/test_regression.py
          python tools/server/premise/test_regression.py --url http://127.0.0.1:8081
        """),
    )
    parser.add_argument("--print-guide", action="store_true", help="Print manual build/start/query commands and exit")
    parser.add_argument("--url", help="Use an already-running premise-mode server; refreshes the demo module in that process")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="GGUF embedding model")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--index-vecs", type=Path, default=DEFAULT_INDEX_VECS, help="Persistent embedding vector cache")
    parser.add_argument("--index-names", dest="index_names", type=Path, default=DEFAULT_INDEX_NAMES, help="Module/name side of the persistent cache pair")
    parser.add_argument("--index-strings", dest="index_names", type=Path, help="Deprecated alias for --index-names")
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--server-bin", type=Path, help="Path to llama-server executable")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--server-arg", action="append", help="Extra argument passed to llama-server; repeat as needed")
    return parser.parse_args()


def main() -> int:
    global BASE_URL
    args = parse_args()

    if args.print_guide:
        print_usage_guide(args)
        return 0

    if args.url:
        BASE_URL = args.url.rstrip("/")
        return 0 if run_tests(args) else 1

    BASE_URL = f"http://{args.host}:{args.port}"
    with ManagedPremiseServer(args):
        return 0 if run_tests(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
