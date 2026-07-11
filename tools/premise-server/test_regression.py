"""
Premise-server regression test and usage guide.

This script builds and starts the standalone `premise-server` tool, then
exercises the Lean premise API:

  - /version
  - /cache
  - /select

"""

from __future__ import annotations

import argparse
import json
import os
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


REPO_ROOT = Path(__file__).resolve().parents[2]
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
THOMAS_ZHU_MODEL_ID = "l3lab/all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne2"
THOMAS_ZHU_MODEL_REVISION = "v4.30.0"

BASE_URL = "http://127.0.0.1:8081"


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


def find_premise_server(server_bin: Path | None) -> Path:
    if server_bin:
        return server_bin
    premise_server = find_exe("premise-server")
    if premise_server:
        return premise_server

    print("premise-server not found; building target premise-server...", flush=True)
    build_target("premise-server")
    premise_server = find_exe("premise-server")
    if not premise_server:
        die("Failed to locate premise-server after building it")
    return premise_server


def premise_server_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(find_premise_server(args.server_bin)),
        "--host", args.host,
        "--port", str(args.port),
        "--model", str(args.model),
        "--pooling", args.pooling,
        "--ctx-size", str(args.ctx_size),
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
                f"Premise-server GGUF does not exist: {self.args.model}\n"
                f"Expected Thomas Zhu model: {THOMAS_ZHU_MODEL_ID} revision {THOMAS_ZHU_MODEL_REVISION}"
            )

        env = os.environ.copy()
        if BUILD_BIN_DIR.exists():
            env["PATH"] = str(BUILD_BIN_DIR) + os.pathsep + env.get("PATH", "")

        log_path = self.args.server_log or (BUILD_DIR / "premise-server-regression.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

        command = premise_server_command(self.args)
        print("Starting premise-server:", " ".join(command), flush=True)

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
                die(f"premise-server exited early with code {self.process.returncode}; see {self.log_file.name}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        print(f"premise-server ready at {health_url}", flush=True)
                        return
            except Exception:
                time.sleep(0.5)

        die(f"Timed out waiting for premise-server; see {self.log_file.name}")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process and self.process.poll() is None:
            print("Stopping premise-server", flush=True)
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


def run_tests(args: argparse.Namespace) -> bool:
    global BASE_URL
    BASE_URL = f"http://{args.host}:{args.port}"

    before = post_json("/version", {"module": "Demo.Module"})
    if before is not None:
        print(f"[FAIL] expected uncached module version to be null, got {before!r}", file=sys.stderr)
        return False

    cache_body = {
        "module": "Demo.Module",
        "imports": [],
        "declarations": [
            {"name": "Demo.zero_add", "decl": "theorem zero_add (n : Nat) : 0 + n = n"},
            {"name": "Demo.add_zero", "decl": "theorem add_zero (n : Nat) : n + 0 = n"},
        ],
        "token": "premise-server-demo-v1",
    }
    cache_result = post_json("/cache", cache_body)
    if cache_result != {"ok": True}:
        print(f"[FAIL] /cache returned {cache_result!r}", file=sys.stderr)
        return False

    after = post_json("/version", {"module": "Demo.Module"})
    if after != "premise-server-demo-v1":
        print(f"[FAIL] expected cached module token, got {after!r}", file=sys.stderr)
        return False

    suggestions = post_json("/select", {
        "imports": ["Demo.Module"],
        "declarations": [],
        "goal": "|- n + 0 = n",
        "k": 2,
    })
    if len(suggestions) != 2 or not all("name" in item and "score" in item for item in suggestions):
        print(f"[FAIL] /select returned malformed suggestions: {suggestions!r}", file=sys.stderr)
        return False

    print("premise-server regression OK:", suggestions, flush=True)
    return True


def print_usage_guide(args: argparse.Namespace) -> None:
    premise_server_exe = args.server_bin or find_exe("premise-server") or (BUILD_BIN_DIR / ("premise-server.exe" if os.name == "nt" else "premise-server"))
    guide = f"""Manual equivalent of this regression runner
==========================================

1. Build premise-server:

   cmake -S {LLAMA_DIR} -B {BUILD_DIR} {' '.join(CMAKE_CONFIGURE_ARGS)}
   cmake --build {BUILD_DIR} --target premise-server

2. Start premise-server:

   Model: {THOMAS_ZHU_MODEL_ID} revision {THOMAS_ZHU_MODEL_REVISION}

   {premise_server_exe} --host {args.host} --port {args.port} --model {args.model} ^
       --pooling {args.pooling} --ctx-size {args.ctx_size}

3. Query the Lean premise API:

   POST http://{args.host}:{args.port}/version
   POST http://{args.host}:{args.port}/cache
   POST http://{args.host}:{args.port}/select

Expected behavior:

   /version returns null before /cache and the token after /cache
   /select returns premise suggestions
"""
    print(guide.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start premise-server and test /version, /cache, and /select.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Common flows:
          python tools/premise-server/test_regression.py --print-guide
          python tools/premise-server/test_regression.py
          python tools/premise-server/test_regression.py --url http://127.0.0.1:8081
        """),
    )
    parser.add_argument("--print-guide", action="store_true", help="Print manual build/start/query commands and exit")
    parser.add_argument("--url", help="Use an already-running premise-server instead of starting one")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="GGUF embedding model")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--server-bin", type=Path, help="Path to premise-server executable")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--server-arg", action="append", help="Extra argument passed to premise-server; repeat as needed")
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

    with ManagedPremiseServer(args):
        return 0 if run_tests(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
