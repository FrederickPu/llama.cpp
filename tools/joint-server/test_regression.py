"""
Joint-server regression test and usage guide.

This file is intentionally written as a runnable recipe for the full local flow:

  1. Quantize the trained F16 model if the quantized GGUF is missing.
  2. Start joint-server with the quantized model and premise index.
  3. Query /v1/chat/completions with streaming enabled.
  4. Assert that premises arrive before generated tool-call output.

Default one-command run:

  conda run -n torch-env python tools/joint-server/test_regression.py

Show the equivalent manual commands:

  conda run -n torch-env python tools/joint-server/test_regression.py --print-guide

Run against an already-started server:

  conda run -n torch-env python tools/joint-server/test_regression.py --url http://127.0.0.1:8080

Run the old dataset smoke-query loop after the regression assertions:

  conda run -n torch-env python tools/joint-server/test_regression.py --dataset-n 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# -----------------------------------------------------------------------------
# Default Local Artifacts
# -----------------------------------------------------------------------------

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
    "-DJointServer=ON",
]

DEFAULT_MODEL = Path("D:/hparam_outputs/joint-qwen.f16.gguf")
DEFAULT_INDEX_VECS = Path("D:/hparam_outputs/premise_vectors.bin")
DEFAULT_INDEX_STRINGS = Path("D:/hparam_outputs/premise_strings.json")
DEFAULT_QUANT = "Q8_0"

BASE_URL = "http://127.0.0.1:8080"
DRAFTS_ID = "chasenorman/subproofs-mathlib-v4.30.0"


# -----------------------------------------------------------------------------
# Request Payloads Used By The Tests
# -----------------------------------------------------------------------------

HAVE_TOOL = {
    "type": "function",
    "function": {
        "name": "have",
        "description": "Introduce a new hypothesis into the proof context.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "clear": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "type", "clear"],
        },
    },
}

SIMPLE_GOAL = "⊢ ∀ n : ℕ, n + 0 = n"


class ParseCase(NamedTuple):
    goal: str
    top_k: int
    min_top_score: float


PARSEABLE_GOALS = [
    ParseCase(
        goal=(
            "M : Type u_3\ninst : CommMonoid M\ns : Multiset M\n"
            "inst_1 : DecidableEq M\na : M\n"
            "⊢ a ^ Multiset.count a s = (Multiset.filter (Eq a) s).prod"
        ),
        top_k=5,
        min_top_score=0.55,
    ),
    ParseCase(
        goal="⊢ ∀ a b : ℕ, a + b = b + a",
        top_k=5,
        min_top_score=0.45,
    ),
]


# -----------------------------------------------------------------------------
# Small Process/Path Helpers
# -----------------------------------------------------------------------------

def die(message: str) -> None:
    raise SystemExit(message)


def quantized_model_path(source_model: Path, quant: str) -> Path:
    name = source_model.name
    suffix = f".{quant}.gguf"
    if name.lower().endswith(".f16.gguf"):
        return source_model.with_name(name[:-len(".f16.gguf")] + suffix)
    if name.lower().endswith(".gguf"):
        return source_model.with_name(name[:-len(".gguf")] + suffix)
    return source_model.with_name(name + suffix)


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


def find_or_build_quantizer() -> Path:
    quantizer = find_exe("llama-quantize")
    if quantizer:
        return quantizer

    print("llama-quantize not found; building target llama-quantize...", flush=True)
    build_target("llama-quantize")
    quantizer = find_exe("llama-quantize")
    if not quantizer:
        die("Failed to locate llama-quantize after building it")
    return quantizer


def find_joint_server(server_bin: Path | None) -> Path:
    if server_bin:
        return server_bin
    joint_server = find_exe("joint-server")
    if joint_server:
        return joint_server

    print("joint-server not found; building target joint-server...", flush=True)
    build_target("joint-server")
    joint_server = find_exe("joint-server")
    if not joint_server:
        die("Failed to locate joint-server after building it")
    return joint_server


# -----------------------------------------------------------------------------
# Step 1: Quantize The Model
# -----------------------------------------------------------------------------

def ensure_quantized_model(args: argparse.Namespace) -> Path:
    output_model = args.quantized_model or quantized_model_path(args.model, args.quant)

    if output_model.exists() and not args.force_quantize:
        print(f"Using existing quantized model: {output_model}", flush=True)
        return output_model

    if not args.model.exists():
        die(f"Source model does not exist: {args.model}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    quantizer = find_or_build_quantizer()
    command = [str(quantizer), str(args.model), str(output_model), args.quant]
    print("Quantizing model:", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=REPO_ROOT)
    return output_model


# -----------------------------------------------------------------------------
# Step 2: Start joint-server
# -----------------------------------------------------------------------------

def joint_server_command(args: argparse.Namespace, model: Path) -> list[str]:
    server_bin = find_joint_server(args.server_bin)
    command = [
        str(server_bin),
        "--host", args.host,
        "--port", str(args.port),
        "--model", str(model),
        "--joint",
        "--index-vecs", str(args.index_vecs),
        "--index-strings", str(args.index_strings),
        "--ctx-size", str(args.ctx_size),
        "--n-predict", str(args.n_predict),
        "--jinja",
    ]
    command.extend(args.server_arg or [])
    return command


class ManagedJointServer:
    def __init__(self, args: argparse.Namespace, model: Path):
        self.args = args
        self.model = model
        self.process: subprocess.Popen | None = None
        self.log_file = None

    def __enter__(self) -> "ManagedJointServer":
        for required in (self.args.index_vecs, self.args.index_strings):
            if not required.exists():
                die(f"Required premise index file does not exist: {required}")

        env = os.environ.copy()
        if BUILD_BIN_DIR.exists():
            env["PATH"] = str(BUILD_BIN_DIR) + os.pathsep + env.get("PATH", "")

        log_path = self.args.server_log or (BUILD_DIR / "joint-server-test.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

        command = joint_server_command(self.args, self.model)
        print("Starting server:", " ".join(command), flush=True)

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
                die(f"joint-server exited early with code {self.process.returncode}; see {self.log_file.name}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        print(f"Server ready at {health_url}", flush=True)
                        return
            except Exception:
                time.sleep(0.5)

        die(f"Timed out waiting for server; see {self.log_file.name}")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.process and self.process.poll() is None:
            print("Stopping server", flush=True)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_file:
            self.log_file.close()


# -----------------------------------------------------------------------------
# Step 3: Query The Streaming API
# -----------------------------------------------------------------------------

def chat_request_body(goal: str, top_k: int) -> dict:
    return {
        "messages": [{"role": "user", "content": goal}],
        "tools": [HAVE_TOOL],
        "retrieval_topk": top_k,
        "stream": True,
    }


def extract_xml_tool_calls(text: str) -> tuple[list[dict], bool]:
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    tool_calls = []
    for match in matches:
        try:
            obj = json.loads(match.strip())
            arguments = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({"name": obj.get("name", ""), "arguments": arguments})
        except json.JSONDecodeError:
            pass
    return tool_calls, bool(tool_calls)


def finalize_streamed_tool_calls(parts: dict[int, dict[str, str]]) -> tuple[list[dict], bool]:
    tool_calls = []
    for index in sorted(parts):
        item = parts[index]
        arguments = item.get("arguments", "")
        try:
            arguments_obj = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments_obj = arguments
        tool_calls.append({"name": item.get("name", ""), "arguments": arguments_obj})
    return tool_calls, bool(tool_calls)


def iter_sse_data(response):
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\n\r")
        if line.startswith("data: "):
            yield line[6:]


def normalize_sse_data(data_items: list[str]) -> list[dict]:
    accumulated_text = ""
    tool_call_parts: dict[int, dict[str, str]] = {}
    events = []

    for data in data_items:
        if data == "[DONE]":
            tool_calls, parse_ok = finalize_streamed_tool_calls(tool_call_parts)
            if not tool_calls:
                tool_calls, parse_ok = extract_xml_tool_calls(accumulated_text)
            events.append({"type": "done", "tool_calls": tool_calls, "parse_ok": parse_ok})
            break

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        if payload.get("type") == "premises":
            events.append(payload)
            continue

        choices = payload.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        content = delta.get("content") or ""
        if content:
            accumulated_text += content
            events.append({"type": "token", "text": content})

        for tool_call in delta.get("tool_calls", []):
            index = int(tool_call.get("index", 0))
            item = tool_call_parts.setdefault(index, {"name": "", "arguments": ""})
            function = tool_call.get("function", {})
            if function.get("name"):
                item["name"] += function["name"]
            if function.get("arguments"):
                item["arguments"] += function["arguments"]
                events.append({"type": "token", "text": function["arguments"]})

    return events


def read_sse_events(response) -> list[dict]:
    return normalize_sse_data(list(iter_sse_data(response)))


def json_sse_payloads(data_items: list[str]) -> list[dict]:
    payloads = []
    for item in data_items:
        if item != "[DONE]":
            payloads.append(json.loads(item))
    return payloads


def post_raw_sse_data(goal: str, top_k: int = 5, timeout: int = 120) -> list[str]:
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(chat_request_body(goal, top_k)).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return list(iter_sse_data(response))


def post_infer(goal: str, top_k: int = 5, timeout: int = 120) -> list[dict]:
    return normalize_sse_data(post_raw_sse_data(goal, top_k=top_k, timeout=timeout))


def post_raw(body_bytes: bytes, content_type: str = "application/json") -> tuple[int, bytes]:
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=body_bytes,
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def post_json(path: str, body: dict, timeout: int = 120):
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# -----------------------------------------------------------------------------
# Optional: Print The Manual Commands This Script Automates
# -----------------------------------------------------------------------------

def print_usage_guide(args: argparse.Namespace) -> None:
    output_model = args.quantized_model or quantized_model_path(args.model, args.quant)
    quantize_exe = find_exe("llama-quantize") or (BUILD_BIN_DIR / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize"))
    joint_server_exe = args.server_bin or find_exe("joint-server") or (BUILD_BIN_DIR / ("joint-server.exe" if os.name == "nt" else "joint-server"))

    body = textwrap.indent(json.dumps(chat_request_body(SIMPLE_GOAL, 3), ensure_ascii=False, indent=2), "       ")
    guide = f"""Manual equivalent of this regression runner
==========================================

1. Build the binaries if needed:

   cmake -S {LLAMA_DIR} -B {BUILD_DIR} {' '.join(CMAKE_CONFIGURE_ARGS)}
   cmake --build {BUILD_DIR} --target llama-quantize
   cmake --build {BUILD_DIR} --target joint-server

2. Quantize the model if {output_model} does not exist:

   {quantize_exe} {args.model} {output_model} {args.quant}

3. Start joint-server:

   {joint_server_exe} --host {args.host} --port {args.port} --model {output_model} --joint ^
       --index-vecs {args.index_vecs} ^
       --index-strings {args.index_strings} ^
       --ctx-size {args.ctx_size} --n-predict {args.n_predict} --jinja

4. Query the streaming API:

   POST http://{args.host}:{args.port}/v1/chat/completions
   Content-Type: application/json

{body}

Expected stream shape:

   data: {{"type":"premises","premises":[...]}}
   data: {{"choices":[{{"delta":...}}]}}
   data: [DONE]

Use tools/premise-server/test_regression.py for standalone premise-server tests.
"""
    print(guide.strip())


# -----------------------------------------------------------------------------
# Regression Assertions
# -----------------------------------------------------------------------------

class TestProtocol(unittest.TestCase):
    def test_bad_json_returns_error(self):
        status, _ = post_raw(b"this is not json")
        self.assertGreaterEqual(status, 400)

    def test_missing_messages_field_returns_400(self):
        status, _ = post_raw(json.dumps({"retrieval_topk": 5}).encode())
        self.assertEqual(status, 400)

    def test_empty_messages_returns_400_or_noop(self):
        status, _ = post_raw(json.dumps({"messages": [], "tools": [HAVE_TOOL]}).encode())
        self.assertIn(status, (200, 400))

class TestResponseStructure(unittest.TestCase):
    TOP_K = 3

    @classmethod
    def setUpClass(cls):
        cls.events = post_infer(SIMPLE_GOAL, top_k=cls.TOP_K)

    def test_response_contains_all_mandatory_message_types(self):
        types = {event["type"] for event in self.events}
        self.assertIn("premises", types)
        self.assertIn("token", types)
        self.assertIn("done", types)

    def test_premises_before_first_token(self):
        types = [event["type"] for event in self.events]
        premise_index = types.index("premises")
        token_index = next((i for i, event_type in enumerate(types) if event_type == "token"), None)
        self.assertIsNotNone(token_index)
        self.assertLess(premise_index, token_index)

    def test_done_is_last_message(self):
        self.assertEqual(self.events[-1]["type"], "done")

    def test_premises_count_matches_top_k(self):
        premises = next(event for event in self.events if event["type"] == "premises")
        self.assertEqual(len(premises["premises"]), self.TOP_K)

    def test_each_premise_has_statement_and_score(self):
        premises = next(event for event in self.events if event["type"] == "premises")
        for premise in premises["premises"]:
            self.assertIsInstance(premise.get("statement"), str)
            self.assertTrue(premise["statement"])
            self.assertIsInstance(premise.get("score"), float)
            self.assertGreaterEqual(premise["score"], 0.0)
            self.assertLessEqual(premise["score"], 1.0)

    def test_token_messages_have_text_field(self):
        for event in self.events:
            if event["type"] == "token":
                self.assertIsInstance(event.get("text"), str)

    def test_done_has_parse_ok_bool(self):
        self.assertIsInstance(self.events[-1].get("parse_ok"), bool)

    def test_done_tool_calls_schema(self):
        for tool_call in self.events[-1].get("tool_calls", []):
            self.assertIsInstance(tool_call.get("name"), str)
            self.assertIsInstance(tool_call.get("arguments"), dict)

    def test_concatenated_tokens_nonempty(self):
        text = "".join(event["text"] for event in self.events if event["type"] == "token")
        self.assertGreater(len(text), 0)


class TestStreaming(unittest.TestCase):
    def test_first_sse_event_is_premises(self):
        raw_events = post_raw_sse_data(SIMPLE_GOAL)
        self.assertGreater(len(raw_events), 0)
        first_payload = json.loads(raw_events[0])
        self.assertEqual(first_payload.get("type"), "premises")

    def test_generation_uses_multiple_sse_events(self):
        raw_events = post_raw_sse_data(SIMPLE_GOAL)
        choice_events = [payload for payload in json_sse_payloads(raw_events) if "choices" in payload]
        self.assertGreaterEqual(len(choice_events), 2)
        self.assertEqual(raw_events[-1], "[DONE]")


class TestLeanPremiseProtocol(unittest.TestCase):
    MODULE = "CanonicalDrafter.TestProtocol.Module"
    TOKEN = "test-token-1"
    DECLARATIONS = [
        {"name": "CanonicalDrafter.TestProtocol.add_zero", "decl": "theorem add_zero (n : Nat) : n + 0 = n"},
        {"name": "CanonicalDrafter.TestProtocol.mul_one", "decl": "theorem mul_one (n : Nat) : n * 1 = n"},
    ]

    def test_version_cache_select_flow(self):
        before = post_json("/version", {"module": self.MODULE})
        self.assertIn(before, (None, self.TOKEN))

        cached = post_json("/cache", {
            "module": self.MODULE,
            "imports": [],
            "declarations": self.DECLARATIONS,
            "token": self.TOKEN,
        })
        self.assertEqual(cached, {"ok": True})

        after = post_json("/version", {"module": self.MODULE})
        self.assertEqual(after, self.TOKEN)

        suggestions = post_json("/select", {
            "imports": [self.MODULE],
            "declarations": [
                {"name": "CanonicalDrafter.TestProtocol.local_comm", "decl": "theorem local_comm (a b : Nat) : a + b = b + a"},
            ],
            "goal": "⊢ n + 0 = n",
            "k": 2,
        })
        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 2)
        self.assertGreater(len(suggestions), 0)
        for suggestion in suggestions:
            self.assertIsInstance(suggestion.get("name"), str)
            self.assertIsInstance(suggestion.get("score"), float)


class TestParseQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.multiset_events = post_infer(PARSEABLE_GOALS[0].goal, top_k=PARSEABLE_GOALS[0].top_k)
        cls.commutative_events = post_infer(PARSEABLE_GOALS[1].goal, top_k=PARSEABLE_GOALS[1].top_k)

    def assert_top_premise_score(self, events: list[dict], case: ParseCase):
        premises = next(event for event in events if event["type"] == "premises")
        score = premises["premises"][0]["score"]
        self.assertGreater(score, case.min_top_score)

    def test_multiset_goal_parse_ok(self):
        done = self.multiset_events[-1]
        raw = "".join(event["text"] for event in self.multiset_events if event["type"] == "token")
        self.assertTrue(done["parse_ok"], f"parse_ok=False; generated: {raw!r}")

    def test_multiset_goal_type_nonempty(self):
        done = self.multiset_events[-1]
        if done["parse_ok"]:
            arguments = done["tool_calls"][0]["arguments"]
            self.assertNotEqual(arguments.get("type", ""), "")

    def test_multiset_goal_top_premise_score(self):
        self.assert_top_premise_score(self.multiset_events, PARSEABLE_GOALS[0])

    def test_commutative_goal_parse_ok(self):
        done = self.commutative_events[-1]
        raw = "".join(event["text"] for event in self.commutative_events if event["type"] == "token")
        self.assertTrue(done["parse_ok"], f"parse_ok=False; generated: {raw!r}")

    def test_commutative_goal_top_premise_score(self):
        self.assert_top_premise_score(self.commutative_events, PARSEABLE_GOALS[1])


# -----------------------------------------------------------------------------
# Optional Dataset Smoke Queries
# -----------------------------------------------------------------------------

def load_dataset_goals(n: int) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(DRAFTS_ID, split="train").shuffle(seed=42)
    goals = []
    for row in dataset:
        goals.append({
            "goal": row["goal"],
            "name": row["name"],
            "type": row["type"],
            "clear": list(row["removals"]),
        })
        if len(goals) >= n:
            break
    return goals


def run_dataset_queries(n: int, top_k: int) -> bool:
    print("Loading goals from dataset...", flush=True)
    examples = load_dataset_goals(n)
    completed = 0

    for index, example in enumerate(examples, 1):
        print(f"\n{'=' * 72}")
        print(f"Dataset query {index}/{len(examples)}: {example['goal'][:80]}")
        print(f"gold: name={example['name']!r} type={example['type'][:60]!r}")
        print("=" * 72)

        events = post_infer(example["goal"], top_k=top_k)
        premises = next((event for event in events if event["type"] == "premises"), None)
        done = events[-1] if events else None

        if premises:
            print(f"premises: {len(premises['premises'])}")
            for premise_index, premise in enumerate(premises["premises"], 1):
                snippet = premise["statement"][:80].replace("\n", " ")
                print(f"  [{premise_index}] {premise['score']:.3f} {snippet}")

        if done and done.get("parse_ok"):
            arguments = done["tool_calls"][0].get("arguments", {})
            print(f"name : {arguments.get('name', '')!r} (gold: {example['name']!r})")
            print(f"type : {str(arguments.get('type', ''))[:80]!r}")
            print(f"clear: {arguments.get('clear', '')}")

        if premises is not None and done is not None:
            completed += 1
            print("[PASS]" if done.get("parse_ok") else "[PARTIAL parse_ok=False]")
        else:
            print("[FAIL] missing premises or done", file=sys.stderr)

    print(f"\nDataset query results: {completed}/{len(examples)} completed")
    return completed == len(examples)


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Quantize a joint model, start joint-server, and run streaming premise regression tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Common flows:
          python tools/joint-server/test_regression.py --print-guide
          python tools/joint-server/test_regression.py --quant Q8_0
          python tools/joint-server/test_regression.py --quant Q4_K_M --force-quantize
          python tools/joint-server/test_regression.py --url http://127.0.0.1:8080
        """),
    )
    parser.add_argument("--print-guide", action="store_true", help="Print manual quantize/start/query commands and exit")
    parser.add_argument("--url", help="Use an already-running server instead of starting one")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Source F16 GGUF model")
    parser.add_argument("--quantized-model", type=Path, help="Output/override quantized GGUF path")
    parser.add_argument("--quant", default=DEFAULT_QUANT, help="llama-quantize type, e.g. Q8_0 or Q4_K_M")
    parser.add_argument("--force-quantize", action="store_true", help="Recreate the quantized model even if it exists")
    parser.add_argument("--index-vecs", type=Path, default=DEFAULT_INDEX_VECS)
    parser.add_argument("--index-strings", type=Path, default=DEFAULT_INDEX_STRINGS)
    parser.add_argument("--server-bin", type=Path, help="Path to joint-server executable")
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--server-arg", action="append", help="Extra argument passed to joint-server; repeat as needed")
    parser.add_argument("--dataset-n", type=int, default=0, help="Run dataset smoke queries after regression tests")
    parser.add_argument("--dataset-top-k", type=int, default=5)
    return parser.parse_known_args()


def run_regression_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main() -> int:
    global BASE_URL

    args, _unittest_args = parse_args()

    if args.print_guide:
        print_usage_guide(args)
        return 0

    if args.url:
        BASE_URL = args.url.rstrip("/")
        ok = run_regression_tests()
        if ok and args.dataset_n > 0:
            ok = run_dataset_queries(args.dataset_n, args.dataset_top_k)
        return 0 if ok else 1

    model = ensure_quantized_model(args)
    BASE_URL = f"http://{args.host}:{args.port}"
    with ManagedJointServer(args, model):
        ok = run_regression_tests()
        if ok and args.dataset_n > 0:
            ok = run_dataset_queries(args.dataset_n, args.dataset_top_k)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
