"""Benchmark cold premise caching with a deterministic synthetic corpus."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import test_regression as regression


DECL_TEMPLATES = (
    "theorem {name} (a b c : Nat) : a + (b + c) = (a + b) + c",
    "theorem {name} (a b : Nat) : a * (b + 1) = a * b + a",
    "theorem {name} (xs ys : List Nat) : (xs ++ ys).length = xs.length + ys.length",
    "theorem {name} (p q : Prop) : p /\\ q -> q /\\ p",
    "def {name} (n : Nat) : Nat := Nat.fold (fun x => x + {salt}) {salt} n",
    "theorem {name} (n : Nat) : n % ({salt} + 1) < {salt} + 1",
    "def {name} (f : Nat -> Nat) (n : Nat) : Nat := f (f (n + {salt}))",
    "theorem {name} (x : Int) : x + {salt} - {salt} = x",
)


def make_module(module_index: int, declarations_per_module: int, body_repeat: int) -> dict:
    module = f"ColdStart.Module{module_index:04d}"
    declarations = []
    for declaration_index in range(declarations_per_module):
        name = f"{module}.declaration{declaration_index:04d}"
        salt = 2 + (module_index * declarations_per_module + declaration_index) % 97
        template = DECL_TEMPLATES[declaration_index % len(DECL_TEMPLATES)]
        body = template.format(name=name, salt=salt)
        declarations.append({"name": name, "decl": " ".join([body] * body_repeat)})
    imports = [] if module_index == 0 else [f"ColdStart.Module{module_index - 1:04d}"]
    return {
        "module": module,
        "imports": imports,
        "declarations": declarations,
        "token": f"cold-start-v1-{module_index}-{declarations_per_module}",
    }


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))]


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def strip_docstring(declaration: str) -> str:
    if not declaration.startswith("/--"):
        return declaration
    end = declaration.find(" -/\n")
    return declaration[end + 4 :] if end >= 0 else declaration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=Path, help="Path to llama-server")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("PREMISE_MODEL", regression.DEFAULT_MODEL)),
        help="Premise embedding GGUF; defaults to PREMISE_MODEL",
    )
    parser.add_argument("--modules", type=int, default=256)
    parser.add_argument("--declarations-per-module", type=int, default=96)
    parser.add_argument("--body-repeat", type=int, default=1)
    parser.add_argument("--corpus", type=Path, help="NDJSON module payloads to replay instead of synthetic data")
    parser.add_argument("--corpus-modules", type=int, help="Evenly sample this many modules from --corpus")
    parser.add_argument("--prop-names", type=Path, help="Keep only declarations listed in this newline-delimited file")
    parser.add_argument("--strip-docstrings", action="store_true")
    parser.add_argument("--max-declaration-characters", type=int)
    parser.add_argument("--local-declarations", type=int, default=0)
    parser.add_argument("--select-repetitions", type=int, default=1)
    parser.add_argument("--corpus-stats-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--server-arg", action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.modules <= 0 or args.declarations_per_module <= 0 or args.body_repeat <= 0:
        raise SystemExit("--modules, --declarations-per-module, and --body-repeat must be positive")
    if args.local_declarations < 0 or args.select_repetitions <= 0:
        raise SystemExit("--local-declarations must be non-negative and --select-repetitions must be positive")
    if args.corpus:
        with args.corpus.open(encoding="utf-8") as corpus_file:
            modules = [json.loads(line) for line in corpus_file if line.strip()]
        if not modules:
            raise SystemExit("--corpus is empty")
        if args.corpus_modules:
            if args.corpus_modules <= 0:
                raise SystemExit("--corpus-modules must be positive")
            if args.corpus_modules < len(modules):
                step = len(modules) / args.corpus_modules
                modules = [modules[int(index * step)] for index in range(args.corpus_modules)]
    else:
        modules = [
            make_module(module_index, args.declarations_per_module, args.body_repeat)
            for module_index in range(args.modules)
        ]
    unfiltered_declaration_count = sum(len(module["declarations"]) for module in modules)
    if args.prop_names:
        with args.prop_names.open(encoding="utf-8") as prop_names_file:
            prop_names = {line.strip() for line in prop_names_file if line.strip()}
        for module in modules:
            module["declarations"] = [
                declaration for declaration in module["declarations"]
                if declaration["name"] in prop_names
            ]
    if args.strip_docstrings:
        for module in modules:
            for declaration in module["declarations"]:
                declaration["decl"] = strip_docstring(declaration["decl"])
    if args.max_declaration_characters:
        if args.max_declaration_characters <= 0:
            raise SystemExit("--max-declaration-characters must be positive")
        for module in modules:
            for declaration in module["declarations"]:
                declaration["decl"] = declaration["decl"][: args.max_declaration_characters]

    declarations = [
        declaration
        for module in modules
        for declaration in module["declarations"]
    ]
    if not declarations:
        raise SystemExit("no declarations remain after filtering")
    if args.local_declarations > len(declarations):
        raise SystemExit("--local-declarations exceeds the declarations in the selected corpus")
    local_declarations = [
        {"name": f"ColdStart.Local{index:04d}", "decl": declaration["decl"]}
        for index, declaration in enumerate(declarations[: args.local_declarations])
    ]
    if args.corpus_stats_only:
        declaration_characters = sum(len(declaration["decl"]) for declaration in declarations)
        print(json.dumps({
            "modules": len(modules),
            "declarations": len(declarations),
            "unfiltered_declarations": unfiltered_declaration_count,
            "declaration_characters": declaration_characters,
            "average_declaration_characters": round(declaration_characters / len(declarations), 2),
        }, indent=2))
        return 0

    with tempfile.TemporaryDirectory(prefix="premise-coldstart-") as temp:
        root = Path(temp)
        args.index_db = root / "premise.db"
        args.server_log = root / "server.log"
        regression.BASE_URL = f"http://{args.host}:{args.port}"

        started = time.perf_counter()
        latencies = []
        with regression.ManagedPremiseServer(args):
            startup_seconds = time.perf_counter() - started
            cache_started = time.perf_counter()
            for module_index, body in enumerate(modules):
                request_started = time.perf_counter()
                result = regression.post_json("/cache", body, timeout=args.request_timeout)
                if not regression.cache_ok(result):
                    raise SystemExit(f"/cache failed for {body['module']}: {result!r}")
                latencies.append(time.perf_counter() - request_started)
                if (module_index + 1) % 16 == 0 or module_index + 1 == len(modules):
                    print(f"cached {module_index + 1}/{len(modules)} modules", flush=True)
            cache_seconds = time.perf_counter() - cache_started

            select_latencies = []
            for _ in range(args.select_repetitions):
                select_started = time.perf_counter()
                suggestions = regression.post_json(
                    "/select",
                    {
                        "imports": [modules[-1]["module"]],
                        "declarations": local_declarations,
                        "goal": "|- a + (b + c) = (a + b) + c",
                        "k": 16,
                    },
                    timeout=args.request_timeout,
                )
                select_latencies.append(time.perf_counter() - select_started)

        total_seconds = time.perf_counter() - started
        declaration_count = sum(len(module["declarations"]) for module in modules)
        declaration_characters = sum(
            len(declaration["decl"])
            for module in modules
            for declaration in module["declarations"]
        )
        summary = {
            "modules": len(modules),
            "declarations": declaration_count,
            "unfiltered_declarations": unfiltered_declaration_count,
            "corpus": str(args.corpus) if args.corpus else "synthetic",
            "prop_names": str(args.prop_names) if args.prop_names else None,
            "strip_docstrings": args.strip_docstrings,
            "max_declaration_characters": args.max_declaration_characters,
            "declaration_characters": declaration_characters,
            "average_declaration_characters": round(declaration_characters / declaration_count, 2),
            "startup_seconds": round(startup_seconds, 3),
            "cache_seconds": round(cache_seconds, 3),
            "declarations_per_second": round(declaration_count / cache_seconds, 2),
            "module_latency_p50_seconds": round(statistics.median(latencies), 3),
            "module_latency_p95_seconds": round(percentile(latencies, 0.95), 3),
            "local_declarations": len(local_declarations),
            "select_repetitions": len(select_latencies),
            "select_first_seconds": round(select_latencies[0], 3),
            "select_repeat_p50_seconds": round(statistics.median(select_latencies[1:]), 3)
                if len(select_latencies) > 1 else None,
            "select_total_seconds": round(sum(select_latencies), 3),
            "suggestions": len(suggestions),
            "suggestion_names": [suggestion["name"] for suggestion in suggestions],
            "shutdown_and_total_seconds": round(total_seconds, 3),
            "database_bytes": file_size(args.index_db),
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
