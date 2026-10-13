"""Run interface-matched decision-model baselines and interventions."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from openrouter_api import utc_now
from study_paths import repo_path, relative_path
from subjective_baseline_runner import DATASETS


ROOT = Path(__file__).resolve().parents[1]
BASELINE_RUNNER = ROOT / "scripts" / "subjective_baseline_runner.py"
INTERVENTION_RUNNER = ROOT / "scripts" / "run_subjective_interventions.py"
OUTPUT_ROOT = ROOT / "experiment_results" / "decision_model_comparisons"
MODEL_ROUTES = {
    "clef_flash": "cloudflare/clef-flash",
    "pplx_decider_v1_1_27b": "perplexity/pplx-decider-v1.1-27b",
}
SCHEMA_SMOKE_DATASETS = (
    "ukp_humor_pairwise.jsonl",
    "humicroedit_scoring.jsonl",
)
JSONL_LOCK = Lock()


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(value, ensure_ascii=False) + "\n")
            target.flush()
            os.fsync(target.fileno())


def command_mode(args):
    return "--live" if args.live else "--preview"


def baseline_command(args, route, output_root, datasets, limit=None):
    command = [
        sys.executable, relative_path(BASELINE_RUNNER), command_mode(args),
        "--model", route,
        "--output-root", relative_path(output_root),
        "--datasets", ",".join(datasets),
        "--batch-size", str(args.batch_size),
        "--max-attempts", str(args.max_attempts),
        "--timeout", str(args.timeout),
    ]
    if limit is not None:
        command += ["--limit-per-dataset", str(limit)]
    return command


def intervention_command(args, route, output_root):
    command = [
        sys.executable, relative_path(INTERVENTION_RUNNER), command_mode(args),
        "--model", route,
        "--output-root", relative_path(output_root),
        "--batch-size", str(args.batch_size),
        "--max-attempts", str(args.max_attempts),
        "--timeout", str(args.timeout),
        "--wave-resume-attempts", str(args.supervisor_attempts),
        "--resume-delay", str(args.resume_delay),
    ]
    if args.limit_per_dataset is not None:
        command += ["--limit-per-dataset", str(args.limit_per_dataset)]
    return command


def run_process(command, cwd, stdout_path, stderr_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        return subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr).returncode


def run_process_until_success(command, cwd, stdout_path, stderr_path, supervisor_path,
                              attempts=0, resume_delay=10, label="worker"):
    """Restart temporary worker failures; checkpoints make every restart incremental."""
    attempt = 0
    while True:
        attempt += 1
        append_jsonl(supervisor_path, {
            "event": "process_start", "timestamp": utc_now(), "label": label,
            "attempt": attempt, "command": [str(value) for value in command],
        })
        try:
            code = run_process(command, cwd, stdout_path, stderr_path)
        except OSError as exc:
            code = 75
            append_jsonl(supervisor_path, {
                "event": "process_start_error", "timestamp": utc_now(), "label": label,
                "attempt": attempt, "error": repr(exc),
            })
        append_jsonl(supervisor_path, {
            "event": "process_end", "timestamp": utc_now(), "label": label,
            "attempt": attempt, "exit_code": code,
        })
        if code == 0:
            return 0
        # Exit 75 means the worker safely checkpointed temporary/rejected items.
        # Exit 2 is argparse/configuration failure; other codes indicate a real crash.
        if code != 75 or (attempts and attempt >= attempts):
            return code
        delay = min(resume_delay * (2 ** min(attempt - 1, 5)), 300)
        append_jsonl(supervisor_path, {
            "event": "process_resume", "timestamp": utc_now(), "label": label,
            "completed_attempt": attempt, "next_attempt": attempt + 1,
            "delay_seconds": delay,
        })
        time.sleep(delay)


def run_schema_smoke(args, alias, route, model_root):
    output = model_root / "schema_smoke"
    command = baseline_command(args, route, output, SCHEMA_SMOKE_DATASETS, limit=1)
    return run_process_until_success(
        command, ROOT, output / "launcher.stdout.log", output / "launcher.stderr.log",
        output / "supervisor.jsonl", args.supervisor_attempts, args.resume_delay,
        f"{alias}:schema-smoke")


def run_baseline(args, alias, route, model_root):
    output = model_root / "baseline"
    log_root = output / "launcher_logs"
    jobs = []
    for dataset in DATASETS:
        command = baseline_command(args, route, output, (dataset,), args.limit_per_dataset)
        stem = Path(dataset).stem
        jobs.append((command, log_root / f"{stem}.stdout.log", log_root / f"{stem}.stderr.log",
                     f"{alias}:baseline:{stem}"))
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        codes = list(pool.map(
            lambda job: run_process_until_success(
                job[0], ROOT, job[1], job[2], log_root / "supervisor.jsonl",
                args.supervisor_attempts, args.resume_delay, job[3]), jobs))
    return 0 if all(code == 0 for code in codes) else 1


def run_interventions(args, alias, route, model_root):
    output = model_root / "interventions"
    command = intervention_command(args, route, output)
    return run_process_until_success(
        command, ROOT, output / "launcher.stdout.log", output / "launcher.stderr.log",
        output / "supervisor.jsonl", args.supervisor_attempts, args.resume_delay,
        f"{alias}:interventions")


def run_model(args, alias):
    route = MODEL_ROUTES[alias]
    model_root = args.output_root / alias
    append_jsonl(args.output_root / "comparison_runs.jsonl", {
        "event": "model_start", "timestamp": utc_now(), "alias": alias,
        "route": route, "phase": args.phase,
    })
    phases = {
        "schema-smoke": (run_schema_smoke,),
        "baseline": (run_baseline,),
        "interventions": (run_interventions,),
        "all": (run_baseline, run_interventions),
    }[args.phase]
    code = 0
    for phase in phases:
        code = phase(args, alias, route, model_root)
        if code:
            break
    append_jsonl(args.output_root / "comparison_runs.jsonl", {
        "event": "model_end", "timestamp": utc_now(), "alias": alias,
        "route": route, "phase": args.phase, "exit_code": code,
    })
    return code


def parse_models(value):
    if value == "all":
        return tuple(MODEL_ROUTES)
    aliases = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [alias for alias in aliases if alias not in MODEL_ROUTES]
    if not aliases or unknown:
        raise argparse.ArgumentTypeError(f"Unknown model alias(es): {', '.join(unknown) or value}")
    return aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--preview", action="store_true")
    parser.add_argument("--models", type=parse_models, default=tuple(MODEL_ROUTES),
                        help="all or comma-separated aliases: " + ", ".join(MODEL_ROUTES))
    parser.add_argument("--phase", choices=("schema-smoke", "baseline", "interventions", "all"),
                        default="schema-smoke")
    parser.add_argument("--parallel-models", action="store_true")
    parser.add_argument("--output-root", type=repo_path, default=relative_path(OUTPUT_ROOT))
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=6,
                        help="HTTP attempts per item within each worker attempt")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--supervisor-attempts", type=int, default=0,
                        help="Worker/wave attempts; 0 keeps retrying temporary failures")
    parser.add_argument("--resume-delay", type=float, default=10,
                        help="Initial seconds between resumptions (exponential, capped at 300)")
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    if not 1 <= args.batch_size <= 64:
        parser.error("--batch-size must be between 1 and 64")
    if args.max_attempts < 1 or args.timeout <= 0:
        parser.error("max attempts and timeout must be positive")
    if args.supervisor_attempts < 0 or args.resume_delay < 0:
        parser.error("supervisor attempts and resume delay cannot be negative")

    if args.parallel_models:
        with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
            codes = list(pool.map(lambda alias: run_model(args, alias), args.models))
    else:
        codes = [run_model(args, alias) for alias in args.models]
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
