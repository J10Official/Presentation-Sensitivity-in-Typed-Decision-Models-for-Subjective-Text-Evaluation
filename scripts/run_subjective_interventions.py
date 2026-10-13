"""Run all 50 intervention conditions in sequential waves with per-dataset workers."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from openrouter_api import utc_now
from study_paths import repo_path, relative_path
from subjective_intervention_runner import (
    MODEL,
    OUTPUT_ROOT,
    PAIRWISE_CONDITIONS,
    PAIRWISE_FILES,
    SCORING_CONDITIONS,
    SCORING_FILES,
)


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "subjective_intervention_runner.py"
WAVES = (
    ("pair_exchange_labels", None),
    ("pair_exchange_positions", None),
    ("pair_exchange_both", None),
    (None, "score_identifier_a"),
    (None, "score_identifier_b"),
    ("pair_numeric_first_1", "score_numeric_identifier_1"),
    ("pair_numeric_first_2", "score_numeric_identifier_2"),
    ("pair_ai_human_anonymous", "score_ai_human_anonymous"),
    ("pair_ai_human_first_ai", "score_ai_human_ai"),
    ("pair_ai_human_first_human", "score_ai_human_human"),
    ("pair_social_none", "score_social_none"),
    ("pair_social_first_high", "score_social_high"),
    ("pair_social_second_high", "score_social_low"),
    ("pair_prior_judge_none", "score_prior_judge_none"),
    ("pair_prior_judge_first_selected", "score_prior_judge_high"),
    ("pair_prior_judge_second_selected", "score_prior_judge_low"),
    ("pair_tie_last", None),
    ("pair_tie_first", None),
    ("pair_tie_middle", None),
    ("pair_rejection", None),
    (None, "score_anchor_reverse"),
    (None, "score_granularity_3"),
    (None, "score_granularity_7"),
    ("pair_injection_clean", "score_injection_clean"),
    ("pair_injection_first_inert", "score_injection_inert"),
    ("pair_injection_first_directive", "score_injection_directive"),
    ("pair_injection_second_inert", None),
    ("pair_injection_second_directive", None),
    ("pair_repeat_1", "score_repeat_1"),
    ("pair_repeat_2", "score_repeat_2"),
    ("pair_repeat_3", "score_repeat_3"),
    ("pair_repeat_4", "score_repeat_4"),
)

assert len(WAVES) == 32
assert sum(value is not None for wave in WAVES for value in wave) == 50
assert {value for wave in WAVES for value in wave if value} == set(PAIRWISE_CONDITIONS + SCORING_CONDITIONS)


def append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(value, ensure_ascii=False) + "\n")


def worker_command(args, condition, dataset):
    command = [
        sys.executable,
        relative_path(WORKER),
        "--live" if args.live else "--preview",
        "--condition", condition,
        "--dataset", dataset,
        "--input-root", relative_path(args.input_root),
        "--output-root", relative_path(args.output_root),
        "--model", args.model,
        "--batch-size", str(args.batch_size),
        "--max-attempts", str(args.max_attempts),
        "--timeout", str(args.timeout),
    ]
    if args.limit_per_dataset is not None:
        command += ["--limit-per-dataset", str(args.limit_per_dataset)]
    return command


def wave_jobs(wave):
    pairwise, scoring = wave
    jobs = []
    if pairwise:
        jobs.extend((pairwise, dataset) for dataset in PAIRWISE_FILES)
    if scoring:
        jobs.extend((scoring, dataset) for dataset in SCORING_FILES)
    return jobs


def run_wave(args, index, wave):
    log_root = args.output_root / "orchestrator_logs" / f"wave_{index:02d}"
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = wave_jobs(wave)
    processes = []
    append_jsonl(args.output_root / "orchestrator_runs.jsonl", {
        "event": "wave_start", "timestamp": utc_now(), "wave": index,
        "conditions": [value for value in wave if value],
        "workers": len(jobs), "batch_size_per_worker": args.batch_size,
        "maximum_parallel_requests": len(jobs) * args.batch_size,
        "limit_per_dataset": args.limit_per_dataset,
    })
    print(f"Wave {index}/{len(WAVES)}: {', '.join(value for value in wave if value)}; "
          f"workers={len(jobs)}; max_parallel={len(jobs) * args.batch_size}", flush=True)
    for condition, dataset in jobs:
        stem = Path(dataset).stem
        stdout_path = log_root / f"{condition}__{stem}.stdout.log"
        stderr_path = log_root / f"{condition}__{stem}.stderr.log"
        stdout = stdout_path.open("a", encoding="utf-8")
        stderr = stderr_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            worker_command(args, condition, dataset),
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
        )
        processes.append((condition, dataset, process, stdout, stderr, stdout_path, stderr_path))

    failures = []
    for condition, dataset, process, stdout, stderr, stdout_path, stderr_path in processes:
        code = process.wait()
        stdout.close()
        stderr.close()
        if code:
            failures.append({
                "condition": condition,
                "dataset": dataset,
                "exit_code": code,
                "stdout": relative_path(stdout_path),
                "stderr": relative_path(stderr_path),
            })
    append_jsonl(args.output_root / "orchestrator_runs.jsonl", {
        "event": "wave_end", "timestamp": utc_now(), "wave": index,
        "conditions": [value for value in wave if value], "failures": failures,
    })
    if failures:
        error = RuntimeError(f"Wave {index} failed: {json.dumps(failures)}")
        error.retryable = all(failure["exit_code"] == 75 for failure in failures)
        raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--preview", action="store_true")
    parser.add_argument("--input-root", type=repo_path,
                        default="prepared_data/subjective_study_1500")
    parser.add_argument("--output-root", type=repo_path, default=relative_path(OUTPUT_ROOT))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--wave-resume-attempts", type=int, default=0,
                        help="Wave attempts; 0 retries transient item failures until success")
    parser.add_argument("--resume-delay", type=float, default=10,
                        help="Initial seconds between transient wave resumptions")
    parser.add_argument("--start-wave", type=int, default=1)
    parser.add_argument("--end-wave", type=int, default=len(WAVES))
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    if not 1 <= args.batch_size <= 64:
        parser.error("--batch-size must be between 1 and 64")
    if args.wave_resume_attempts < 0 or args.resume_delay < 0:
        parser.error("wave resume attempts and delay cannot be negative")
    if not 1 <= args.start_wave <= args.end_wave <= len(WAVES):
        parser.error(f"waves must satisfy 1 <= start <= end <= {len(WAVES)}")

    append_jsonl(args.output_root / "orchestrator_runs.jsonl", {
        "event": "orchestrator_start", "timestamp": utc_now(),
        "mode": "live" if args.live else "preview",
        "condition_count": 50, "wave_count": len(WAVES),
        "start_wave": args.start_wave, "end_wave": args.end_wave,
        "batch_size_per_worker": args.batch_size,
        "limit_per_dataset": args.limit_per_dataset,
    })
    for index, wave in enumerate(WAVES, start=1):
        if not args.start_wave <= index <= args.end_wave:
            continue
        attempt = 0
        while True:
            attempt += 1
            try:
                run_wave(args, index, wave)
                break
            except RuntimeError as exc:
                retryable = getattr(exc, "retryable", False)
                exhausted = args.wave_resume_attempts and attempt >= args.wave_resume_attempts
                if not retryable or exhausted:
                    raise
                delay = min(args.resume_delay * (2 ** min(attempt - 1, 5)), 300)
                append_jsonl(args.output_root / "orchestrator_runs.jsonl", {
                    "event": "wave_resume", "timestamp": utc_now(), "wave": index,
                    "completed_attempt": attempt, "next_attempt": attempt + 1,
                    "delay_seconds": delay,
                })
                print(f"Wave {index} had rejected items; resuming completed outputs "
                      f"after {delay:.1f}s (next attempt {attempt + 1})", flush=True)
                time.sleep(delay)
    append_jsonl(args.output_root / "orchestrator_runs.jsonl", {
        "event": "orchestrator_end", "timestamp": utc_now(),
        "start_wave": args.start_wave, "end_wave": args.end_wave,
    })
    print(json.dumps({"event": "orchestrator_end", "conditions": 50,
                      "waves": len(WAVES)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
