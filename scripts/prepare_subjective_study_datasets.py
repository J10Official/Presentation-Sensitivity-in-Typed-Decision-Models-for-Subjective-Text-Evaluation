"""Download, select, balance, and audit the six subjective-study datasets.

The script uses only Python's standard library. It downloads missing files from
first-party/author-provided locations, selects 1,500 records per component, and
writes one JSONL plus one analysis JSON per component. Existing verified source
files are reused. No model calls or semantic eligibility judgments are made.

Run:
    python scripts/prepare_subjective_study_datasets.py
    python scripts/prepare_subjective_study_datasets.py --no-download

Only ``record["input"]`` is judge-visible. ``reference`` and ``metadata`` are
analysis fields and must never be included in a judge request.
"""

from __future__ import annotations

import argparse
from study_paths import repo_path, relative_path
import csv
import gzip
import hashlib
import io
import json
import math
import random
import re
import statistics
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261010
TARGET = 1500
HELPSTEER_WORD_LIMIT = 5000

UKP_REVISION = "5f2a432c4db5191467a3315220b78c926f66f6b3"
HELPSTEER_REVISION = "990b2711a36180dd19d9c94b8627844866f8982a"

UKP_RESULTS = ROOT / "datasets/complementary_audit/ukp/results.tsv"
UKP_TEXT_DIR = ROOT / "datasets/ukp-humor-official/data/pl-humor-full"
HUMICRO_ZIP = ROOT / "datasets/example_sources/semeval-2020-task-7-dataset.zip"
HELPSTEER_DIR = ROOT / "datasets/complementary_audit/helpsteer2"
OPENAI_DIR = ROOT / "datasets/large_subjective_candidates/dataset/comparisons"
SUMMEVAL_ANNOTATIONS = ROOT / "datasets/example_sources/model_annotations.aligned.jsonl"
SUMMEVAL_STORY_DIR = ROOT / "datasets/example_sources/summeval_recovered"

UKP_FILES = {
    UKP_RESULTS: "data/pl-humor-full/results.tsv",
    UKP_TEXT_DIR / "jokes_heterographic_puns.txt": "data/pl-humor-full/jokes_heterographic_puns.txt",
    UKP_TEXT_DIR / "jokes_homographic_puns.txt": "data/pl-humor-full/jokes_homographic_puns.txt",
    UKP_TEXT_DIR / "jokes_nonpuns.txt": "data/pl-humor-full/jokes_nonpuns.txt",
    UKP_TEXT_DIR / "nonjokes.txt": "data/pl-humor-full/nonjokes.txt",
}

HELPSTEER_FILES = [
    "train.jsonl.gz",
    "validation.jsonl.gz",
    "preference/preference.jsonl.gz",
    "disagreements/disagreements.jsonl.gz",
]

OPENAI_FILES = [
    "batch0_cnndm.json", "batch10.json", "batch11.json", "batch12.json",
    "batch13.json", "batch14.json", "batch15.json", "batch16.json",
    "batch17.json", "batch18.json", "batch19.json", "batch20.json",
    "batch22.json", "batch3.json", "batch4.json", "batch5.json",
    "batch6.json", "batch7.json", "batch8.json", "batch9.json",
    "cnndm0.json", "cnndm2.json", "edit_b2_eval_test.json",
]

SUMMEVAL_ARCHIVES = {
    "cnn": "0BwmD_VLjROrfTHk4NFg2SndKcjQ",
    "dailymail": "0BwmD_VLjROrfM1BxdkxVaTY2bWs",
}

KNOWN_SHA256 = {
    str(HUMICRO_ZIP.relative_to(ROOT)): "12a6cbf28c8b698ad80be42a65ac867b57e4c71662eedab607805e167ba791ab",
    str(SUMMEVAL_ANNOTATIONS.relative_to(ROOT)): "f0d4166e0cdeb439b387c4449634b067b7e9f721b9ef4c2b722f6b7550d3d6ab",
}

SOURCES = {
    "ukp_humor": {
        "release": f"https://github.com/UKPLab/acl2019-GPPL-humour-metaphor/tree/{UKP_REVISION}",
        "paper": "https://aclanthology.org/P19-1572/",
    },
    "humicroedit": {
        "release": "https://zenodo.org/records/3969509",
        "paper": "https://aclanthology.org/2020.semeval-1.98/",
    },
    "helpsteer2": {
        "release": f"https://huggingface.co/datasets/nvidia/HelpSteer2/tree/{HELPSTEER_REVISION}",
        "paper": "https://arxiv.org/abs/2410.01257",
    },
    "openai_summarize_from_feedback": {
        "release": "https://github.com/openai/summarize-from-feedback/tree/master/dataset",
        "paper": "https://arxiv.org/abs/2009.01325",
    },
    "summeval": {
        "release": "https://github.com/Yale-LILY/SummEval",
        "paper": "https://aclanthology.org/2021.tacl-1.24/",
        "article_archives": "https://cs.nyu.edu/~kcho/DMQA/",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: str) -> str:
    return sha256_bytes("\0".join(parts).encode("utf-8"))


def rng_for(name: str) -> random.Random:
    salt = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    return random.Random(SEED ^ salt)


def jsonl(path: Path):
    stream = (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
              else path.open("rt", encoding="utf-8"))
    with stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic_download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "jev-study-dataset-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as temporary:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                temporary.write(block)
            temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def ensure_download(url: str, path: Path, allow_download: bool, expected_sha256: str | None = None) -> None:
    if expected_sha256 is None:
        frozen = json.loads((ROOT / "provenance/dataset_manifest.json").read_text(encoding="utf-8"))
        relative = path.relative_to(ROOT).as_posix()
        expected_sha256 = frozen["source_manifest"]["files"][relative]
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing source file: {path}")
        print(json.dumps({"download": url, "to": str(path.relative_to(ROOT))}), flush=True)
        atomic_download(url, path)
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"Checksum mismatch for {path}: {actual}")


def download_sources(allow_download: bool) -> None:
    raw_base = f"https://raw.githubusercontent.com/UKPLab/acl2019-GPPL-humour-metaphor/{UKP_REVISION}/"
    for path, relative_url in UKP_FILES.items():
        ensure_download(raw_base + relative_url, path, allow_download)

    ensure_download(
        "https://zenodo.org/records/3969509/files/semeval-2020-task-7-dataset.zip?download=1",
        HUMICRO_ZIP,
        allow_download,
        KNOWN_SHA256[str(HUMICRO_ZIP.relative_to(ROOT))],
    )

    hf_base = f"https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/{HELPSTEER_REVISION}/"
    for relative in HELPSTEER_FILES:
        ensure_download(hf_base + relative + "?download=true", HELPSTEER_DIR / relative, allow_download)

    openai_base = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/comparisons/"
    for filename in OPENAI_FILES:
        ensure_download(openai_base + filename, OPENAI_DIR / filename, allow_download)

    ensure_download(
        "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl",
        SUMMEVAL_ANNOTATIONS,
        allow_download,
        KNOWN_SHA256[str(SUMMEVAL_ANNOTATIONS.relative_to(ROOT))],
    )
    ensure_summeval_stories(allow_download)
    verify_frozen_sources()


def verify_frozen_sources() -> None:
    """Reject changed sources before selection, including recovered article bytes."""
    frozen = json.loads((ROOT / "provenance/dataset_manifest.json").read_text(encoding="utf-8"))["source_manifest"]
    for relative, expected in frozen["files"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"Frozen source checksum mismatch: {relative}")
    stories = sorted((SUMMEVAL_STORY_DIR / "cnndm").glob("*/stories/*.story"))
    if (len(stories) != frozen["summeval_story_files"] or
            stable_id(*(sha256_file(path) for path in stories)) != frozen["summeval_story_set_sha256"]):
        raise ValueError("Frozen SummEval article-set checksum mismatch")


def parse_story(raw: str) -> str:
    return " ".join(line.strip() for line in raw.split("@highlight")[0].splitlines() if line.strip())


def ensure_summeval_stories(allow_download: bool) -> None:
    annotations = list(jsonl(SUMMEVAL_ANNOTATIONS))
    required = defaultdict(set)
    for row in annotations:
        parts = Path(row["filepath"]).parts
        if len(parts) != 4 or parts[0] != "cnndm" or parts[2] != "stories":
            raise ValueError(f"Unexpected SummEval filepath: {row['filepath']}")
        required[parts[1]].add(parts[3])

    for group, filenames in required.items():
        directory = SUMMEVAL_STORY_DIR / "cnndm" / group / "stories"
        directory.mkdir(parents=True, exist_ok=True)
        missing = {name for name in filenames if not (directory / name).exists()}
        if not missing:
            continue
        if not allow_download:
            raise FileNotFoundError(f"Missing {len(missing)} SummEval {group} stories")
        url = ("https://drive.usercontent.google.com/download?id="
               + SUMMEVAL_ARCHIVES[group] + "&export=download&confirm=t")
        print(json.dumps({"stream_archive": url, "needed_stories": len(missing)}), flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    basename = member.name.rsplit("/", 1)[-1]
                    if basename not in missing:
                        continue
                    if not member.isfile() or not re.fullmatch(r"[0-9a-f]{40}\.story", basename):
                        raise ValueError(f"Unsafe archive member: {member.name}")
                    extracted = archive.extractfile(member)
                    raw = extracted.read() if extracted else b""
                    if not parse_story(raw.decode("utf-8")):
                        raise ValueError(f"Empty story: {basename}")
                    (directory / basename).write_bytes(raw)
                    missing.remove(basename)
                    if not missing:
                        break
        if missing:
            raise FileNotFoundError(f"Archive did not contain {len(missing)} required {group} stories")


def population_sd(values: list[float]) -> float:
    return statistics.pstdev(values)


def median(values) -> float:
    return statistics.median(list(values))


def choose(items: list[dict], count: int, rng: random.Random) -> list[dict]:
    if len(items) < count:
        raise ValueError(f"Need {count} items but only {len(items)} are available")
    return rng.sample(items, count)


def proportional_quotas(counts: Counter, total: int) -> dict[str, int]:
    """Largest-remainder allocation preserving a stratum distribution."""
    population = sum(counts.values())
    if population < total:
        raise ValueError(f"Cannot select {total} records from a population of {population}")
    exact = {key: total * count / population for key, count in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), str(key)))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def add_score_sampling_metadata(items: list[dict], population: Counter, quotas: dict[str, int]) -> None:
    for item in items:
        cell = item["selection_cell"]
        item["metadata"].update({
            "sampling_stratum": cell,
            "eligible_stratum_count": population[cell],
            "selected_stratum_count": quotas[cell],
            "poststratification_weight": round(population[cell] / quotas[cell], 8),
        })


def choose_unique_groups_by_cell(
    items: list[dict], quotas: dict[str, int], cell_key: str, group_key: str, rng: random.Random
) -> list[dict]:
    pools = {cell: [x for x in items if x[cell_key] == cell] for cell in quotas}
    order = sorted(quotas, key=lambda cell: len({x[group_key] for x in pools[cell]}) / quotas[cell])
    for _ in range(500):
        selected, used = [], set()
        for cell in order:
            available = [x for x in pools[cell] if x[group_key] not in used]
            rng.shuffle(available)
            take, cell_groups = [], set(used)
            for item in available:
                group = item[group_key]
                if group in cell_groups:
                    continue
                take.append(item)
                cell_groups.add(group)
                if len(take) == quotas[cell]:
                    break
            if len(take) != quotas[cell]:
                break
            selected.extend(take)
            used = cell_groups
        else:
            return selected
    raise ValueError(f"Could not satisfy unique-{group_key} quotas after 500 attempts")


def closest_cell_quotas(capacities: dict[str, tuple[int, int]], cell_total: int, class_total: int) -> dict[str, int]:
    """Allocate class-1 counts near equal per cell, respecting both class capacities."""
    target = class_total / len(capacities)
    lower = {k: max(0, cell_total - second) for k, (_, second) in capacities.items()}
    upper = {k: min(cell_total, first) for k, (first, _) in capacities.items()}
    quotas = {k: min(upper[k], max(lower[k], math.floor(target))) for k in capacities}
    while sum(quotas.values()) < class_total:
        eligible = [k for k in quotas if quotas[k] < upper[k]]
        if not eligible:
            raise ValueError("Insufficient class-1 capacity")
        key = min(eligible, key=lambda k: (abs((quotas[k] + 1) - target), quotas[k], k))
        quotas[key] += 1
    while sum(quotas.values()) > class_total:
        eligible = [k for k in quotas if quotas[k] > lower[k]]
        if not eligible:
            raise ValueError("Insufficient class-2 capacity")
        key = min(eligible, key=lambda k: (abs((quotas[k] - 1) - target), -quotas[k], k))
        quotas[key] -= 1
    return quotas


def choose_with_group_cap_and_coverage(
    items: list[dict], quotas: dict[str, int], cell_key: str, group_key: str,
    cap: int, rng: random.Random
) -> list[dict]:
    if set(quotas) != {"clear", "disputed"}:
        raise ValueError("This selector expects clear/disputed quotas")
    groups = defaultdict(list)
    for item in items:
        groups[item[group_key]].append(item)
    target = sum(quotas.values())
    maximum = {group: min(cap, len(values)) for group, values in groups.items()}
    if sum(maximum.values()) < target:
        raise ValueError("Group cap leaves too few items")
    for _ in range(1000):
        group_totals = dict(maximum)
        remove = sum(group_totals.values()) - target
        while remove:
            available = [group for group, count in group_totals.items() if count > 1]
            if not available:
                break
            group = rng.choice(available)
            group_totals[group] -= 1
            remove -= 1
        if remove:
            continue

        clear_counts = {group: sum(x[cell_key] == "clear" for x in values)
                        for group, values in groups.items()}
        disputed_counts = {group: len(values) - clear_counts[group] for group, values in groups.items()}
        lower = {group: max(0, group_totals[group] - disputed_counts[group]) for group in groups}
        upper = {group: min(group_totals[group], clear_counts[group]) for group in groups}
        if not (sum(lower.values()) <= quotas["clear"] <= sum(upper.values())):
            continue
        chosen_clear = dict(lower)
        need_clear = quotas["clear"] - sum(chosen_clear.values())
        group_order = list(groups)
        rng.shuffle(group_order)
        while need_clear:
            available = [group for group in group_order if chosen_clear[group] < upper[group]]
            if not available:
                break
            group = rng.choice(available)
            chosen_clear[group] += 1
            need_clear -= 1
        if need_clear:
            continue

        selected = []
        for group, values in groups.items():
            clear = [x for x in values if x[cell_key] == "clear"]
            disputed = [x for x in values if x[cell_key] == "disputed"]
            selected += choose(clear, chosen_clear[group], rng)
            selected += choose(disputed, group_totals[group] - chosen_clear[group], rng)
        if (len(selected) == target and
                Counter(x[cell_key] for x in selected) == Counter(quotas)):
            return selected
    raise ValueError("Could not satisfy agreement, context coverage, and reuse cap")


def assign_pairwise_positions(items: list[dict], stratum_keys: tuple[str, ...], rng: random.Random) -> list[dict]:
    """Balance strict winners globally and within each stratum; randomize non-winners."""
    groups = defaultdict(list)
    ambiguous = []
    for item in items:
        if item["winner_index"] in (0, 1):
            groups[tuple(item[k] for k in stratum_keys)].append(item)
        else:
            ambiguous.append(item)

    odd_groups = [key for key, values in groups.items() if len(values) % 2]
    rng.shuffle(odd_groups)
    extra_a = (len(odd_groups) + 1) // 2
    extra_a_groups = set(odd_groups[:extra_a])
    output = []
    for key in sorted(groups, key=str):
        values = groups[key]
        rng.shuffle(values)
        a_count = len(values) // 2 + int(key in extra_a_groups)
        target_positions = [0] * a_count + [1] * (len(values) - a_count)
        rng.shuffle(target_positions)
        for item, target in zip(values, target_positions):
            display_order = [0, 1] if item["winner_index"] == target else [1, 0]
            output.append({**item, "display_order": display_order})
    for item in ambiguous:
        output.append({**item, "display_order": [0, 1] if rng.randrange(2) == 0 else [1, 0]})
    rng.shuffle(output)
    return output


def label_for(index: int) -> str:
    return "A" if index == 0 else "B"


def pairwise_record(item: dict, dataset: str, task: str, instruction: str) -> dict:
    order = item["display_order"]
    candidates = [item["candidates"][i] for i in order]
    underlying_ids = [item["candidate_ids"][i] for i in order]
    winner = item["winner_index"]
    winner_position = order.index(winner) if winner in (0, 1) else None
    raw_choices = [label_for(order.index(v)) if v in (0, 1) else v for v in item.get("raw_choices", [])]
    return {
        "id": item["item_id"],
        "dataset": dataset,
        "task": task,
        "input": {
            "instruction": instruction,
            "context": item.get("context", ""),
            "options": [{"label": label_for(i), "text": text} for i, text in enumerate(candidates)],
        },
        "reference": {
            "choice": label_for(winner_position) if winner_position is not None else item["reference_label"],
            "winner_position": winner_position + 1 if winner_position is not None else None,
            "individual_choices": raw_choices,
        },
        "metadata": {
            **item["metadata"],
            "agreement_group": item["agreement_group"],
            "annotation_depth": item["annotation_depth"],
            "candidate_ids": underlying_ids,
            "orientation_swapped": order == [1, 0],
        },
    }


def scoring_record(item: dict, dataset: str, task: str, instruction: str, scale: dict) -> dict:
    return {
        "id": item["item_id"],
        "dataset": dataset,
        "task": task,
        "input": {
            "instruction": instruction,
            "context": item["context"],
            "text": item["text"],
            "scale": scale,
        },
        "reference": {
            "score": item["score"],
            "individual_scores": item["ratings"],
            **item.get("reference_extra", {}),
        },
        "metadata": {
            **item["metadata"],
            "agreement_group": item["agreement_group"],
            "annotation_depth": item["annotation_depth"],
            "human_sd": item["human_sd"],
        },
    }


def load_ukp() -> list[dict]:
    category = {}
    mappings = [
        ("jokes_heterographic_puns.txt", "heterographic_pun"),
        ("jokes_homographic_puns.txt", "homographic_pun"),
        ("jokes_nonpuns.txt", "nonpun_joke"),
        ("nonjokes.txt", "nonhumorous"),
    ]
    for filename, name in mappings:
        for text in (UKP_TEXT_DIR / filename).read_text(encoding="utf-8").splitlines():
            category[text] = name

    lines = UKP_RESULTS.read_text(encoding="utf-8").splitlines()
    columns = lines[0].split("\t")
    grouped = defaultdict(list)
    for line in lines[1:]:
        row = dict(zip(columns, line.split("\t")))
        first, second = sorted((row["Text A"], row["Text B"]))
        answer = row["Answer"]
        choice = "X" if answer == "X" else int((row["Text A"] if answer == "A" else row["Text B"]) == second)
        grouped[(first, second)].append((row["Worker ID"], choice))

    items = []
    for (first, second), annotations in grouped.items():
        if len(annotations) != 5 or len({worker for worker, _ in annotations}) != 5:
            continue
        choices = [choice for _, choice in annotations]
        counts = Counter(choices)
        winner = 0 if counts[0] >= 3 else 1 if counts[1] >= 3 else None
        reference_label = "X" if counts["X"] >= 3 else "NO_MAJORITY"
        agreement = "clear" if max(counts[0], counts[1], counts["X"]) >= 4 else "disputed"
        subtype = ("4-5_AB" if max(counts[0], counts[1]) >= 4 else
                   "4-5_X" if counts["X"] >= 4 else "3_AB" if winner is not None else
                   "3_X" if counts["X"] == 3 else "no_majority")
        category_pair = "+".join(sorted((category[first], category[second])))
        items.append({
            "item_id": stable_id("ukp", first, second),
            "candidates": [first, second],
            "candidate_ids": [stable_id("ukp-text", first), stable_id("ukp-text", second)],
            "winner_index": winner,
            "reference_label": reference_label,
            "raw_choices": choices,
            "agreement_group": agreement,
            "annotation_depth": 5,
            "category_pair": category_pair,
            "metadata": {"category_pair": category_pair, "consensus_subtype": subtype,
                         "vote_counts": {str(k): v for k, v in counts.items()}},
        })
    return items


def prepare_ukp() -> list[dict]:
    items, rng = load_ukp(), rng_for("ukp")
    cells = sorted({x["category_pair"] for x in items})
    capacities = {
        cell: (sum(x["category_pair"] == cell and x["agreement_group"] == "clear" for x in items),
               sum(x["category_pair"] == cell and x["agreement_group"] == "disputed" for x in items))
        for cell in cells
    }
    clear_quotas = closest_cell_quotas(capacities, 150, 750)
    selected = []
    for cell in cells:
        clear = [x for x in items if x["category_pair"] == cell and x["agreement_group"] == "clear"]
        disputed = [x for x in items if x["category_pair"] == cell and x["agreement_group"] == "disputed"]
        selected += choose(clear, clear_quotas[cell], rng)
        selected += choose(disputed, 150 - clear_quotas[cell], rng)
    positioned = assign_pairwise_positions(selected, ("category_pair", "agreement_group"), rng)
    return [pairwise_record(x, "ukp_humor", "humor_pairwise",
                            "Which option contains the funnier text?") for x in positioned]


def load_humicroedit() -> list[dict]:
    items = []
    with zipfile.ZipFile(HUMICRO_ZIP) as archive:
        for split in ("train", "dev", "test"):
            member = f"semeval-2020-task-7-dataset/subtask-1/{split}.csv"
            rows = csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig")))
            for row in rows:
                ratings = [int(value) for value in row["grades"]]
                if len(ratings) != 5:
                    continue
                original = re.sub(r"<(.*?)/>", r"\1", row["original"])
                edited = re.sub(r"<.*?/>", lambda _: row["edit"], row["original"])
                score = statistics.mean(ratings)
                score_band = "0_to_lt1" if score < 1 else "1_to_lt2" if score < 2 else "2_to_3"
                items.append({
                    "item_id": stable_id("humicroedit", row["id"], edited),
                    "context": original,
                    "text": edited,
                    "ratings": ratings,
                    "score": score,
                    "human_sd": population_sd(ratings),
                    "annotation_depth": 5,
                    "score_band": score_band,
                    "original_id": stable_id("humicroedit-original", original),
                    "metadata": {"source_id": row["id"], "split": split, "score_band": score_band,
                                 "original_headline_id": stable_id("humicroedit-original", original)},
                })
    return items


def prepare_humicroedit() -> list[dict]:
    items, rng = load_humicroedit(), rng_for("humicroedit")
    cutoff = median(x["human_sd"] for x in items)
    for item in items:
        item["agreement_group"] = "clear" if item["human_sd"] <= cutoff else "disputed"
        item["selection_cell"] = item["score_band"] + "/" + item["agreement_group"]
        item["metadata"]["agreement_cutoff_sd"] = cutoff
    quotas = {}
    for agreement in ("clear", "disputed"):
        counts = Counter(x["score_band"] for x in items if x["agreement_group"] == agreement)
        quotas.update({band + "/" + agreement: count
                       for band, count in proportional_quotas(counts, 750).items()})
    population = Counter(x["selection_cell"] for x in items)
    add_score_sampling_metadata(items, population, quotas)
    selected = choose_unique_groups_by_cell(items, quotas, "selection_cell", "original_id", rng)
    scale = {"minimum": 0, "maximum": 3, "direction": "higher_is_funnier"}
    return [scoring_record(x, "humicroedit", "humor_scoring",
                           "How funny is the edited headline?", scale) for x in selected]


def helpsteer_individuals() -> dict[tuple[str, str], dict]:
    path = HELPSTEER_DIR / "disagreements/disagreements.jsonl.gz"
    return {(row["prompt"], row["response"]): row for row in jsonl(path)}


def load_helpsteer_pairs() -> list[dict]:
    items = []
    for row in jsonl(HELPSTEER_DIR / "preference/preference.jsonl.gz"):
        strengths = [value["strength"] for value in row["all_preferences_unprocessed"]]
        words = len(row["prompt"].split()) + len(row["response_1"].split()) + len(row["response_2"].split())
        if len(strengths) != 5 or row["response_1"] == row["response_2"] or words > HELPSTEER_WORD_LIMIT:
            continue
        response_2_votes = sum(value > 0 for value in strengths)
        raw_choices = [1 if value > 0 else 0 for value in strengths]
        majority_share = max(response_2_votes, 5 - response_2_votes) / 5
        published = row["preference_strength"]
        winner = 1 if published > 0 else 0 if published < 0 else None
        items.append({
            "item_id": stable_id("helpsteer-pair", row["prompt"], row["response_1"], row["response_2"]),
            "context": row["prompt"],
            "candidates": [row["response_1"], row["response_2"]],
            "candidate_ids": [stable_id("helpsteer-response", row["prompt"], row["response_1"]),
                              stable_id("helpsteer-response", row["prompt"], row["response_2"])],
            "winner_index": winner,
            "reference_label": "ABOUT_THE_SAME",
            "raw_choices": raw_choices,
            "agreement_group": "clear" if majority_share >= 0.8 else "disputed",
            "annotation_depth": 5,
            "metadata": {"split": row["split"], "majority_share": majority_share,
                         "published_preference_strength": published,
                         "source_strengths": strengths, "content_words": words},
        })
    return items


def prepare_helpsteer_pairs() -> list[dict]:
    items, rng = load_helpsteer_pairs(), rng_for("helpsteer-pairs")
    selected = choose([x for x in items if x["agreement_group"] == "clear"], 750, rng)
    selected += choose([x for x in items if x["agreement_group"] == "disputed"], 750, rng)
    positioned = assign_pairwise_positions(selected, ("agreement_group",), rng)
    return [pairwise_record(x, "helpsteer2_preference", "helpfulness_pairwise",
                            "Which response is more helpful overall for the user's request?") for x in positioned]


def load_helpsteer_scores() -> list[dict]:
    individuals = helpsteer_individuals()
    items = []
    for split in ("train", "validation"):
        for row in jsonl(HELPSTEER_DIR / f"{split}.jsonl.gz"):
            raw = individuals.get((row["prompt"], row["response"]))
            words = len(row["prompt"].split()) + len(row["response"].split())
            if not raw or len(raw["helpfulness"]) != 5 or words > HELPSTEER_WORD_LIMIT:
                continue
            ratings = raw["helpfulness"]
            items.append({
                "item_id": stable_id("helpsteer-score", row["prompt"], row["response"]),
                "context": row["prompt"],
                "text": row["response"],
                "ratings": ratings,
                "score": statistics.mean(ratings),
                "human_sd": population_sd(ratings),
                "annotation_depth": 5,
                "prompt_id": stable_id("helpsteer-prompt", row["prompt"]),
                "score_level": str(row["helpfulness"]),
                "metadata": {"split": split, "score_level": row["helpfulness"],
                             "published_helpfulness": row["helpfulness"], "content_words": words,
                             "prompt_id": stable_id("helpsteer-prompt", row["prompt"])},
            })
    return items


def prepare_helpsteer_scores() -> list[dict]:
    items, rng = load_helpsteer_scores(), rng_for("helpsteer-scores")
    cutoff = median(x["human_sd"] for x in items)
    for item in items:
        item["agreement_group"] = "clear" if item["human_sd"] <= cutoff else "disputed"
        item["selection_cell"] = item["score_level"] + "/" + item["agreement_group"]
        item["metadata"]["agreement_cutoff_sd"] = cutoff
    quotas = {}
    for agreement in ("clear", "disputed"):
        counts = Counter(x["score_level"] for x in items if x["agreement_group"] == agreement)
        quotas.update({level + "/" + agreement: count
                       for level, count in proportional_quotas(counts, 750).items()})
    population = Counter(x["selection_cell"] for x in items)
    add_score_sampling_metadata(items, population, quotas)
    selected = choose_unique_groups_by_cell(items, quotas, "selection_cell", "prompt_id", rng)
    scale = {"minimum": 0, "maximum": 4, "direction": "higher_is_more_helpful"}
    return [scoring_record(x, "helpsteer2", "helpfulness_scoring",
                           "How helpful is the response overall for the user's request?", scale)
            for x in selected]


def load_openai_pairs() -> list[dict]:
    groups, contexts, candidates = {}, {}, {}
    for path in sorted(OPENAI_DIR.glob("*.json")):
        for row in jsonl(path):
            info = row["info"]
            context = info.get("post") or info.get("article")
            if not info.get("post") or not context or row.get("choice") not in (0, 1):
                continue
            first, second = [summary["text"] for summary in row["summaries"]]
            if not first.strip() or not second.strip() or first == second:
                continue
            low, high = sorted((first, second))
            context_id = stable_id("openai-context", context)
            low_id, high_id = stable_id("openai-summary", low), stable_id("openai-summary", high)
            key = (context_id, low_id, high_id)
            contexts[context_id] = context
            candidates[low_id], candidates[high_id] = low, high
            group = groups.setdefault(key, {"workers": defaultdict(set), "titles": set()})
            chosen = (first, second)[row["choice"]]
            group["workers"][row["worker"]].add(int(chosen == high))
            group["titles"].add(info.get("title") or "")

    items = []
    for (context_id, low_id, high_id), group in groups.items():
        votes = [next(iter(values)) for values in group["workers"].values() if len(values) == 1]
        if len(votes) < 7:
            continue
        high_votes = sum(votes)
        winner = 1 if high_votes > len(votes) / 2 else 0 if high_votes < len(votes) / 2 else None
        share = max(high_votes, len(votes) - high_votes) / len(votes)
        title = sorted(group["titles"])[0]
        context = (title + "\n\n" if title else "") + contexts[context_id]
        items.append({
            "item_id": stable_id("openai-pair", context_id, low_id, high_id),
            "context": context,
            "candidates": [candidates[low_id], candidates[high_id]],
            "candidate_ids": [low_id, high_id],
            "winner_index": winner,
            "reference_label": "SPLIT",
            "raw_choices": votes,
            "agreement_group": "clear" if share >= 0.8 else "disputed",
            "annotation_depth": len(votes),
            "context_id": context_id,
            "metadata": {"context_id": context_id, "majority_share": share,
                         "title_variants": len(group["titles"])},
        })
    return items


def prepare_openai_pairs() -> list[dict]:
    items, rng = load_openai_pairs(), rng_for("openai-pairs")
    selected = choose_with_group_cap_and_coverage(
        items, {"clear": 750, "disputed": 750}, "agreement_group", "context_id", 3, rng
    )
    positioned = assign_pairwise_positions(selected, ("agreement_group",), rng)
    return [pairwise_record(x, "openai_summarize_from_feedback", "summary_pairwise",
                            "Which option is the better summary of the source text?") for x in positioned]


def load_summeval() -> list[dict]:
    rows = sorted(jsonl(SUMMEVAL_ANNOTATIONS), key=lambda row: (str(row["id"]), row["model_id"]))
    duplicate_models = defaultdict(list)
    canonical = {}
    for row in rows:
        key = (str(row["id"]), row["decoded"])
        duplicate_models[key].append(row["model_id"])
        canonical.setdefault(key, row)

    items = []
    for key, row in canonical.items():
        story_path = SUMMEVAL_STORY_DIR / row["filepath"]
        article = parse_story(story_path.read_text(encoding="utf-8"))
        expert = [annotation["relevance"] for annotation in row["expert_annotations"]]
        crowd = [annotation["relevance"] for annotation in row["turker_annotations"]]
        if len(expert) != 3 or len(crowd) != 5:
            continue
        score = (sum(expert) + sum(crowd)) / 8
        within_group_sd = (population_sd(expert) + population_sd(crowd)) / 2
        items.append({
            "item_id": stable_id("summeval", str(row["id"]), row["decoded"]),
            "context": article,
            "text": row["decoded"],
            "ratings": expert + crowd,
            "score": score,
            "human_sd": within_group_sd,
            "annotation_depth": 8,
            "article_id": str(row["id"]),
            "reference_extra": {"expert_scores": expert, "crowd_scores": crowd,
                                "expert_mean": statistics.mean(expert),
                                "crowd_mean": statistics.mean(crowd)},
            "metadata": {"article_id": str(row["id"]), "model_id": row["model_id"],
                         "duplicate_model_ids": duplicate_models[key],
                         "disagreement_measure": "mean(expert_population_sd,crowd_population_sd)"},
        })
    return items


def prepare_summeval() -> list[dict]:
    items, rng = load_summeval(), rng_for("summeval")
    cutoff = median(x["human_sd"] for x in items)
    for item in items:
        item["agreement_group"] = "clear" if item["human_sd"] <= cutoff else "disputed"
        item["metadata"]["agreement_cutoff_sd"] = cutoff
        score = item["score"]
        score_band = ("1_to_lt2" if score < 2 else "2_to_lt3" if score < 3 else
                      "3_to_lt4" if score < 4 else "4_to_5")
        item["metadata"]["score_band"] = score_band
        item["selection_cell"] = score_band + "/" + item["agreement_group"]

    by_article = defaultdict(list)
    for item in items:
        by_article[item["article_id"]].append(item)
    removable_articles = [article for article, values in by_article.items() if len(values) == 16]
    population = Counter(x["selection_cell"] for x in items)
    quotas = {}
    for agreement in ("clear", "disputed"):
        counts = Counter(x["metadata"]["score_band"] for x in items
                         if x["agreement_group"] == agreement)
        quotas.update({band + "/" + agreement: count
                       for band, count in proportional_quotas(counts, 750).items()})
    removals = {cell: population[cell] - quotas[cell] for cell in population}
    for _ in range(500):
        used, removed = set(), []
        cells = [cell for cell, count in removals.items() if count]
        rng.shuffle(cells)
        cells.sort(key=lambda cell: sum(
            any(x["selection_cell"] == cell for x in by_article[article])
            for article in removable_articles
        ) / removals[cell])
        for cell in cells:
            candidates = [x for x in items
                          if x["selection_cell"] == cell
                          and x["article_id"] in removable_articles
                          and x["article_id"] not in used]
            rng.shuffle(candidates)
            chosen = []
            for candidate in candidates:
                if candidate["article_id"] in {x["article_id"] for x in chosen}:
                    continue
                chosen.append(candidate)
                if len(chosen) == removals[cell]:
                    break
            if len(chosen) != removals[cell]:
                break
            removed.extend(chosen)
            used.update(x["article_id"] for x in chosen)
        else:
            removed_ids = {x["item_id"] for x in removed}
            selected = [x for x in items if x["item_id"] not in removed_ids]
            if len(selected) == TARGET:
                break
    else:
        raise ValueError("Could not balance SummEval disagreement while preserving article coverage")

    add_score_sampling_metadata(selected, population, quotas)
    scale = {"minimum": 1, "maximum": 5, "direction": "higher_is_more_relevant"}
    return [scoring_record(x, "summeval", "summary_relevance_scoring",
                           "How relevant is the summary to the source article?", scale)
            for x in selected]


def input_words(record: dict) -> int:
    value = record["input"]
    text = " ".join([value.get("instruction", ""), value.get("context", ""), value.get("text", "")]
                    + [option["text"] for option in value.get("options", [])])
    return len(text.split())


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def analyze(records: list[dict]) -> dict:
    words = [input_words(record) for record in records]
    metadata = [record["metadata"] for record in records]
    result = {
        "items": len(records),
        "unique_ids": len({record["id"] for record in records}),
        "agreement_groups": dict(sorted(Counter(x["agreement_group"] for x in metadata).items())),
        "annotation_depth": dict(sorted(Counter(str(x["annotation_depth"]) for x in metadata).items(), key=lambda x: int(x[0]))),
        "input_words": {"minimum": min(words), "median": round(percentile(words, 0.5), 2),
                        "p95": round(percentile(words, 0.95), 2), "maximum": max(words)},
    }
    if "options" in records[0]["input"]:
        strict = [record for record in records if record["reference"]["winner_position"] is not None]
        result["reference_choices"] = dict(sorted(Counter(record["reference"]["choice"] for record in records).items()))
        result["strict_winner_positions"] = dict(sorted(Counter(str(record["reference"]["winner_position"]) for record in strict).items()))
    else:
        result["score_bands"] = dict(sorted(Counter(str(x.get("score_band", x.get("score_level", ""))) for x in metadata).items()))
        result["score_band_by_agreement"] = dict(sorted(Counter(
            str(x.get("score_band", x.get("score_level", ""))) + "/" + x["agreement_group"]
            for x in metadata
        ).items()))
        if all("sampling_stratum" in x for x in metadata):
            result["eligible_score_agreement_strata"] = dict(sorted({
                x["sampling_stratum"]: x["eligible_stratum_count"] for x in metadata
            }.items()))
            result["selected_score_agreement_strata"] = dict(sorted(Counter(
                x["sampling_stratum"] for x in metadata
            ).items()))
    if any("category_pair" in x for x in metadata):
        result["category_pairs"] = dict(sorted(Counter(x["category_pair"] for x in metadata).items()))
        result["category_pair_by_agreement"] = dict(sorted(Counter(
            x["category_pair"] + "/" + x["agreement_group"] for x in metadata
        ).items()))
    if any("context_id" in x for x in metadata):
        counts = Counter(x["context_id"] for x in metadata)
        result["contexts"] = len(counts)
        result["maximum_pairs_per_context"] = max(counts.values())
    if any("article_id" in x for x in metadata):
        counts = Counter(x["article_id"] for x in metadata)
        result["articles"] = len(counts)
        result["summaries_per_article"] = dict(sorted(Counter(str(v) for v in counts.values()).items()))
    return result


def verify(name: str, records: list[dict], analysis: dict) -> None:
    if len(records) != TARGET or analysis["unique_ids"] != TARGET:
        raise AssertionError(f"{name}: expected {TARGET} unique records")
    if analysis["agreement_groups"] != {"clear": 750, "disputed": 750}:
        raise AssertionError(f"{name}: agreement is not 750/750")
    if "strict_winner_positions" in analysis:
        positions = analysis["strict_winner_positions"]
        if abs(positions.get("1", 0) - positions.get("2", 0)) > 1:
            raise AssertionError(f"{name}: winner positions are not balanced")
    expected_depth = {"ukp_humor_pairwise": {"5": 1500}, "humicroedit_scoring": {"5": 1500},
                      "helpsteer_pairwise": {"5": 1500}, "helpsteer_scoring": {"5": 1500},
                      "summeval_relevance_scoring": {"8": 1500}}
    if name in expected_depth and analysis["annotation_depth"] != expected_depth[name]:
        raise AssertionError(f"{name}: unexpected annotation depth")
    if name == "openai_summary_pairwise" and min(map(int, analysis["annotation_depth"])) < 7:
        raise AssertionError("OpenAI summary pair has fewer than seven annotations")
    if name == "ukp_humor_pairwise":
        if len(analysis["category_pairs"]) != 10 or set(analysis["category_pairs"].values()) != {150}:
            raise AssertionError("UKP category-pair cells are not all 150")
        if len(analysis["category_pair_by_agreement"]) != 20 or set(analysis["category_pair_by_agreement"].values()) != {75}:
            raise AssertionError("UKP category/agreement cells are not all 75")
    if "selected_score_agreement_strata" in analysis:
        if sum(analysis["selected_score_agreement_strata"].values()) != TARGET:
            raise AssertionError(f"{name}: invalid score/agreement stratum totals")
    if name.startswith("helpsteer") and analysis["input_words"]["maximum"] > HELPSTEER_WORD_LIMIT + 20:
        raise AssertionError("HelpSteer content exceeds the 5,000-word ceiling")
    if name == "openai_summary_pairwise" and analysis.get("maximum_pairs_per_context", 99) > 3:
        raise AssertionError("OpenAI context reuse exceeds three pairs")
    if name == "summeval_relevance_scoring" and analysis.get("articles") != 100:
        raise AssertionError("SummEval does not cover all 100 articles")
    group_key = {"humicroedit_scoring": "original_headline_id",
                 "helpsteer_scoring": "prompt_id"}.get(name)
    if group_key and len({record["metadata"][group_key] for record in records}) != TARGET:
        raise AssertionError(f"{name}: repeated {group_key}")


def write_jsonl(path: Path, records: list[dict]) -> None:
    # Match the frozen collection bytes on every OS, not only Windows.
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\r\n" for record in records), encoding="utf-8", newline="")


def source_manifest() -> dict:
    files = [*UKP_FILES.keys(), HUMICRO_ZIP,
             *[HELPSTEER_DIR / x for x in HELPSTEER_FILES],
             *[OPENAI_DIR / x for x in OPENAI_FILES], SUMMEVAL_ANNOTATIONS]
    stories = sorted((SUMMEVAL_STORY_DIR / "cnndm").glob("*/stories/*.story"))
    return {
        "sources": SOURCES,
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in files},
        "summeval_story_files": len(stories),
        "summeval_story_set_sha256": stable_id(*(sha256_file(path) for path in stories)),
    }


def markdown_report(analyses: dict[str, dict], output_hashes: dict[str, str]) -> str:
    rows = []
    for name, result in analyses.items():
        winner = result.get("strict_winner_positions", {})
        rows.append(
            f"| {name} | {result['items']} | {result['agreement_groups'].get('clear', 0)} / "
            f"{result['agreement_groups'].get('disputed', 0)} | "
            f"{winner.get('1', 'n.a.')} / {winner.get('2', 'n.a.')} | "
            f"{result['input_words']['median']:.1f} / {result['input_words']['p95']:.1f} / {result['input_words']['maximum']} |"
        )
    details = []
    for name, result in analyses.items():
        details.append(f"## {name}\n\n```json\n{json.dumps(result, indent=2)}\n```\n")
    source_lines = []
    for name, source in SOURCES.items():
        source_lines.append(f"- **{name}:** [release]({source['release']}), [paper]({source['paper']})")
    return f"""# Prepared subjective-study datasets

Generated deterministically with seed **{SEED}**. Every component has 1,500 unique records. Agreement groups are balanced 750/750. Pairwise strict human-reference winners are balanced between display positions. HelpSteer uses a 5,000-word content ceiling and no semantic eligibility screen.

Only each JSONL record's `input` object is judge-visible. The `reference` and `metadata` objects are analysis-only.

| Component | Items | Clear / disputed | Winner position 1 / 2 | Input words median / p95 / max |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Annotation depth is fixed at five for UKP, Humicroedit and both HelpSteer components; SummEval retains three expert plus five crowd relevance scores. OpenAI summary pairs retain all consistent votes from at least seven distinct workers, so their native depth varies.

Agreement is defined before sampling. UKP clear items have at least 4/5 agreement on A, B or X; HelpSteer clear items have at least 4/5 directional agreement; OpenAI clear items have at least 80% majority share. The remainder are disputed. Scoring clear/disputed groups use the eligible-pool median human-rating dispersion. Within each agreement group, every scoring dataset preserves its eligible score-band distribution proportionally rather than forcing equal score bands.

## First-party sources

{chr(10).join(source_lines)}

{chr(10).join(details)}
## Output hashes

```json
{json.dumps(output_hashes, indent=2)}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="Require all first-party source files to exist locally")
    parser.add_argument("--output-dir", type=repo_path,
                        default="prepared_data/subjective_study_1500")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    download_sources(not args.no_download)
    builders = {
        "ukp_humor_pairwise": prepare_ukp,
        "humicroedit_scoring": prepare_humicroedit,
        "helpsteer_pairwise": prepare_helpsteer_pairs,
        "helpsteer_scoring": prepare_helpsteer_scores,
        "openai_summary_pairwise": prepare_openai_pairs,
        "summeval_relevance_scoring": prepare_summeval,
    }
    analyses, output_hashes = {}, {}
    for name, builder in builders.items():
        records = builder()
        analysis = analyze(records)
        verify(name, records, analysis)
        data_path = output_dir / f"{name}.jsonl"
        analysis_path = output_dir / f"{name}.analysis.json"
        write_jsonl(data_path, records)
        analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8", newline="\r\n")
        analyses[name] = analysis
        output_hashes[data_path.name] = sha256_file(data_path)
        output_hashes[analysis_path.name] = sha256_file(analysis_path)
        print(json.dumps({"prepared": name, "items": len(records)}), flush=True)

    manifest = {
        "seed": SEED,
        "target_per_component": TARGET,
        "helpsteer_content_word_limit": HELPSTEER_WORD_LIMIT,
        "semantic_screening": False,
        "source_manifest": source_manifest(),
        "output_sha256": output_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\r\n")
    report_path = output_dir / "analysis.md"
    report_path.write_text(markdown_report(analyses, output_hashes), encoding="utf-8")
    print(json.dumps({"manifest": relative_path(manifest_path), "report": relative_path(report_path)}), flush=True)


if __name__ == "__main__":
    main()
