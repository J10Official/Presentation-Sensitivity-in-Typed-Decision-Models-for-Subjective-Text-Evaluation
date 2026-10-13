"""Compile the single manuscript PDF with inline appendices; no downloads or API calls."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path
from study_paths import repo_path, relative_path

HERE = Path(__file__).resolve().parents[1] / "paper"


def run(command):
    result = subprocess.run(command, cwd=HERE, text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command[0]}\n{result.stdout[-6000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex", default="pdflatex", help="Executable name or installed path")
    parser.add_argument("--bibtex", default="bibtex", help="Executable name or installed path")
    parser.add_argument("--output-directory", type=lambda value: repo_path(value, HERE), default="output/current", help="Build directory relative to paper/; useful when an open PDF is locked")
    args = parser.parse_args()
    latex, bibtex = shutil.which(args.latex), shutil.which(args.bibtex)
    if not latex or not bibtex:
        parser.error("Install an ACL-compatible TeX distribution or pass --latex and --bibtex paths.")
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    # MiKTeX must not silently fetch packages; TeX Live does not use this flag.
    version = subprocess.run([latex, "--version"], capture_output=True, text=True).stdout
    options = ["--disable-installer"] if "miktex" in version.lower() else []
    command = [latex, *options, "-interaction=nonstopmode", "-halt-on-error",
               "-jobname=manuscript", "-output-directory=" + relative_path(output, HERE), "main.tex"]
    run(command)
    run([bibtex, relative_path(output / "manuscript", HERE)])
    for _ in range(3):
        run(command)
    log = (output / "manuscript.log").read_text(encoding="utf-8", errors="replace")
    bad = re.findall(r"^.*(?:Overfull|undefined|multiply defined|Fatal error|Label\(s\) may have changed|Line number reference failed).*$", log, re.M)
    if bad:
        raise RuntimeError("Inspect build warnings:\n" + "\n".join(bad))
    print(f"Built {relative_path(output / 'manuscript.pdf')}: one bibliography and internal appendix references.")


if __name__ == "__main__":
    main()
