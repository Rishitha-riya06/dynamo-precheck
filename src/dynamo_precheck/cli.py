"""
dynamo-precheck - run the locally-checkable part of the Dynamo PR pipeline
before you push.

Run it from anywhere inside a Dynamo task repo (the repo root or task/):

    dynamo-precheck                  everything: static checks, docker build +
                                     leak check, harbor oracle and nop runs
    dynamo-precheck --skip-harbor    static checks only (a few seconds)
    dynamo-precheck --install-hook   run the full check automatically before
                                     every `git push` in this repo

Exit code is 0 only if no check FAILed. WARN items are advisory and don't
block.

What it mirrors locally:
  - static checks: task layout, extraneous files, LF line endings, script
    syntax, task.toml metadata + diversity labels + 3600s agent timeout cap,
    instruction.md (no time-budget boilerplate, absolute paths, length),
    Dockerfile hygiene (approved digest-pinned base, apt hygiene, pinned pip
    deps, no COPY of solution/ or tests/), no verify-time installs in
    tests/*.sh, a docstring on every test function
  - validation: environment builds, the built image holds no copy of
    solution/ or tests/ files, oracle scores 1.0, nop scores < 1.0

What it can't cover: pass@2 / pass@5 agent trials, the rubric review, the
automated deep review and the duplicate check. Those need the hosted
pipeline, so a clean run here means "won't bounce for a locally-checkable
reason", not "will be accepted".
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from . import __version__

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
MARKERS = {PASS: "[ OK ]", FAIL: "[FAIL]", WARN: "[WARN]", SKIP: "[SKIP]"}

results = []  # list of (status, name, detail)

MAX_AGENT_TIMEOUT_SEC = 3600
INSTRUCTION_TOKEN_CAP = 1500
LARGE_FILE_BYTES = 100 * 1024 * 1024

REQUIRED_FILES = (
    "instruction.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
)

REQUIRED_METADATA = (
    "task_objective",
    "artifact_type",
    "expert_time_estimate_hours",
    "difficulty_explanation",
    "solution_explanation",
    "verification_explanation",
)

# Fallback copy of references/check-base-image.sh's pre-approved images, used only
# when the task repo doesn't ship that script next to task/.
APPROVED_BASE_DIGESTS = {
    "golang": "sha256:1a6d4452c65dea36aac2e2d606b01b4a029ec90cc1ae53890540ce6173ea77ac",
    "python": "sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb",
    "debian": "sha256:4724b8cc51e33e398f0e2e15e18d5ec2851ff0c2280647e1310bc1642182655d",
    "rust": "sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36",
    "node": "sha256:f3a68cf41a855d227d1b0ab832bed9749469ef38cf4f58182fb8c893bc462383",
    "ubuntu": "sha256:0d39fcc8335d6d74d5502f6df2d30119ff4790ebbb60b364818d5112d9e3e932",
    "eclipse-temurin": "sha256:25d1276565738d3c805e632a4542c3a7598866ef967f4def6544c15de3a74b14",
    "ruby": "sha256:e76733e94b3a5893e4a141024ef3a583dc10781dc24becebf74f9c9f9a33e3df",
    "maven": "sha256:3a4ab3276a087bf276f79cae96b1af04f53731bec53fb2e651aca79e4b10211e",
    "gcc": "sha256:930f2ebe239275fa67226654cb79273ea34eee672ae61c8a39f689c37fb7ac5c",
}

TASK_ROOT_ENTRIES = {
    "instruction.md", "task.toml", "environment", "solution", "tests",
    ".dockerignore", ".gitattributes", ".gitignore",
}

JUNK_RE = re.compile(
    r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.venv|node_modules|\.ipynb_checkpoints)/"
    r"|^jobs/"
    r"|\.(pyc|pyo|bak|orig|rej|swp|swo)$"
    r"|(^|/)(\.DS_Store|Thumbs\.db|desktop\.ini)$"
    r"|~$",
    re.IGNORECASE,
)

STRICT_LF_SUFFIXES = {
    ".sh", ".py", ".toml", ".md", ".yaml", ".yml", ".service", ".cfg", ".conf", ".ini", ".json",
}

SKIP_WALK_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules", "jobs"}

# The rubric's instruction_concision criterion fails this TB3 boilerplate; the time
# budget belongs only in task.toml's [agent].timeout_sec.
BOILERPLATE_RE = re.compile(
    r"you have\s+\d+(?:\.\d+)?\s+seconds|do not cheat by using online solutions", re.IGNORECASE
)

PLACEHOLDER_RE = re.compile(
    r"\b(TODO|TBD|FIXME)\b|lorem ipsum|<[^<>\n]*\b(fill|your|todo)\b[^<>\n]*>", re.IGNORECASE
)

REL_PATH_RE = re.compile(
    r"^(?:\.{1,2}/)?[\w.-]+(?:/[\w.-]+)+/?$"
    r"|^\.?[\w-]+\.(?:csv|tsv|json|jsonl|txt|py|sh|md|ya?ml|toml|log|conf|cfg|ini|xml|html|sql|db|sqlite|parquet|service)$"
)

INSTALL_RE = re.compile(
    r"\b(?:pip[\d.]*|uv\s+pip|conda|mamba|micromamba)\s+install\b"
    r"|\buv\s+(?:add|sync|tool\s+install)\b|\buvx\b|\buv\s+run\b.*\s--with\b"
    r"|\bapt(?:-get)?\s+(?:-\S+\s+)*install\b|\bapk\s+add\b|\b(?:yum|dnf|zypper)\s+install\b"
    r"|\b(?:npm|pnpm|yarn)\s+(?:install|add|i)\b|\bnpx\b"
    r"|\bgem\s+install\b|\bcargo\s+install\b|\bgo\s+(?:install|get)\b"
)
FETCH_RE = re.compile(r"\b(?:curl|wget)\b.*https?://")
LOCAL_URL_RE = re.compile(r"https?://(?:localhost|127\.|0\.0\.0\.0|\[::1\])")

PIP_VALUE_FLAGS = {
    "-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-i", "--index-url",
    "--extra-index-url", "-f", "--find-links", "-t", "--target", "--prefix", "--root",
    "--platform", "--python-version", "--implementation", "--abi", "--src", "--trusted-host",
    "--cache-dir", "--python", "-p", "--progress-bar",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def record(status, name, detail=""):
    results.append((status, name, detail))
    line = f"{MARKERS[status]} {name}"
    if detail:
        line += " - " + str(detail).replace("\n", "\n       ")
    print(line, flush=True)


def short_list(items, limit=8):
    items = list(items)
    text = ", ".join(items[:limit])
    if len(items) > limit:
        text += f", ... (+{len(items) - limit} more)"
    return text


def tail(output, lines=25):
    return "\n".join(output.rstrip().splitlines()[-lines:])


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def run(cmd, cwd=None, verbose=False):
    """Run a command, return (returncode, stdout+stderr combined)."""
    if verbose:
        print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    if verbose:
        print(proc.stdout, flush=True)
    return proc.returncode, proc.stdout


def git_out(cwd, *args):
    """stdout of a git command run in cwd, or None if git is missing or refuses."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def git_ls(task_dir, *args):
    out = git_out(task_dir, "ls-files", "-z", *args, "--", ".")
    return None if out is None else [p for p in out.split("\0") if p]


def iter_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_WALK_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def list_task_files(task_dir):
    """Files git would ship (tracked + untracked-not-ignored); plain walk outside git."""
    rels = git_ls(task_dir, "--cached", "--others", "--exclude-standard")
    if rels is None:
        return list(iter_files(task_dir))
    return [task_dir / r for r in dict.fromkeys(rels) if (task_dir / r).is_file()]


def rel(path, task_dir):
    return path.relative_to(task_dir).as_posix()


def find_task_dir(start):
    start = start.resolve()
    for d in (start, *start.parents):
        if (d / "task.toml").is_file():
            return d
        if (d / "task" / "task.toml").is_file():
            return d / "task"
    return None


def find_bash():
    if os.name != "nt":
        return shutil.which("bash")
    # On Windows, System32\bash.exe is the WSL launcher; use Git for Windows' bash.
    git = shutil.which("git")
    if not git:
        return None
    base = Path(git).resolve().parent
    for root in (base.parent, base.parent.parent):
        for cand in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe"):
            if cand.is_file():
                return str(cand)
    return None


def is_strict_lf(path):
    if path.suffix.lower() in STRICT_LF_SUFFIXES or path.name == "Dockerfile":
        return True
    try:
        with path.open("rb") as fh:
            return fh.read(2) == b"#!"
    except OSError:
        return False


def content_hashes(raw):
    return {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def norm_label(value):
    return re.sub(r"[\s\-]+", "_", str(value).strip().lower())


def shell_segments(command):
    for segment in re.split(r"&&|\|\||;|\|", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        while tokens and (re.match(r"^\w+=", tokens[0]) or tokens[0] in ("sudo", "env")):
            tokens = tokens[1:]
        if tokens:
            yield tokens


def dockerfile_instructions(text):
    # Docker drops full-line comments, even inside a line continuation.
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    joined = re.sub(r"\\[ \t]*\n", " ", "\n".join(lines))
    out = []
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped:
            parts = stripped.split(None, 1)
            out.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
    return out


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def check_structure(task_dir):
    missing = [r for r in REQUIRED_FILES if not (task_dir / r).is_file()]
    if missing:
        record(FAIL, "Required task files present", "missing: " + ", ".join(missing))
    else:
        record(PASS, "Required task files present")
    if (task_dir / "tests" / "Dockerfile").exists():
        record(WARN, "No separate tests/Dockerfile",
               "canonical TB2 tasks use environment/Dockerfile for both agent and verifier")


def check_extraneous(task_dir):
    name = "No extraneous files (no_extraneous_files)"
    tracked = git_ls(task_dir, "--cached")
    if tracked is None:
        junk = [rel(p, task_dir) for p in task_dir.rglob("*")
                if p.is_file() and ".git" not in p.parts and JUNK_RE.search(rel(p, task_dir))]
        if junk:
            record(WARN, name, "not a git checkout (or git refused it), so can't tell what gets "
                               "committed; junk on disk: " + short_list(junk))
        else:
            record(PASS, name)
        return
    untracked = git_ls(task_dir, "--others", "--exclude-standard") or []
    committed_junk = [p for p in tracked if JUNK_RE.search(p)]
    pending_junk = [p for p in untracked if JUNK_RE.search(p)]
    stray = sorted({p.split("/", 1)[0] for p in tracked + untracked} - TASK_ROOT_ENTRIES)
    if committed_junk:
        record(FAIL, name, "committed: " + short_list(committed_junk) + " (git rm --cached them)")
    elif pending_junk:
        record(WARN, name, "untracked and not ignored, `git add -A` would commit: " + short_list(pending_junk))
    else:
        record(PASS, name)
    if stray:
        record(WARN, "Nothing unexpected at task/ root",
               "reviewers flag files that don't carry weight: " + short_list(stray))


def check_line_endings(task_dir, files):
    name = "LF line endings"
    bad, loose, worktree_only = [], [], []
    report = git_out(task_dir, "ls-files", "--eol", "-z", "--cached", "--others", "--exclude-standard", "--", ".")
    if report is not None:
        for entry in report.split("\0"):
            if "\t" not in entry:
                continue
            info, path = entry.split("\t", 1)
            fields = info.split()
            index_eol = next((f[2:] for f in fields if f.startswith("i/")), "")
            work_eol = next((f[2:] for f in fields if f.startswith("w/")), "")
            full = task_dir / path
            if not full.is_file():
                continue
            crlf_index = index_eol in ("crlf", "mixed")
            crlf_work = work_eol in ("crlf", "mixed")
            if crlf_index or (not index_eol and crlf_work):
                (bad if is_strict_lf(full) else loose).append(path)
            elif crlf_work and is_strict_lf(full):
                worktree_only.append(path)
    else:
        for f in files:
            try:
                data = f.read_bytes()
            except OSError:
                continue
            if b"\0" not in data[:8192] and b"\r\n" in data:
                (bad if is_strict_lf(f) else loose).append(rel(f, task_dir))
    if bad:
        record(FAIL, name, "CRLF in: " + short_list(bad) +
               ". Fix: put `* text=auto eol=lf` in .gitattributes and run `git add --renormalize .`")
    else:
        record(PASS, name)
    if worktree_only:
        record(WARN, "LF line endings in your working copy",
               "committed as LF but checked out as CRLF (core.autocrlf), so local harbor runs see CRLF: "
               + short_list(worktree_only))
    if loose:
        record(WARN, "LF line endings (data files)", "CRLF in: " + short_list(loose))


def check_syntax(task_dir, files):
    py_bad = []
    for f in files:
        if f.suffix == ".py":
            try:
                compile(f.read_bytes(), str(f), "exec", dont_inherit=True)
            except (SyntaxError, ValueError) as exc:
                py_bad.append(f"{rel(f, task_dir)}:{getattr(exc, 'lineno', '?')}")
    if py_bad:
        record(FAIL, "Python files compile", "syntax errors: " + short_list(py_bad))
    else:
        record(PASS, "Python files compile")

    scripts = [f for f in files if f.suffix == ".sh"]
    if not scripts:
        return
    bash = find_bash()
    if not bash:
        record(SKIP, "Shell scripts parse (bash -n)", "bash not found")
        return
    sh_bad = []
    for f in scripts:
        rc, out = run([bash, "-n", f.as_posix()])
        if rc != 0:
            sh_bad.append(f"{rel(f, task_dir)}: {tail(out, 1)}")
    if sh_bad:
        record(FAIL, "Shell scripts parse (bash -n)", "\n".join(sh_bad))
    else:
        record(PASS, "Shell scripts parse (bash -n)")


def check_large_files(task_dir, files):
    big = []
    for f in files:
        try:
            if f.stat().st_size > LARGE_FILE_BYTES:
                big.append(f"{rel(f, task_dir)} ({f.stat().st_size // (1024 * 1024)} MB)")
        except OSError:
            pass
    if big:
        record(WARN, "No files over ~100 MB", "download these at build time instead: " + short_list(big))


# ---------------------------------------------------------------------------
# task.toml
# ---------------------------------------------------------------------------

def load_task_toml(task_dir):
    path = task_dir / "task.toml"
    if not path.is_file():
        return None, "not found"
    try:
        return tomllib.loads(read_text(path)), None
    except tomllib.TOMLDecodeError as exc:
        return None, str(exc)


def check_task_toml(data):
    meta = data.get("metadata", {})
    empty = [k for k in REQUIRED_METADATA if not meta.get(k)]
    if empty:
        record(FAIL, "task.toml metadata filled in", "empty/missing: " + ", ".join(empty))
    else:
        record(PASS, "task.toml metadata filled in")

    placeholders = [k for k, v in meta.items() if isinstance(v, str) and PLACEHOLDER_RE.search(v)]
    if placeholders:
        record(FAIL, "task.toml has no placeholder text", "TODO/TBD/<fill in> left in: " + ", ".join(placeholders))

    timeout = data.get("agent", {}).get("timeout_sec")
    if timeout is None:
        record(FAIL, f"[agent].timeout_sec set and <= {MAX_AGENT_TIMEOUT_SEC}", "missing")
    elif float(timeout) > MAX_AGENT_TIMEOUT_SEC:
        record(FAIL, f"[agent].timeout_sec set and <= {MAX_AGENT_TIMEOUT_SEC}",
               f"{timeout} is over the project-wide {MAX_AGENT_TIMEOUT_SEC}s cap; "
               "if trials time out at the cap, make the task faster instead")
    else:
        record(PASS, f"[agent].timeout_sec set and <= {MAX_AGENT_TIMEOUT_SEC}", f"{timeout}")

    if data.get("environment", {}).get("allow_internet") is not True:
        record(WARN, "[environment].allow_internet = true", "Dynamo tasks are expected to run with open internet")

    artifacts = data.get("artifacts", [])
    relative = [a for a in artifacts if not str(a).startswith("/")]
    if relative:
        record(FAIL, "artifacts use absolute paths", "relative: " + short_list(relative))


def check_diversity_labels(data, refs_dir):
    name = "Diversity labels match references/diversity-taxonomy.toml"
    taxonomy_path = refs_dir / "diversity-taxonomy.toml" if refs_dir else None
    if not taxonomy_path or not taxonomy_path.is_file():
        record(SKIP, name, "no references/diversity-taxonomy.toml next to task/")
        return
    try:
        taxonomy = tomllib.loads(read_text(taxonomy_path))
    except tomllib.TOMLDecodeError as exc:
        record(SKIP, name, f"couldn't parse taxonomy: {exc}")
        return

    meta = data.get("metadata", {})
    categories = {norm_label(k): {norm_label(v) for v in vals}
                  for k, vals in taxonomy.get("categories", {}).items()}
    problems = []
    category = norm_label(meta.get("category", ""))
    if category not in categories:
        problems.append(f"category {meta.get('category')!r} isn't a taxonomy category")
    for field in ("task_objective", "artifact_type"):
        allowed = {norm_label(v) for v in taxonomy.get(field, [])}
        values = meta.get(field) or []
        if isinstance(values, str):
            values = [values]
        unknown = [v for v in values if norm_label(v) not in allowed]
        if unknown:
            problems.append(f"{field} not in the closed set: {short_list(unknown)}")
    if problems:
        record(FAIL, name, "; ".join(problems))
    else:
        record(PASS, name)

    subcategory = meta.get("subcategory")
    if category in categories and subcategory and norm_label(subcategory) not in categories[category]:
        record(WARN, "Subcategory is a listed taxonomy value",
               f"{subcategory!r} isn't listed under {meta.get('category')!r} (allowed, but flagged); listed: "
               + short_list(sorted(categories[category]), limit=12))


# ---------------------------------------------------------------------------
# instruction.md
# ---------------------------------------------------------------------------

def check_instruction(task_dir):
    path = task_dir / "instruction.md"
    if not path.is_file():
        return
    text = read_text(path)

    match = BOILERPLATE_RE.search(text)
    if match:
        line = next(l for l in text.splitlines() if match.group(0) in l).strip()
        record(FAIL, "instruction.md has no time-budget / anti-cheat boilerplate",
               f"found: {line[:100]!r}. The rubric's instruction_concision criterion fails this line; "
               "the time budget lives only in task.toml [agent].timeout_sec")
    else:
        record(PASS, "instruction.md has no time-budget / anti-cheat boilerplate")

    if PLACEHOLDER_RE.search(text):
        record(FAIL, "instruction.md has no placeholder text", "found TODO/TBD/<fill in>-style text")

    approx_tokens = max(len(text) / 4, len(text.split()) * 1.33)
    if approx_tokens > INSTRUCTION_TOKEN_CAP * 1.2:
        record(FAIL, f"instruction.md under {INSTRUCTION_TOKEN_CAP} tokens",
               f"~{int(approx_tokens)} tokens (estimate)")
    elif approx_tokens > INSTRUCTION_TOKEN_CAP * 0.9:
        record(WARN, f"instruction.md under {INSTRUCTION_TOKEN_CAP} tokens",
               f"~{int(approx_tokens)} tokens (estimate), close to the cap; check with a real tokenizer")
    else:
        record(PASS, f"instruction.md under {INSTRUCTION_TOKEN_CAP} tokens", f"~{int(approx_tokens)} (estimate)")

    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    spans = re.findall(r"`([^`\n]+)`", prose)
    relative = sorted({s for s in spans if "://" not in s and REL_PATH_RE.match(s.strip())})
    if relative:
        record(WARN, "instruction.md uses absolute paths",
               "backticked paths that aren't absolute: " + short_list(relative)
               + " (fine if they're names inside an absolute directory stated nearby)")
    else:
        record(PASS, "instruction.md uses absolute paths")


# ---------------------------------------------------------------------------
# environment/Dockerfile
# ---------------------------------------------------------------------------

def approved_digests(refs_dir):
    script = refs_dir / "check-base-image.sh" if refs_dir else None
    if script and script.is_file():
        found = dict(re.findall(r'^\s*([\w.-]+)\)\s+echo\s+"(sha256:[0-9a-f]{64})"', read_text(script), re.MULTILINE))
        if found:
            return found
    return APPROVED_BASE_DIGESTS


def check_dockerfile(task_dir, refs_dir):
    path = task_dir / "environment" / "Dockerfile"
    if not path.is_file():
        return
    instructions = dockerfile_instructions(read_text(path))

    # base image
    digests = approved_digests(refs_dir)
    stages, wrong, unpinned, unapproved = set(), [], [], []
    for ins, args in instructions:
        if ins != "FROM":
            continue
        tokens = [t for t in args.split() if not t.startswith("--")]
        if not tokens:
            continue
        image = tokens[0]
        if len(tokens) >= 3 and tokens[1].lower() == "as":
            stages.add(tokens[2].lower())
        if image.lower() in stages or not re.search(r"[/:@]", image):
            continue
        name_tag, _, digest = image.partition("@")
        family = name_tag.rsplit("/", 1)[-1].split(":", 1)[0]
        approved = digests.get(family)
        if approved and digest == approved:
            continue
        if approved:
            wrong.append(f"{image} -> use public.ecr.aws/docker/library/{family}@{approved}")
        elif not digest:
            unpinned.append(image)
        else:
            unapproved.append(image)
    if wrong or unpinned:
        record(FAIL, "Base image is the pre-approved digest-pinned image",
               "\n".join(wrong + [f"{i} is not pinned by @sha256 digest" for i in unpinned]))
    else:
        record(PASS, "Base image is the pre-approved digest-pinned image")
    if unapproved:
        record(WARN, "Base image is one of the 10 approved families",
               short_list(unapproved) + " (allowed only if none of the approved bases fit)")

    # COPY/ADD of solution/ or tests/
    leaks = []
    for ins, args in instructions:
        if ins not in ("COPY", "ADD") or "--from" in args:
            continue
        stripped = args.strip()
        if stripped.startswith("["):
            try:
                tokens = json.loads(stripped)
            except json.JSONDecodeError:
                tokens = stripped.split()
        else:
            tokens = [t for t in stripped.split() if not t.startswith("--")]
        for src in tokens[:-1]:
            if {"solution", "tests"} & set(re.split(r"[\\/]+", src)):
                leaks.append(f"{ins} {args}")
                break
    if leaks:
        record(FAIL, "Dockerfile doesn't COPY solution/ or tests/", short_list(leaks))
    else:
        record(PASS, "Dockerfile doesn't COPY solution/ or tests/")

    # apt + pip hygiene
    no_update, no_clean, apt_pins, pip_unpinned, npm_unpinned = [], [], [], [], []
    for ins, args in instructions:
        if ins != "RUN":
            continue
        has_install = has_update = False
        for tokens in shell_segments(args):
            program = os.path.basename(tokens[0])
            if program in ("apt-get", "apt"):
                sub = next((t for t in tokens[1:] if not t.startswith("-")), None)
                has_update |= sub == "update"
                if sub == "install":
                    has_install = True
                    after = tokens[tokens.index("install") + 1:]
                    apt_pins += [t for t in after if not t.startswith("-") and "=" in t]
            for i, tok in enumerate(tokens):
                if tok == "install" and i > 0 and re.fullmatch(r"pip[\d.]*", os.path.basename(tokens[i - 1])):
                    skip_next = False
                    for arg in tokens[i + 1:]:
                        if skip_next:
                            skip_next = False
                            continue
                        if arg.startswith("-"):
                            skip_next = arg in PIP_VALUE_FLAGS
                            continue
                        if "/" in arg or arg.startswith(".") or "@" in arg or arg.endswith((".whl", ".tar.gz", ".zip")):
                            continue
                        if "==" not in arg:
                            pip_unpinned.append(arg)
            if program in ("npm", "pnpm", "yarn") and len(tokens) > 1 and tokens[1] in ("install", "i", "add"):
                pkgs = [t for t in tokens[2:] if not t.startswith("-")]
                npm_unpinned += [p for p in pkgs if "@" not in p.lstrip("@")]
        if has_install and not has_update:
            no_update.append(args[:70])
        if has_install and "/var/lib/apt/lists" not in args:
            no_clean.append(args[:70])

    if no_update:
        record(FAIL, "apt-get update runs in the same RUN as apt-get install", short_list(no_update, 3))
    if no_clean:
        record(WARN, "apt cache cleaned in the same RUN (rm -rf /var/lib/apt/lists/*)", short_list(no_clean, 3))
    if apt_pins:
        record(WARN, "apt packages not version-pinned", "pinned apt versions go stale: " + short_list(apt_pins))
    if pip_unpinned:
        record(FAIL, "pip dependencies pinned with ==", "unpinned: " + short_list(pip_unpinned))
    if npm_unpinned:
        record(WARN, "npm dependencies pinned", "unpinned: " + short_list(npm_unpinned))
    if not (no_update or pip_unpinned):
        record(PASS, "Dockerfile apt/pip hygiene")


# ---------------------------------------------------------------------------
# tests/
# ---------------------------------------------------------------------------

def check_verify_time_installs(task_dir):
    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        return
    installs, fetches = [], []
    for script in sorted(p for p in iter_files(tests_dir) if p.suffix == ".sh"):
        for n, line in enumerate(read_text(script).splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            where = f"{rel(script, task_dir)}:{n}: {line.strip()[:80]}"
            if INSTALL_RE.search(line):
                installs.append(where)
            elif FETCH_RE.search(line) and not LOCAL_URL_RE.search(line):
                fetches.append(where)
    if installs:
        record(FAIL, "tests/*.sh install nothing at verify time",
               "bake these into environment/Dockerfile instead:\n" + "\n".join(installs))
    else:
        record(PASS, "tests/*.sh install nothing at verify time")
    if fetches:
        record(WARN, "tests/*.sh download nothing at verify time", "\n".join(fetches))


def check_test_docstrings(task_dir):
    name = "Every test function has a docstring (test_instruction_alignment)"
    tests_dir = task_dir / "tests"
    files = [p for p in iter_files(tests_dir) if p.suffix == ".py"
             and (p.name.startswith("test_") or p.name.endswith("_test.py"))] if tests_dir.is_dir() else []
    if not files:
        record(WARN, name, "no test_*.py files found in tests/")
        return
    total, missing = 0, []
    for f in files:
        try:
            tree = ast.parse(f.read_bytes(), filename=str(f))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                total += 1
                if not ast.get_docstring(node):
                    missing.append(f"{rel(f, task_dir)}::{node.name}")
    if total == 0:
        record(WARN, name, "no test functions found")
    elif missing:
        record(FAIL, name, f"{len(missing)} of {total} have none, so reviewers can't trace them to "
                           "instruction.md: " + short_list(missing))
    else:
        record(PASS, name, f"{total} test functions")


# ---------------------------------------------------------------------------
# docker + harbor (slow)
# ---------------------------------------------------------------------------

def check_image_leaks(task_dir, data, verbose):
    name = "Built image holds no copy of solution/ or tests/ files"
    docker = shutil.which("docker")
    if not docker:
        record(FAIL, "environment/Dockerfile builds", "docker not found on PATH")
        return
    env_dir = task_dir / "environment"
    if not (env_dir / "Dockerfile").is_file():
        return

    task_name = str(data.get("task", {}).get("name") or task_dir.parent.name).split("/")[-1]
    tag = "dynamo-precheck-" + (re.sub(r"[^a-z0-9_.-]+", "-", task_name.lower()).strip("-.") or "task") + ":local"
    print("    docker build environment/ (reuses cached layers)...", flush=True)
    rc, out = run([docker, "build", "-t", tag, str(env_dir)], verbose=verbose)
    if rc != 0:
        record(FAIL, "environment/Dockerfile builds", "docker build failed:\n" + tail(out))
        return
    record(PASS, "environment/Dockerfile builds")

    # Files byte-identical to something in environment/ are public inputs that the
    # verifier keeps its own copy of, not leaks.
    public = set()
    for p in iter_files(env_dir):
        public |= content_hashes(p.read_bytes())
    secret = {}
    for sub in ("solution", "tests"):
        for p in iter_files(task_dir / sub):
            raw = p.read_bytes()
            hashes = content_hashes(raw)
            if len(raw) < 16 or hashes & public:
                continue
            for h in hashes:
                secret.setdefault(h, rel(p, task_dir))
    if not secret:
        record(PASS, name, "no solution/ or tests/ files to look for")
        return

    names = sorted({Path(r).name for r in secret.values()})
    find_expr = " -o ".join("-name " + shlex.quote(n) for n in names)
    script = f"find / -xdev -type f \\( {find_expr} \\) -exec sha256sum {{}} + 2>/dev/null; exit 0"
    rc, out = run([docker, "run", "--rm", "--network", "none", "--entrypoint", "sh", tag, "-c", script],
                  verbose=verbose)
    hits = [m for m in (re.match(r"^([0-9a-f]{64})\s+(.+)$", line) for line in out.splitlines()) if m]
    if rc != 0 and not hits:
        record(WARN, name, f"couldn't run `sh` inside {tag}:\n" + tail(out, 5))
        return
    leaked = [f"{m.group(2)} (= {secret[m.group(1)]})" for m in hits if m.group(1) in secret]
    if leaked:
        record(FAIL, name, "the agent can read: " + short_list(leaked))
    else:
        record(PASS, name)


def extract_reward_from_job(job_dir):
    """
    Primary source: stats.evals.<agent>__adhoc.metrics[0].mean in the job's
    result.json, the number Harbor itself computed. Falls back to the
    per-trial verifier/reward.txt.
    """
    result_json = job_dir / "result.json"
    if result_json.exists():
        try:
            data = json.loads(read_text(result_json))
            for eval_data in data.get("stats", {}).get("evals", {}).values():
                metrics = eval_data.get("metrics") or []
                if metrics and "mean" in metrics[0]:
                    return float(metrics[0]["mean"])
        except (json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError):
            pass
    reward_files = sorted(job_dir.rglob("verifier/reward.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for reward_file in reward_files:
        try:
            return float(read_text(reward_file).strip())
        except (ValueError, OSError):
            pass
    return None


def check_oracle_and_nop(task_dir, verbose):
    harbor = shutil.which("harbor")
    if not harbor:
        record(FAIL, "harbor oracle/nop runs", "`harbor` not found on PATH")
        return
    # Keep job output out of task/ so it can never be committed by accident.
    jobs_dir = Path(tempfile.mkdtemp(prefix="dynamo-precheck-jobs-"))
    keep_logs = False
    for agent in ("oracle", "nop"):
        job_name = f"precheck-{agent}-{int(time.time())}"
        print(f"    harbor run --agent {agent} (can take several minutes)...", flush=True)
        rc, out = run([harbor, "run", "-p", str(task_dir), "--agent", agent,
                       "--jobs-dir", str(jobs_dir), "--job-name", job_name], cwd=task_dir, verbose=verbose)
        job_dir = jobs_dir / job_name
        reward = extract_reward_from_job(job_dir) if job_dir.is_dir() else None
        if reward is None:
            m = re.search(r"reward[:=]?\s+?([01](?:\.\d+)?)", out, re.IGNORECASE)
            reward = float(m.group(1)) if m else None

        label = "oracle scores reward 1.0" if agent == "oracle" else "nop scores reward < 1.0"
        if reward is None:
            keep_logs = True
            status = FAIL if rc != 0 else WARN
            record(status, f"harbor {label}",
                   f"no reward found (exit {rc}); job output kept in {job_dir}\n" + tail(out))
        elif (agent == "oracle" and reward == 1.0) or (agent == "nop" and reward < 1.0):
            record(PASS, f"harbor {label}", f"reward {reward}")
        else:
            keep_logs = True
            why = ("your solution doesn't pass your own tests" if agent == "oracle"
                   else "doing nothing already passes, so the verifier is too weak")
            record(FAIL, f"harbor {label}", f"got reward {reward}: {why}. Logs: {job_dir}")
    if not keep_logs:
        shutil.rmtree(jobs_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# git pre-push hook
# ---------------------------------------------------------------------------

HOOK_BODY = """#!/bin/sh
# Installed by `dynamo-precheck --install-hook`.
# Blocks the push if any check FAILs. Skip once with: git push --no-verify
exec dynamo-precheck "$(git rev-parse --show-toplevel)" < /dev/null
"""


def install_hook(start):
    top = git_out(start, "rev-parse", "--show-toplevel")
    if top is None:
        print("Not inside a git repository (or git refused it; run `git status` to see why).")
        sys.exit(1)
    root = Path(top.strip())
    hooks = git_out(root, "rev-parse", "--git-path", "hooks")
    hooks_dir = root / hooks.strip() if hooks else root / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / "pre-push"
    if hook.exists() and "dynamo-precheck" not in read_text(hook):
        backup = hook.with_name("pre-push.before-dynamo-precheck")
        hook.replace(backup)
        print(f"Moved your existing pre-push hook to {backup}")
    hook.write_text(HOOK_BODY, encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    print(f"Installed pre-push hook at {hook}")
    print("Every `git push` now runs dynamo-precheck first and is blocked if any check fails.")
    print("To skip it once: git push --no-verify")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    ap = argparse.ArgumentParser(prog="dynamo-precheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=".",
                    help="task repo root or its task/ directory (default: current directory)")
    ap.add_argument("--skip-harbor", "--fast", action="store_true",
                    help="static checks only; skip docker build, leak check and harbor runs")
    ap.add_argument("--keep-going", action="store_true",
                    help="run the slow docker/harbor stage even if static checks failed")
    ap.add_argument("-v", "--verbose", action="store_true", help="print raw docker/harbor output")
    ap.add_argument("--install-hook", action="store_true",
                    help="install a git pre-push hook that runs this before every push, then exit")
    ap.add_argument("--version", action="version", version=f"dynamo-precheck {__version__}")
    args = ap.parse_args()

    if args.install_hook:
        install_hook(Path(args.path))
        return

    task_dir = find_task_dir(Path(args.path))
    if task_dir is None:
        print("Couldn't find task.toml here, in a task/ subfolder, or in any parent directory.")
        print("Run this from inside a Dynamo task repo.")
        sys.exit(2)
    refs_dir = task_dir.parent / "references"
    refs_dir = refs_dir if refs_dir.is_dir() else None

    print(f"dynamo-precheck {__version__}: checking {task_dir}\n")
    files = list_task_files(task_dir)
    data, toml_error = load_task_toml(task_dir)

    print("-- structure --")
    check_structure(task_dir)
    check_extraneous(task_dir)
    check_line_endings(task_dir, files)
    check_syntax(task_dir, files)
    check_large_files(task_dir, files)

    print("\n-- task.toml --")
    if data is None:
        record(FAIL, "task.toml parses", toml_error)
    else:
        check_task_toml(data)
        check_diversity_labels(data, refs_dir)

    print("\n-- instruction.md --")
    check_instruction(task_dir)

    print("\n-- environment/Dockerfile --")
    check_dockerfile(task_dir, refs_dir)

    print("\n-- tests/ --")
    check_verify_time_installs(task_dir)
    check_test_docstrings(task_dir)

    static_failed = any(s == FAIL for s, _, _ in results)
    harbor_ran = False
    if args.skip_harbor:
        print("\n(--skip-harbor: skipped docker build, leak check and oracle/nop runs)")
    elif static_failed and not args.keep_going:
        print("\n(skipped the slow docker/harbor stage because static checks failed; "
              "fix those first, or pass --keep-going)")
    else:
        harbor_ran = True
        print("\n-- docker + harbor (slow) --")
        check_image_leaks(task_dir, data or {}, args.verbose)
        check_oracle_and_nop(task_dir, args.verbose)

    counts = {s: sum(1 for r in results if r[0] == s) for s in MARKERS}
    print(f"\n{len(results)} checks: {counts[PASS]} passed, {counts[WARN]} warnings, "
          f"{counts[FAIL]} failed" + (f", {counts[SKIP]} skipped" if counts[SKIP] else "") + ".")

    failed = [name for status, name, _ in results if status == FAIL]
    if failed:
        print("\nNOT READY TO PUSH. Fix:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    if harbor_ran:
        print("\nReady to push. pass@2 / pass@5, the rubric review and the duplicate check "
              "still only run on the hosted pipeline.")
    else:
        print("\nStatic checks are clean. Run `dynamo-precheck` without --skip-harbor for the full check.")
    sys.exit(0)


if __name__ == "__main__":
    main()
