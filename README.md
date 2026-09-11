# dynamo-precheck

One command that runs everything in the Dynamo PR pipeline that can run on your own machine, so a push only fails for reasons that genuinely need the hosted pipeline.

## Install

```
uv tool install git+https://github.com/Rishitha-riya06/dynamo-precheck
```

Needs [uv](https://astral.sh/uv), plus Docker and `harbor` on your PATH for the full run. The repo is private, so you need to be a collaborator; git uses your normal GitHub login to fetch it.

Update later with:

```
uv tool install --reinstall git+https://github.com/Rishitha-riya06/dynamo-precheck
```

## Use

From anywhere inside a task repo (the repo root or `task/`):

```
dynamo-precheck
```

Or once per repo, so it runs by itself before every `git push` and blocks the push on a failure:

```
dynamo-precheck --install-hook
```

Options:

- `--skip-harbor` / `--fast`: static checks only (a few seconds)
- `--keep-going`: run the docker/harbor stage even if static checks already failed (by default it stops early so you get feedback fast)
- `-v`: print raw docker/harbor output
- `git push --no-verify`: skip the hook for one push

Exit code is 0 only if no check failed. `[WARN]` lines are advisory.

## What it checks

**Structure**
- required files exist (`instruction.md`, `task.toml`, `environment/Dockerfile`, `solution/solve.sh`, `tests/test.sh`)
- no junk committed (`__pycache__`, `.pyc`, `jobs/`, editor backups, `.DS_Store`) and nothing unexpected at `task/` root
- LF line endings in what git will commit (and a warning if your checkout converts them to CRLF locally)
- every `.py` compiles, every `.sh` passes `bash -n`

**task.toml**
- the six fellow-filled metadata fields are set, with no TODO/TBD placeholders
- `task_objective`, `artifact_type` and `category` are in `references/diversity-taxonomy.toml` (unlisted subcategory is a warning)
- `[agent].timeout_sec` is at most 3600
- `artifacts` paths are absolute, `allow_internet = true`

**instruction.md**
- no "You have N seconds… do not cheat" boilerplate (the rubric's `instruction_concision` criterion fails it; the time budget belongs only in `task.toml`)
- under the 1500-token cap (estimate)
- backticked paths are absolute (warning)

**environment/Dockerfile**
- `FROM` is the exact pre-approved digest from `references/check-base-image.sh`
- no `COPY`/`ADD` of `solution/` or `tests/`
- `apt-get update` in the same `RUN` as `apt-get install`, apt cache cleaned, apt versions not pinned
- every `pip install` package pinned with `==`

**tests/**
- `tests/*.sh` install nothing at verify time
- every test function has a docstring

**Docker + Harbor (the slow part)**
- `environment/Dockerfile` builds
- the built image contains no copy of any `solution/` or `tests/` file (checked by content hash, so public inputs the verifier keeps its own copy of don't count)
- `harbor run --agent oracle` scores 1.0
- `harbor run --agent nop` scores below 1.0

Harbor job output goes to a temp directory, never into `task/`, and is deleted unless something failed.

## What it can't check

pass@2, pass@5, the rubric review, the automated deep review and the duplicate check all need the hosted agent. A clean run here means the task won't bounce for a locally-checkable reason, not that it will be accepted.

## License

MIT
