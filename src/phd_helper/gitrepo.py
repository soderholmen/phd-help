"""Per-project git (issue: every project a repo, commit + push doors).

Each project directory can become a git repo the user commits and
pushes when they are done. Two layers, like the mineru adapter: pure
parsers over git's own text output, and async operations over an
injectable `run` seam — argv is the contract, so tests prove the exact
commands without spawning git.

The posture this module keeps:
- **Lazy init.** Nothing here runs until a commit/status is asked for;
  `ensure_repo` is idempotent and the doors call it first.
- **Identity per command.** `-c user.name=… -c user.email=…` rides the
  commit itself; the server never writes global or system config.
- **`-c core.autocrlf=false` on every call** (in the real runner): a
  box with autocrlf=true would restage every LF file after every
  commit, leaving the tree permanently dirty.
- **`GIT_TERMINAL_PROMPT=0`**: Git Credential Manager must fail fast
  rather than hang a request waiting for a GUI this server has.
- **Direct-URL push** (`push <url> main`): the remote is stored in
  the project's `.phd-helper/git.json`, not in git config, so the
  door needs no repo mutation to set it.

Subprocess_exec (never a shell) is the injection guard; URL validation
is UX, not security.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

# The paper is the repo; agent state and LaTeX build dross are not.
GITIGNORE_CONTENT = ".phd-helper/\n*.aux\n*.log\n*.pdf\n"

INITIAL_COMMIT = "phd-helper: initial commit"
LOG_FORMAT = "%h|%cI|%s"
_TIMEOUT_S = 60.0


def default_message(prefix: str) -> str:
    """The auto-dated message a commit falls back to when the user (or
    the voice) passed none — the door and the voice tool share it."""
    return f"{prefix} ({time.strftime('%Y-%m-%d %H:%M')})"


# -- pure kernels -----------------------------------------------------------


def parse_porcelain(out: str) -> dict:
    """`git status --porcelain` -> {dirty, paths}. Rename/copy lines
    report the new path; git's C-style quoting on odd names is
    stripped (escapes inside stay — an honest edge, paths are only
    ever shown, never spliced)."""
    paths = []
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        paths.append(path)
    return {"dirty": bool(paths), "paths": paths}


def parse_last_commit(out: str) -> dict | None:
    """`git log -1 --format=%h|%cI|%s` -> {sha, date, subject}; the
    subject may itself contain pipes, so split from the left twice.
    Empty output (no commits yet) is None."""
    line = out.strip()
    if not line:
        return None
    parts = line.split("|", 2)
    if len(parts) != 3:
        return None
    return {"sha": parts[0], "date": parts[1], "subject": parts[2]}


_SCP_LIKE = re.compile(r"^[A-Za-z0-9._+-]+@[^:@/]+:.+$")
_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def valid_remote_url(url: str) -> bool:
    """http(s)://, ssh://, scp-like git@host:path, or an absolute local
    path (a bare repo on disk is a real remote — the tests push to
    one). Whitespace and leading dashes are refused: subprocess_exec
    already rules out shell injection, this keeps the stored string
    honest and unambiguous."""
    u = (url or "").strip()
    if not u or u.startswith("-") or any(c.isspace() for c in u):
        return False
    if u.startswith(("http://", "https://", "ssh://")):
        rest = u.split("://", 1)[1]
        host = rest.split("/", 1)[0].rsplit("@", 1)[-1].split(":", 1)[0]
        return bool(host)
    if _SCP_LIKE.match(u):
        return True
    return u.startswith("/") or bool(_WINDOWS_ABS.match(u))


def read_remote(root: Path) -> str:
    """The stored push target, or "" (no git.json, or unreadable —
    agent state is allowed to be broken without sinking a door)."""
    try:
        data = json.loads(
            (root / ".phd-helper" / "git.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("remote", ""))


def write_remote(root: Path, url: str) -> None:
    d = root / ".phd-helper"
    d.mkdir(parents=True, exist_ok=True)
    (d / "git.json").write_text(json.dumps({"remote": url}, indent=2) + "\n",
                                encoding="utf-8")


def gitignore_with(existing: str) -> str:
    """The user's own .gitignore lines survive; only the patterns the
    kernel needs (agent state, build dross) are appended if missing."""
    have = {ln.strip() for ln in existing.splitlines() if ln.strip()}
    base = existing if not existing or existing.endswith("\n") \
        else existing + "\n"
    add = [p for p in GITIGNORE_CONTENT.splitlines() if p not in have]
    return base + ("\n".join(add) + "\n" if add else "")


# -- the run seam -----------------------------------------------------------


def _label(argv: list[str]) -> str:
    """The subcommand a argv will run, skipping `-c <pair>` prefixes —
    the name that goes in the error message and the test script."""
    i = 0
    while i < len(argv):
        if argv[i] == "-c":
            i += 2
            continue
        return argv[i]
    return "git"


async def _run(argv: list[str], cwd: Path,
               timeout_s: float = _TIMEOUT_S) -> tuple[int, str, str]:
    """The real seam: git, pinned to cwd, autocrlf off, prompts off.
    Returns (rc, stdout, stderr) — git argues on stderr, so unlike
    mineru's runner this one keeps it."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", str(cwd), "-c", "core.autocrlf=false", *argv,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout_s)
    except TimeoutError:
        proc.kill()  # a git child that outlived its door (a push into a
        raise        # blackholed network) must not orphan on the box
    return (proc.returncode,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"))


def _fail(argv, rc, err) -> None:
    raise RuntimeError(f"git {_label(argv)} failed ({rc}): {err[:300]}")


async def _ok(run, argv, cwd) -> str:
    rc, out, err = await run(argv, cwd)
    if rc != 0:
        _fail(argv, rc, err)
    return out


# -- the operations ---------------------------------------------------------


async def ensure_repo(root: Path, name: str, email: str, run=None) -> None:
    """Idempotent: init on main, merge the .gitignore, initial commit.
    A repo that exists is left completely alone."""
    if (root / ".git").is_dir():
        return
    run = run or _run
    await _ok(run, ["init", "-b", "main"], root)
    gi = root / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.is_file() else ""
    gi.write_text(gitignore_with(existing), encoding="utf-8")
    await commit(root, INITIAL_COMMIT, name, email, run=run)


async def commit(root: Path, message: str, name: str, email: str,
                 run=None) -> str | None:
    """Stage everything the .gitignore lets in, commit if that made a
    difference, return the short sha — or None for a clean tree,
    which is a success, not an error."""
    run = run or _run
    await _ok(run, ["add", "-A"], root)
    if not parse_porcelain(await _ok(run, ["status", "--porcelain"],
                                     root))["dirty"]:
        return None
    await _ok(run, ["-c", f"user.name={name}", "-c", f"user.email={email}",
                    "commit", "-m", message], root)
    return (await _ok(run, ["rev-parse", "--short", "HEAD"], root)).strip()


async def status(root: Path, run=None) -> dict:
    """{initialized, dirty, last} — an uninitialized directory is a
    zero answer, not an error; a repo with no commits yet has no
    `last` (git log fails there, and that is expected)."""
    if not (root / ".git").is_dir():
        return {"initialized": False, "dirty": False, "last": None}
    run = run or _run
    dirty = parse_porcelain(
        await _ok(run, ["status", "--porcelain"], root))["dirty"]
    rc, out, _err = await run(["log", "-1", f"--format={LOG_FORMAT}"], root)
    return {"initialized": True, "dirty": dirty,
            "last": parse_last_commit(out) if rc == 0 else None}


async def push(root: Path, url: str, credential_helper: str = "",
                run=None) -> None:
    """Push main to the stored URL directly (no remote-name config).
    The credential helper, when configured, rides as `-c` before the
    subcommand — the knob for a box whose GCM cannot authenticate
    non-interactively (docs note it)."""
    run = run or _run
    argv = (["-c", f"credential.helper={credential_helper}"]
            if credential_helper else []) + ["push", url, "main"]
    await _ok(run, argv, root)


async def auto_commit(root: Path, message: str, name: str, email: str,
                      run=None) -> str | None:
    """ensure + commit: what the sitting-end hook and the push door's
    dirty-tree guard call. Clean tree -> None, silently."""
    await ensure_repo(root, name, email, run=run)
    return await commit(root, message, name, email, run=run)
