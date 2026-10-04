"""The per-project git kernel (issue: every project a repo).

Two layers, like the mineru adapter: pure parsers over git's own text
output (porcelain, log format, remote URLs, git.json), and the async
operations over an injectable `run` seam — argv is the contract, so a
fake run proves what the door will actually execute. The real-git
integration tests at the bottom prove the seam against real git
(skipif no git on PATH), including a push to a bare remote in tmp_path
(no auth, no network).
"""

import json
import shutil
from pathlib import Path

import pytest

from phd_helper import gitrepo

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# -- pure kernels -----------------------------------------------------------


def test_gitignore_hides_agent_state_and_build_dross():
    lines = gitrepo.GITIGNORE_CONTENT.splitlines()
    assert ".phd-helper/" in lines  # agent state is not the paper
    assert "*.aux" in lines and "*.log" in lines and "*.pdf" in lines


def test_parse_porcelain_clean_output_is_not_dirty():
    assert gitrepo.parse_porcelain("") == {"dirty": False, "paths": []}
    assert gitrepo.parse_porcelain("\n") == {"dirty": False, "paths": []}


def test_parse_porcelain_sees_untracked_and_modified():
    out = "?? main.tex\n M sections/intro.tex\n"
    got = gitrepo.parse_porcelain(out)
    assert got["dirty"] is True
    assert got["paths"] == ["main.tex", "sections/intro.tex"]


def test_parse_porcelain_rename_reports_the_new_path():
    got = gitrepo.parse_porcelain("R  old.tex -> new.tex\n")
    assert got["dirty"] is True and got["paths"] == ["new.tex"]


def test_parse_porcelain_strips_git_quotes_on_odd_names():
    got = gitrepo.parse_porcelain('?? "weird name.tex"\n')
    assert got["paths"] == ["weird name.tex"]


def test_parse_last_commit_parses_the_agreed_format():
    got = gitrepo.parse_last_commit("1a2b3c4|2026-10-04T09:00:00+02:00|"
                                    "Auto-commit at sitting end (switch)")
    assert got == {"sha": "1a2b3c4",
                   "date": "2026-10-04T09:00:00+02:00",
                   "subject": "Auto-commit at sitting end (switch)"}


def test_parse_last_commit_subject_may_contain_pipes():
    got = gitrepo.parse_last_commit("abc|2026-01-01T00:00:00+00:00|a | b")
    assert got["subject"] == "a | b"


def test_parse_last_commit_of_an_empty_repo_is_none():
    assert gitrepo.parse_last_commit("") is None


def test_valid_remote_url_accepts_the_real_ways_to_point_at_a_repo():
    for url in ("https://github.com/u/r.git", "http://host/r.git",
                "git@github.com:u/r.git", "ssh://git@host:22/srv/r.git",
                "/srv/git/r.git", "C:/repos/r.git", "C:\\repos\\r.git"):
        assert gitrepo.valid_remote_url(url), url


def test_valid_remote_url_rejects_things_that_are_not_urls():
    for url in ("", "   ", "origin", "main", "some/relative",
                "https://", "git@", "C:", "--upload-pack=evil"):
        assert not gitrepo.valid_remote_url(url), url


def test_remote_round_trips_through_the_state_dir(tmp_path):
    assert gitrepo.read_remote(tmp_path) == ""  # no git.json yet
    gitrepo.write_remote(tmp_path, "https://github.com/u/r.git")
    assert gitrepo.read_remote(tmp_path) == "https://github.com/u/r.git"
    stored = json.loads(
        (tmp_path / ".phd-helper" / "git.json").read_text(encoding="utf-8"))
    assert stored["remote"] == "https://github.com/u/r.git"


def test_write_remote_replaces_the_previous_url(tmp_path):
    gitrepo.write_remote(tmp_path, "https://a.example/r.git")
    gitrepo.write_remote(tmp_path, "https://b.example/r.git")
    assert gitrepo.read_remote(tmp_path) == "https://b.example/r.git"


def test_gitignore_merge_keeps_the_users_own_lines(tmp_path):
    merged = gitrepo.gitignore_with("build/\n*.aux\n")
    assert merged.startswith("build/\n")  # the user's line survives
    assert ".phd-helper/" in merged
    assert merged.count("*.aux") == 1  # already there — not duplicated


# -- the run seam: argv is the contract ---------------------------------------


def fake_run(script):
    """A run(argv, cwd) that answers by subcommand label and records
    every argv it saw. `script` maps label -> (rc, out, err) or a
    callable(argv) -> (rc, out, err)."""
    calls = []

    async def run(argv, cwd):
        calls.append(list(argv))
        label = gitrepo._label(argv)
        step = script.get(label, (0, "", ""))
        return step(argv) if callable(step) else step

    return run, calls


async def test_commit_stages_then_commits_then_reads_the_sha(tmp_path):
    run, calls = fake_run({
        "status": (0, "M  main.tex\n", ""),
        "commit": (0, "", ""),
        "rev-parse": (0, "1a2b3c4\n", ""),
    })
    sha = await gitrepo.commit(tmp_path, "message", "N", "e@x", run=run)
    assert sha == "1a2b3c4"
    assert calls[0] == ["add", "-A"]
    assert "commit" in calls[2] and "-m" in calls[2]
    # identity rides the command, never global config
    assert "-c" in calls[2] and "user.name=N" in calls[2]
    assert "user.email=e@x" in calls[2]


async def test_commit_of_a_clean_tree_is_not_an_error(tmp_path):
    run, calls = fake_run({"status": (0, "", "")})
    assert await gitrepo.commit(tmp_path, "m", "N", "e@x", run=run) is None
    assert not any("commit" in c for c in calls)  # nothing to commit


async def test_a_failing_git_raises_with_the_subcommand_and_stderr(tmp_path):
    run, _ = fake_run({"status": (0, "M  main.tex\n", ""),
                       "commit": (128, "", "fatal: " + "x" * 500)})
    with pytest.raises(RuntimeError) as e:
        await gitrepo.commit(tmp_path, "m", "N", "e@x", run=run)
    assert "git commit failed (128)" in str(e.value)
    assert "fatal: x" in str(e.value)
    assert "x" * 500 not in str(e.value)  # the stderr slice, not the ocean


async def test_ensure_repo_inits_once_and_never_again(tmp_path):
    run, calls = fake_run({"status": (0, "?? main.tex\n", ""),
                           "commit": (0, "", ""),
                           "rev-parse": (0, "abc1234\n", "")})
    await gitrepo.ensure_repo(tmp_path, "N", "e@x", run=run)
    assert calls[0] == ["init", "-b", "main"]
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8") \
        == gitrepo.GITIGNORE_CONTENT
    (tmp_path / ".git").mkdir()  # what a real init would have left
    n = len(calls)
    await gitrepo.ensure_repo(tmp_path, "N", "e@x", run=run)  # second time
    assert len(calls) == n  # a repo that exists is left alone entirely


async def test_ensure_repo_merges_an_existing_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    run, _ = fake_run({"status": (0, "?? x\n", ""), "commit": (0, "", ""),
                       "rev-parse": (0, "abc\n", "")})
    await gitrepo.ensure_repo(tmp_path, "N", "e@x", run=run)
    text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert text.startswith("build/\n") and ".phd-helper/" in text


async def test_status_reports_initialized_dirty_and_last(tmp_path):
    (tmp_path / ".git").mkdir()
    run, _ = fake_run({"status": (0, " M a.tex\n", ""),
                       "log": (0, "abc|2026-01-01T00:00:00+00:00|msg\n",
                               "")})
    got = await gitrepo.status(tmp_path, run=run)
    assert got["initialized"] is True and got["dirty"] is True
    assert got["last"]["sha"] == "abc"


async def test_status_of_an_uninitialized_dir_never_touches_git(tmp_path):
    run, calls = fake_run({})
    got = await gitrepo.status(tmp_path, run=run)
    assert got == {"initialized": False, "dirty": False, "last": None}
    assert calls == []


async def test_status_survives_a_repo_with_no_commits_yet(tmp_path):
    (tmp_path / ".git").mkdir()
    run, _ = fake_run({"status": (0, "?? main.tex\n", ""),
                       "log": (128, "", "fatal: bad revision 'HEAD'")})
    got = await gitrepo.status(tmp_path, run=run)
    assert got["dirty"] is True and got["last"] is None


async def test_push_targets_the_url_and_main_directly(tmp_path):
    run, calls = fake_run({"push": (0, "", "")})
    await gitrepo.push(tmp_path, "https://example/r.git", run=run)
    assert calls[0] == ["push", "https://example/r.git", "main"]


async def test_push_can_carry_a_credential_helper_before_the_subcommand(
        tmp_path):
    run, calls = fake_run({"push": (0, "", "")})
    await gitrepo.push(tmp_path, "https://example/r.git",
                       credential_helper="manager", run=run)
    assert calls[0] == ["-c", "credential.helper=manager",
                        "push", "https://example/r.git", "main"]


async def test_push_failure_raises_with_git_stderr(tmp_path):
    run, _ = fake_run({"push": (1, "", "remote: Permission denied")})
    with pytest.raises(RuntimeError) as e:
        await gitrepo.push(tmp_path, "https://example/r.git", run=run)
    assert "git push failed (1)" in str(e.value)
    assert "Permission denied" in str(e.value)


async def test_auto_commit_inits_then_commits(tmp_path):
    run, calls = fake_run({"status": (0, "?? main.tex\n", ""),
                           "commit": (0, "", ""),
                           "rev-parse": (0, "deadbee\n", "")})
    sha = await gitrepo.auto_commit(tmp_path, "Auto-commit", "N", "e@x",
                                    run=run)
    assert sha == "deadbee"
    assert calls[0] == ["init", "-b", "main"]


async def test_auto_commit_of_a_clean_project_is_silent(tmp_path):
    (tmp_path / ".git").mkdir()
    run, calls = fake_run({"status": (0, "", "")})
    assert await gitrepo.auto_commit(tmp_path, "m", "N", "e@x",
                                     run=run) is None
    assert not any("init" in c for c in calls)


# -- real git: the seam against the real thing --------------------------------


needs_git = pytest.mark.skipif(not shutil.which("git"),
                               reason="git is not installed")


@needs_git
async def test_real_round_trip_init_commit_status(tmp_path):
    (tmp_path / "main.tex").write_text("\\title{X}\n", encoding="utf-8")
    await gitrepo.ensure_repo(tmp_path, "T", "t@t")
    assert (tmp_path / ".git").is_dir()
    got = await gitrepo.status(tmp_path)
    assert got["initialized"] and not got["dirty"]  # the initial commit
    assert got["last"]["subject"] == "phd-helper: initial commit"
    # agent state stays out of the repo
    (tmp_path / ".phd-helper").mkdir(exist_ok=True)
    (tmp_path / ".phd-helper" / "chat.jsonl").write_text("{}\n",
                                                         encoding="utf-8")
    assert not (await gitrepo.status(tmp_path))["dirty"]
    # an edit makes it dirty; a commit lands it
    (tmp_path / "main.tex").write_text("\\title{Y}\n", encoding="utf-8")
    assert (await gitrepo.status(tmp_path))["dirty"]
    sha = await gitrepo.commit(tmp_path, "second", "T", "t@t")
    assert sha and not (await gitrepo.status(tmp_path))["dirty"]
    assert (await gitrepo.status(tmp_path))["last"]["subject"] == "second"


@needs_git
async def test_real_push_to_a_bare_remote(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "main.tex").write_text("\\title{X}\n", encoding="utf-8")
    await gitrepo.ensure_repo(work, "T", "t@t")
    bare = tmp_path / "remote.git"
    proc = await gitrepo._run(["init", "--bare", str(bare)], tmp_path)
    assert proc[0] == 0
    await gitrepo.push(work, str(bare))
    # the remote really holds the commit — on main, which is what the
    # push created (the bare HEAD may name another branch entirely)
    rc, out, _ = await gitrepo._run(["log", "--format=%s", "main"], bare)
    assert rc == 0 and "phd-helper: initial commit" in out


@needs_git
async def test_autocrlf_cannot_make_the_tree_permanently_dirty(tmp_path,
                                                               monkeypatch):
    # A box with core.autocrlf=true would restage every LF file after
    # every commit; the -c flag on every call is what pins that down.
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    (tmp_path / "main.tex").write_text("line\nline2\n", encoding="utf-8")
    await gitrepo.ensure_repo(tmp_path, "T", "t@t")
    assert not (await gitrepo.status(tmp_path))["dirty"]
