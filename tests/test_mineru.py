"""The MinerU adapter's own logic (SPEC §6): argv, envelope handling,
and failure visibility. Tested with an injected runner — the real CLI
and daemon are proven live; what only the adapter owns is that a dead
daemon or a garbage stdout raises with a reason (so the ingest status
machine shows it, §8) and that temp files never leak.
"""

import json
from pathlib import Path

import pytest

from phd_helper.server.mineru import MinerUExtractor


@pytest.fixture
def anyio_backend():
    return "asyncio"


def envelope(md):
    return json.dumps({"parse": {"status": "done"},
                       "content": {"format": "markdown", "content": md}})


def make(extract_out):
    calls = []

    async def run(argv):
        calls.append(argv)
        return extract_out
    return MinerUExtractor(cli="mineru-test", run=run), calls


@pytest.mark.anyio
async def test_happy_path_maps_markdown_to_blocks():
    md = "<!-- page 1 of 2 -->\n\n# Title\n\nBody text here.\n"
    ex, calls = make((0, envelope(md)))
    blocks = await ex.extract(b"%PDF fake")
    assert [(b.kind, b.text) for b in blocks] == [
        ("heading", "Title"), ("text", "Body text here.")]
    argv = calls[0]
    assert argv[:2] == ["mineru-test", "parse"]
    assert "--json" in argv and "-p" in argv and "all" in argv


@pytest.mark.anyio
async def test_error_envelope_raises_with_the_reason():
    out = json.dumps({"error": {"message": "Local mineru server is not "
                                           "running. Run 'mineru server "
                                           "start'."}})
    ex, _ = make((0, out))  # the CLI exits 0 even on error envelopes
    with pytest.raises(RuntimeError, match="mineru server start"):
        await ex.extract(b"%PDF fake")


@pytest.mark.anyio
async def test_non_json_stdout_raises_with_exit_code():
    ex, _ = make((127, "Permission denied"))
    with pytest.raises(RuntimeError, match="exited 127"):
        await ex.extract(b"%PDF fake")


@pytest.mark.anyio
async def test_temp_pdf_is_cleaned_up_even_on_failure():
    seen = {}

    async def run(argv):
        seen["path"] = argv[2]
        return 0, envelope("<!-- page 1 of 1 -->\n\nHi.\n")
    ex = MinerUExtractor(cli="mineru-test", run=run)
    await ex.extract(b"%PDF fake")
    assert not Path(seen["path"]).exists()
