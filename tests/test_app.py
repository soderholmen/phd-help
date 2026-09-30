"""App assembly at the HTTP seam (SPEC §1): startup/shutdown wiring and
turn-error containment. The voice loop itself is proven end-to-end against
live vLLM (§2); this file covers what only the app owns.
"""

import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace

from phd_helper.corpus import Corpus
from phd_helper.project import Project
from phd_helper.server.app import Session, create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_app_shutdown_closes_the_http_client(tmp_path):
    closed = []

    class Http:
        async def aclose(self):
            closed.append(True)

    state = SimpleNamespace(http=Http(), corpus=Corpus(tmp_path / "c"),
                            ingestor=None, ingest_tasks=set())
    with TestClient(create_app(state=state)):
        pass
    assert closed == [True]  # shared httpx client must not leak


class BoomLlm:
    async def chat(self, *args, **kwargs):
        raise RuntimeError("boom")  # anything outside the known envelope


@pytest.mark.anyio
async def test_unexpected_error_becomes_an_error_event_not_silence(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n"
        "\\input{sections/intro}\n\\end{document}\n", encoding="utf-8")
    (tmp_path / "sections" / "intro.tex").write_text("Hi.\n", encoding="utf-8")
    state = SimpleNamespace(project=Project(tmp_path), llm=BoomLlm(),
                            tts=None, config=None, fetch=None,
                            corpus=Corpus(tmp_path / "c"),
                            corpus_store=None, crossref_mailto="",
                            openalex_mailto="")
    session = Session(state, "c1")
    events = []

    async def capture(event):
        events.append(event)

    session.send = capture
    await session.run_turn("hello")  # must not raise out of the task
    assert any(e.get("type") == "error" and e.get("where") == "turn"
               for e in events)
