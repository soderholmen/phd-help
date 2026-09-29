"""App assembly at the HTTP seam (SPEC §1): startup/shutdown wiring only.

The voice loop itself is proven end-to-end against live vLLM (§2); this
file covers what only the app owns — resource lifecycle.
"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from phd_helper.server.app import create_app


def test_app_shutdown_closes_the_http_client():
    closed = []

    class Http:
        async def aclose(self):
            closed.append(True)

    with TestClient(create_app(state=SimpleNamespace(http=Http()))):
        pass
    assert closed == [True]  # shared httpx client must not leak
