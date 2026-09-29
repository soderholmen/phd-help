"""Real HTTP adapter (SPEC §6): the thin production half of the cascade.

Seam: HttpFetcher with an injected httpx-like client. Network failures must
come back as failed Responses (cascade misses) so the turn bounces with a
reason — never exceptions that kill it silently (SPEC §8 error matrix).
"""

import httpx
import pytest

from phd_helper.server.http import HttpFetcher


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FailingClient:
    def __init__(self, exc):
        self._exc = exc

    async def get(self, url, headers=None):
        raise self._exc

    async def aclose(self):
        pass


@pytest.mark.anyio
@pytest.mark.parametrize("exc", [httpx.ConnectError("no route"),
                                 httpx.ReadTimeout("slow")])
async def test_network_failure_bounces_as_a_failed_response(exc):
    fetcher = HttpFetcher(http=FailingClient(exc))
    resp = await fetcher("https://arxiv.org/bibtex/1706.03762")
    assert resp.status == 0  # cascade treats it as a miss
    assert type(exc).__name__ in resp.body  # the reason survives for the bounce
