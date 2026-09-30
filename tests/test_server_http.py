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


class BinaryClient:
    def __init__(self, status=200, content=b"%PDF-1.7", exc=None):
        self.status, self.content, self.exc = status, content, exc

    async def get(self, url, headers=None):
        if self.exc:
            raise self.exc

        class R:
            status_code = self.status
            content = self.content
        return R()

    async def aclose(self):
        pass


@pytest.mark.anyio
async def test_fetch_bytes_returns_raw_pdf_bytes():
    fetcher = HttpFetcher(http=BinaryClient())
    status, body = await fetcher.fetch_bytes("https://arxiv.org/pdf/1706.03762")
    assert status == 200 and body == b"%PDF-1.7"  # not text-decoded


@pytest.mark.anyio
async def test_fetch_bytes_failure_is_status_zero_not_an_exception():
    fetcher = HttpFetcher(http=BinaryClient(exc=httpx.ConnectError("down")))
    status, body = await fetcher.fetch_bytes("https://arxiv.org/pdf/x")
    assert status == 0 and b"ConnectError" in body
