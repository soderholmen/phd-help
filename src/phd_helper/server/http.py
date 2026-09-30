"""Real HTTP adapter for the BibTeX cascade (SPEC §6).

The cascade is tested against a fake fetcher; this is the thin production
half — one shared client, redirects followed (Crossref/DOI need it), a
generous-but-bounded timeout. Politeness (arXiv 3 s) is applied by the
caller's ArxivRateLimited wrapper, not here.
"""

import httpx

from phd_helper.cascade import Response


class HttpFetcher:
    def __init__(self, http: httpx.AsyncClient | None = None):
        self._http = http or httpx.AsyncClient(
            follow_redirects=True, timeout=httpx.Timeout(20.0, connect=10.0),
            headers={"Accept": "application/x-bibtex, application/json, */*"})

    async def _get(self, url: str, headers: dict | None):
        """One GET, one failure envelope: a dead connection is status 0
        with the reason, never an exception — the turn bounces with a
        reason instead of dying silently (SPEC §8 error matrix)."""
        try:
            return await self._http.get(url, headers=headers or {}), None
        except httpx.HTTPError as e:
            return None, f"{type(e).__name__}: {e}"

    async def __call__(self, url: str, headers: dict | None = None) -> Response:
        r, err = await self._get(url, headers)
        return Response(0, err) if r is None else Response(r.status_code,
                                                           r.text)

    async def fetch_bytes(self, url: str,
                          headers: dict | None = None) -> tuple[int, bytes]:
        """Binary GET for corpus PDFs — r.text would corrupt them."""
        r, err = await self._get(url, headers)
        return (0, err.encode()) if r is None else (r.status_code,
                                                    r.content)

    async def aclose(self):
        await self._http.aclose()
