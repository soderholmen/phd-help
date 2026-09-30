"""MinerU extractor adapter (SPEC §6).

The heavy ONNX stack lives in the side venv (.venv-mineru), never in the
app's venv; MinerU 4.0 is a client/daemon design, so `mineru server
start` (managed local tier) must be running — ops starts it. A dead
daemon or failed parse comes back as an error envelope and raises, so
the ingest status machine marks the doc failed with the reason visible
(§8), never silently.

The CLI's content format is markdown; the pure parser (mdblocks) maps
it onto Blocks. `run` is the test seam: async (argv) -> (rc, stdout).
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from phd_helper.chunking import Block
from phd_helper.mdblocks import markdown_to_blocks
from phd_helper.server.config import REPO_ROOT

DEFAULT_CLI = os.environ.get(
    "PHD_MINERU_CLI",
    str(REPO_ROOT / ".venv-mineru" / "Scripts" / "mineru.exe"))


class MinerUExtractor:
    def __init__(self, cli: str = DEFAULT_CLI, timeout_s: float = 900.0,
                 run=None):
        self.cli = cli
        self.timeout_s = timeout_s
        self._run = run

    async def _subprocess(self, argv: list[str]) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        out, _err = await asyncio.wait_for(proc.communicate(),
                                           self.timeout_s + 30)
        return proc.returncode, out.decode("utf-8", "replace")

    async def extract(self, pdf: bytes) -> list[Block]:
        fd, path = tempfile.mkstemp(suffix=".pdf", prefix="phd-ingest-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(pdf)
            argv = [self.cli, "parse", path, "--json", "-p", "all",
                    "--wait", str(int(self.timeout_s))]
            rc, out = await (self._run or self._subprocess)(argv)
            try:
                envelope = json.loads(out)
            except json.JSONDecodeError:
                raise RuntimeError(f"mineru CLI exited {rc}: {out[:300]}")
            content = (envelope.get("content") or {}).get("content")
            if not content:
                err = envelope.get("error") or {}
                raise RuntimeError(err.get("message")
                                   or f"mineru returned no content (rc {rc})")
            return markdown_to_blocks(content)
        finally:
            Path(path).unlink(missing_ok=True)
