"""Lifecycle helper for the GROBID Docker container.

Wraps `docker run`/`docker stop`/`docker inspect` so callers can spin up a
local GROBID instance without writing shell. Defaults to the CRF-only
("lightweight") image, which is the sensible choice on Apple Silicon CPU;
the full deep-learning image is available behind a flag, with optional
`--gpus all` for hosts that have one.

Public surface:
    GrobidVariant   - enum: CRF (default), FULL
    GrobidDocker    - container manager; usable as a context manager
    GrobidError     - raised on docker-CLI failures
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class GrobidVariant(str, Enum):
    CRF = "crf"
    FULL = "full"


# Pinned tags for reproducibility. lfoppiano/grobid is the canonical image.
IMAGE_CRF = "grobid/grobid:0.9.0-crf"
IMAGE_FULL = "grobid/grobid:0.9.0"


class GrobidError(Exception):
    """Raised on docker-CLI failures or readiness timeouts."""


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


class GrobidDocker:
    """Manage a single GROBID container by name.

    Idempotent: starting when already running is a no-op; stopping when
    not running swallows the error. Use as a context manager to bracket a
    batch ingestion run.
    """

    def __init__(
        self,
        variant: GrobidVariant = GrobidVariant.CRF,
        gpu: bool = False,
        port: int = 8070,
        container_name: str = "rag_grobid",
    ):
        if gpu and variant == GrobidVariant.CRF:
            raise GrobidError("CRF image does not use GPU; pick variant=FULL or gpu=False")
        self.variant = variant
        self.gpu = gpu
        self.port = port
        self.container_name = container_name
        self.already_running = False

    @property
    def image(self) -> str:
        return IMAGE_CRF if self.variant == GrobidVariant.CRF else IMAGE_FULL

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def is_running(self) -> bool:
        if shutil.which("docker") is None:
            raise GrobidError("docker CLI not found on PATH")
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False
        return proc.stdout.strip() == "true"

    def start(self, wait: bool = True, timeout: float = 120.0) -> None:
        if shutil.which("docker") is None:
            raise GrobidError("docker CLI not found on PATH")

        if self.is_running():
            logger.info("GROBID container %r already running", self.container_name)
            self.already_running = True
            if wait:
                self.wait_until_ready(timeout)
            return

        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self.container_name,
            "-p", f"{self.port}:8070",
        ]
        if self.gpu:
            cmd += ["--gpus", "all"]
        cmd.append(self.image)

        try:
            _run(cmd)
        except subprocess.CalledProcessError as e:
            raise GrobidError(
                f"docker run failed (exit {e.returncode}): {e.stderr.strip()}"
            ) from e

        if wait:
            self.wait_until_ready(timeout)

    def stop(self) -> None:
        if shutil.which("docker") is None or self.already_running:
            return
        proc = subprocess.run(
            ["docker", "stop", self.container_name],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip().lower()
            if "no such container" in stderr or "is not running" in stderr:
                return
            logger.warning("docker stop returned %d: %s", proc.returncode, proc.stderr.strip())

    def wait_until_ready(self, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{self.url}/api/isalive", timeout=2.0)
                if r.status_code == 200 and r.text.strip().lower() == "true":
                    logger.info("GROBID ready at %s", self.url)
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(1.0)
        raise GrobidError(
            f"GROBID did not become ready at {self.url} within {timeout:.0f}s"
            + (f" (last error: {last_err!r})" if last_err else "")
        )

    def __enter__(self) -> "GrobidDocker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
