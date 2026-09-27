"""
Shared helper for reading TotalSegmentator's Zenodo zips either from a local file or
directly over HTTP via range requests (same technique used during this dataset's
pre-flight checklist — reads only the zip's central directory + the specific entries
asked for, not the whole archive). Used by both 01_create_splits/01_01_create_splits.py
(case-id listing) and 02_nnunet/02_00_convert.py (per-case image/label extraction), so
CT's 200-case subsample never needs the full 1228-case/23GB archive on disk.

CT_ZENODO_URL / MRI_ZENODO_URL are the same URLs verified during pre-flight (Zenodo API
`.../files/<name>/content` links, confirmed to return the exact claimed byte size via
HEAD).
"""
from __future__ import annotations

import http.client
import io
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

CT_ZENODO_URL = "https://zenodo.org/api/records/10047292/files/Totalsegmentator_dataset_v201.zip/content"
MRI_ZENODO_URL = "https://zenodo.org/api/records/11367005/files/TotalsegmentatorMRI_dataset_v100.zip/content"

# Rate-limit handling: Zenodo returned 503 after a burst of small range requests (each
# case needs ~11 — 1 image + 10 label masks — across up to 498 cases), and separately a
# transient 500 mid-run (2026-09-15, case 34/200 of the CT pass) that the original
# 503/429-only retry set did not cover and which killed the process outright. Retry on
# any 5xx (transient server-side) plus 429, and pace consecutive requests with a small
# fixed delay so we don't just immediately re-trigger the same limit. Also retry on
# URLError (connection reset/timeout) — HTTPError alone doesn't cover those.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 8
_BACKOFF_START_S = 2.0
_BACKOFF_CAP_S = 60.0
_INTER_REQUEST_DELAY_S = 0.3


class HTTPRangeFile(io.IOBase):
    """Minimal seekable/readable file-like object over HTTP Range requests — enough for
    zipfile.ZipFile to read the central directory and individual entries without ever
    downloading the whole remote file. Retries on 503/429 with exponential backoff and
    paces requests to avoid re-triggering the same rate limit.

    Uses a single persistent HTTPSConnection (keep-alive) across all range reads instead
    of a fresh urllib.request.urlopen() call per read — each case needs ~11 range
    requests, and re-doing a full TLS handshake for every single one of them (the
    original implementation) turned out to be the dominant per-case cost during the BIDS
    rework's CT fetch (2026-09-15): ~90-110s/case observed, vs. ~13-15s/case in the
    original merge-in-memory pass making the exact same 11 requests per case. Falls back
    to a fresh connection on any failure (a dropped keep-alive connection is a normal,
    expected occurrence, not an error worth failing the whole read over)."""

    def __init__(self, url: str):
        self.url = url
        parsed = urllib.parse.urlsplit(url)
        self._host = parsed.netloc
        self._path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
        self._conn: http.client.HTTPSConnection | None = None
        r = self._request("HEAD")
        self.size = int(r.getheader("Content-Length"))
        r.read()
        self.pos = 0
        self._last_request_t = 0.0

    def _connect(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(self._host, timeout=120)
        return self._conn

    def _request(self, method: str, headers: dict | None = None):
        """Issue one request over the persistent connection, with retry/backoff on
        transient errors (5xx/429) and automatic reconnect on a broken keep-alive
        connection (BrokenPipeError, ConnectionResetError, http.client exceptions).

        A User-Agent header is required — Zenodo's nginx front-end returns a bare 403
        for requests without one (urllib.request.urlopen sets a default UA
        automatically, which is why the original per-request-urlopen implementation
        never hit this; a raw http.client.HTTPSConnection sends none unless told to)."""
        delay = _BACKOFF_START_S
        last_exc = None
        req_headers = {"User-Agent": "totalseg-pelvic-fetch/1.0 (mri_synthesis_project)"}
        req_headers.update(headers or {})
        for attempt in range(_MAX_RETRIES + 1):
            try:
                conn = self._connect()
                conn.request(method, self._path, headers=req_headers)
                resp = conn.getresponse()
                if resp.status in _RETRY_STATUS:
                    body = resp.read()  # drain so the connection can be reused
                    last_exc = urllib.error.HTTPError(self.url, resp.status, resp.reason, resp.headers, None)
                    if attempt == _MAX_RETRIES:
                        raise last_exc
                    print(f"[zenodo_zip] HTTP {resp.status} on attempt {attempt + 1}/{_MAX_RETRIES + 1} "
                          f"— retrying in {delay:.1f}s ({self.url})")
                    time.sleep(delay)
                    delay = min(delay * 2, _BACKOFF_CAP_S)
                    continue
                if resp.status >= 400:
                    raise urllib.error.HTTPError(self.url, resp.status, resp.reason, resp.headers, None)
                return resp
            except (urllib.error.HTTPError,) as e:
                if e.code not in _RETRY_STATUS or attempt == _MAX_RETRIES:
                    raise
                last_exc = e
                print(f"[zenodo_zip] HTTP {e.code} on attempt {attempt + 1}/{_MAX_RETRIES + 1} "
                      f"— retrying in {delay:.1f}s ({self.url})")
                time.sleep(delay)
                delay = min(delay * 2, _BACKOFF_CAP_S)
            except (http.client.HTTPException, OSError, ConnectionError, TimeoutError) as e:
                # Broken keep-alive connection (server closed idle conn, etc.) or a
                # transient network error — drop and reconnect fresh next attempt.
                last_exc = e
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                self._conn = None
                if attempt == _MAX_RETRIES:
                    raise
                print(f"[zenodo_zip] {type(e).__name__} on attempt {attempt + 1}/{_MAX_RETRIES + 1} "
                      f"— reconnecting, retrying in {delay:.1f}s ({self.url}): {e}")
                time.sleep(delay)
                delay = min(delay * 2, _BACKOFF_CAP_S)
        raise last_exc if last_exc is not None else AssertionError("unreachable")

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if n is None or n < 0 else min(self.pos + n - 1, self.size - 1)
        if end < self.pos:
            return b""
        # Pace requests — a fixed minimum gap between consecutive range requests, not just
        # reactive backoff after already getting rate-limited.
        since_last = time.monotonic() - self._last_request_t
        if since_last < _INTER_REQUEST_DELAY_S:
            time.sleep(_INTER_REQUEST_DELAY_S - since_last)
        resp = self._request("GET", headers={"Range": f"bytes={self.pos}-{end}"})
        data = resp.read()
        self._last_request_t = time.monotonic()
        self.pos += len(data)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        super().close()


def open_zip(local_path: Path, url: str) -> zipfile.ZipFile:
    """Open `local_path` if it exists (fast, no network per-read), else stream `url` over
    HTTP range requests (slower per-entry, but avoids downloading the whole archive when
    only a subset of entries is ever needed — e.g. CT's 200/1228-case subsample)."""
    if local_path.exists():
        return zipfile.ZipFile(local_path)
    return zipfile.ZipFile(HTTPRangeFile(url))
