"""Shared test utilities for all dataset tests."""
import subprocess
from pathlib import Path


def source_env(env_sh: Path, var: str) -> str:
    """Source a dataset env.sh and return the resolved value of one variable."""
    r = subprocess.run(
        ["bash", "-c", f'source "{env_sh}" && printf "%s" "${{{var}}}"'],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def env_vars(env_sh: Path) -> dict[str, str]:
    """Source env.sh and return all exported variables as a dict."""
    r = subprocess.run(
        ["bash", "-c", f'source "{env_sh}" && env'],
        capture_output=True, text=True, check=True,
    )
    result = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k] = v
    return result
