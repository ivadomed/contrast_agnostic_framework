#!/usr/bin/env python3
"""
Run this ON VULCAN, interactively, whenever the persistent SSH control
socket to killarney has expired (ControlPersist is 30 days of inactivity;
run this once a month or whenever refresh_all.sh logs a sync failure).

Stdlib-only (no pexpect / no third-party dependency): it just execs `ssh`
with your terminal's stdin/stdout/stderr inherited directly, so Alliance
Canada's Duo menu ("1. Duo Push to ...") shows up exactly as it would if
you'd typed the ssh command yourself. Select "1" and approve the push on
your phone as usual. If the control socket is already alive, ssh will
return immediately with nothing to approve.

IMPORTANT: `vulcan.alliancecan.ca` is round-robin DNS over two login nodes
(vulcan1 / vulcan2) that do NOT share kernel-level state. The dashboard's
refresh cron job runs only on vulcan1, so the persistent control socket to
killarney has to be opened there too -- a socket opened on vulcan2 shows up
in the shared NFS home directory and looks valid, but vulcan1 gets a plain
"Connection refused" trying to use it (verified: sockets are visible
cross-node over NFS but not connectable cross-node -- the listening
process lives in one node's kernel only). If you're not already on
vulcan1, this script re-execs itself there over `ssh -t vulcan1` before
doing anything else, so it's safe to run from either login node.

Usage: python3 refresh_killarney_socket.py
"""
import socket
import subprocess
import sys
import os

REMOTE = "killarney.alliancecan.ca"
REQUIRED_HOST = "vulcan1"


def _short_hostname():
    return socket.gethostname().split(".")[0]


def main():
    hostname = _short_hostname()
    if hostname != REQUIRED_HOST:
        print(f"Currently on '{hostname}'. The killarney control socket must be opened on "
              f"'{REQUIRED_HOST}' specifically -- that's the only node where the dashboard's "
              f"refresh cron runs and can reuse it (see module docstring). Re-launching this "
              f"script there over 'ssh -t {REQUIRED_HOST}'...\n", flush=True)
        script_path = os.path.abspath(__file__)
        result = subprocess.run(["ssh", "-t", REQUIRED_HOST, "python3", script_path])
        sys.exit(result.returncode)

    print(f"Connecting to {REMOTE} to (re)open the persistent SSH control socket...", flush=True)
    print("If prompted with a Duo menu, type the number for 'Duo Push' and press "
          "Enter, then tap Approve on your phone.", flush=True)
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=15", REMOTE, "true"])
    if result.returncode == 0:
        print("Success. Persistent control socket to killarney should now be open for ~30 days, "
              f"on {REQUIRED_HOST} where the refresh cron can use it.", flush=True)
    else:
        print(f"ssh exited with code {result.returncode}. Socket not confirmed open -- "
              f"try again, or run manually: ssh {REMOTE} true", flush=True)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
