"""Count running python processes whose command line contains every substring.

Used by the duplicate-launch guards. Lives in a file because the equivalent
inline PowerShell one-liner needed four levels of nested quoting inside a shell
script, and the quoting broke in a way that silently disabled the *next* check
in the same script rather than failing loudly.

Prints a single integer. Exit code is always 0 -- the caller decides what a
non-zero count means.

    python scripts/count_running.py train_prior oakink2
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    needles = [a.lower() for a in sys.argv[1:]]
    if not needles:
        print(0)
        return 0

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:  # noqa: BLE001 - a guard that crashes must not block the run
        print(0)
        return 0

    me = str(os.getpid())
    n = 0
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid == me or not cmd:
            continue
        low = cmd.lower()
        # Skip every copy of this script, not just this pid. The needles are on
        # our own command line by construction, and each python here is paired
        # with a uv shim process carrying identical arguments under a different
        # pid -- so self-matching inflated the count to 1 with nothing running
        # and would have made the guard abort every launch.
        if "count_running.py" in low:
            continue
        if all(needle in low for needle in needles):
            n += 1
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
