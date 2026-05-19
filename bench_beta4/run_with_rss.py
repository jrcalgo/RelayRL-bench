#!/usr/bin/env python3
"""Run a command and report peak RSS of the process tree."""
import subprocess
import psutil
import threading
import sys
import os

def run_with_rss(cmd, env=None, cwd=None):
    proc = subprocess.Popen(
        cmd, env=env or os.environ.copy(), cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    p = psutil.Process(proc.pid)
    peak_rss_bytes = [0]

    def monitor():
        while proc.poll() is None:
            try:
                rss = p.memory_info().rss
                for child in p.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                if rss > peak_rss_bytes[0]:
                    peak_rss_bytes[0] = rss
            except psutil.NoSuchProcess:
                break
            import time; time.sleep(0.05)

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    t.join(timeout=1)
    return proc.returncode, peak_rss_bytes[0] / 1024**2

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_with_rss.py <command> [args...]")
        sys.exit(1)
    rc, peak_mb = run_with_rss(sys.argv[1:])
    print(f"\n  peak RSS (tree)   : {peak_mb:.0f} MB")
    sys.exit(rc)
