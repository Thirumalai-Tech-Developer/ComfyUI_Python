# =====================================================================
# ComfyUI Watchdog -- paste this into a Kaggle cell in place of the
# "🌐 ComfyUI on CloudFlare" and "🔄 Loop to Keep active Session" cells.
#
# What it does, continuously, for the rest of the notebook's runtime:
#   1. Runs ComfyUI as a subprocess. If it crashes (OOM, disk-full, any
#      nonzero exit), automatically restarts it.
#   2. Runs `cloudflared tunnel` pointed at it. If the tunnel drops or
#      ComfyUI restarts, restarts the tunnel too, and picks the new public
#      URL out of its logs.
#   3. Publishes the current public URL to a free ntfy.sh topic every time
#      it changes, so your LOCAL machine (running generate.py --watch
#      --server-discovery-ntfy YOUR_TOPIC) can automatically pick up the
#      new address without you copy-pasting it.
#   4. Watches disk usage and deletes old files out of ComfyUI's output/
#      and temp/ folders (oldest first) so a full disk doesn't crash the
#      whole session. Only deletes files older than OUTPUT_RETENTION_MIN,
#      so make sure your local generate.py polls/downloads more often than
#      that.
#
# REQUIRED: set NTFY_TOPIC below to something unguessable (it's a public
# pub/sub system -- anyone who knows the topic name can read your tunnel
# URL). e.g. NTFY_TOPIC = "comfy-tunnel-x7f3q9k2"
# =====================================================================

import os
import re
import shutil
import subprocess
import threading
import time
import requests

# ---------------------------- CONFIG ----------------------------------
PORT = 8188
NTFY_TOPIC = "CHANGE_ME_comfy_tunnel_xxxxxxxx"   # <-- set this to something unique/secret
COMFYUI_DIR = "/root/comfy/ComfyUI"
OUTPUT_DIRS = [os.path.join(COMFYUI_DIR, "output"), os.path.join(COMFYUI_DIR, "temp")]
OUTPUT_RETENTION_MIN = 20     # delete finished outputs older than this (make sure your
                               # local client downloads more often than this window)
MIN_FREE_GB = 8               # emergency cleanup trigger: free space drops below this
DISK_CHECK_INTERVAL = 30      # seconds
PROCESS_CHECK_INTERVAL = 3    # seconds
HEARTBEAT_INTERVAL = 60       # seconds
# ------------------------------------------------------------------------

_state = {
    "comfy_proc": None,
    "tunnel_proc": None,
    "current_url": None,
}
_lock = threading.Lock()


def _stream_output(proc, prefix):
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end="")


def supervise_comfyui():
    while True:
        proc = _state.get("comfy_proc")
        if proc is None or proc.poll() is not None:
            if proc is not None:
                print(f"\n[watchdog] ComfyUI exited (code={proc.poll()}). Restarting...")
            os.chdir(COMFYUI_DIR)
            new_proc = subprocess.Popen(
                ["python", "main.py", "--enable-manager", "--listen", "0.0.0.0",
                 "--port", str(PORT), "--enable-cors-header", "*"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            threading.Thread(target=_stream_output, args=(new_proc, "comfyui"), daemon=True).start()
            with _lock:
                _state["comfy_proc"] = new_proc
            print(f"[watchdog] ComfyUI (re)started, pid={new_proc.pid}")
        time.sleep(PROCESS_CHECK_INTERVAL)


def _wait_for_port(port, timeout=120):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def supervise_tunnel():
    while True:
        proc = _state.get("tunnel_proc")
        needs_restart = proc is None or proc.poll() is not None
        if needs_restart:
            if proc is not None:
                print(f"\n[watchdog] cloudflared exited (code={proc.poll()}). Restarting...")
            if not _wait_for_port(PORT):
                print("[watchdog] ComfyUI port not up yet, waiting to start tunnel...")
                time.sleep(3)
                continue

            new_proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            with _lock:
                _state["tunnel_proc"] = new_proc
                _state["current_url"] = None  # forget old URL until we see a fresh one

            def watch_for_url(p=new_proc):
                for line in p.stdout:
                    print(f"[cloudflared] {line}", end="")
                    m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
                    if m:
                        url = m.group(0)
                        with _lock:
                            changed = _state.get("current_url") != url
                            _state["current_url"] = url
                        if changed:
                            print(f"\n[watchdog] Public URL: {url}")
                            if NTFY_TOPIC.startswith("CHANGE_ME"):
                                print("[watchdog] WARNING: set NTFY_TOPIC to a real value "
                                      "so your local machine can discover this URL automatically.")
                            else:
                                ok = False
                                try:
                                    r = requests.post(f"https://ntfy.sh/{NTFY_TOPIC}",
                                                       data=url.encode("utf-8"),
                                                       headers={"Title": "ComfyUI URL updated"},
                                                       timeout=10)
                                    ok = r.status_code == 200
                                except requests.RequestException as e:
                                    print(f"[watchdog] ntfy publish failed: {e}")
                                print(f"[watchdog] published to ntfy.sh/{NTFY_TOPIC}: {'ok' if ok else 'FAILED'}")

            threading.Thread(target=watch_for_url, daemon=True).start()
        time.sleep(PROCESS_CHECK_INTERVAL)


def _dir_size_and_free(path):
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)  # GB


def _list_files_oldest_first(dirs):
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, names in os.walk(d):
            for n in names:
                fp = os.path.join(root, n)
                try:
                    files.append((os.path.getmtime(fp), fp))
                except OSError:
                    continue
    files.sort()  # oldest first
    return files


def supervise_disk():
    while True:
        now = time.time()
        files = _list_files_oldest_first(OUTPUT_DIRS)

        # 1) routine cleanup: anything older than the retention window
        for mtime, fp in files:
            if now - mtime > OUTPUT_RETENTION_MIN * 60:
                try:
                    os.remove(fp)
                    print(f"[watchdog] pruned old output (>{OUTPUT_RETENTION_MIN}min): {fp}")
                except OSError:
                    pass

        # 2) emergency cleanup: keep deleting oldest files, regardless of
        #    age, until we're back above MIN_FREE_GB free space
        free_gb = _dir_size_and_free(COMFYUI_DIR)
        if free_gb < MIN_FREE_GB:
            print(f"[watchdog] LOW DISK: {free_gb:.1f}GB free (< {MIN_FREE_GB}GB), emergency pruning...")
            files = _list_files_oldest_first(OUTPUT_DIRS)
            for mtime, fp in files:
                if _dir_size_and_free(COMFYUI_DIR) >= MIN_FREE_GB:
                    break
                try:
                    os.remove(fp)
                    print(f"[watchdog] emergency-pruned: {fp}")
                except OSError:
                    pass

        time.sleep(DISK_CHECK_INTERVAL)


def heartbeat():
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        with _lock:
            url = _state.get("current_url")
            comfy_alive = _state["comfy_proc"] is not None and _state["comfy_proc"].poll() is None
            tunnel_alive = _state["tunnel_proc"] is not None and _state["tunnel_proc"].poll() is None
        free_gb = _dir_size_and_free(COMFYUI_DIR)
        print(f"\n[watchdog] heartbeat: comfyui={'up' if comfy_alive else 'DOWN'} "
              f"tunnel={'up' if tunnel_alive else 'DOWN'} url={url} free_disk={free_gb:.1f}GB")


print("[watchdog] starting supervisors...")
threading.Thread(target=supervise_comfyui, daemon=True).start()
threading.Thread(target=supervise_tunnel, daemon=True).start()
threading.Thread(target=supervise_disk, daemon=True).start()
threading.Thread(target=heartbeat, daemon=True).start()

print("[watchdog] running. This cell will block forever to keep the session alive "
      "(same as the notebook's old 'Loop to Keep active Session' cell) -- stop the "
      "cell manually when you're done.")
while True:
    time.sleep(1)
