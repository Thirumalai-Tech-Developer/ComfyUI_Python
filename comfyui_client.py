"""
comfyui_client.py

A small, dependency-light client for driving a ComfyUI server headlessly:
  - submit a prompt (API-format workflow dict)
  - wait for it to finish via the websocket progress stream
  - pull the resulting files (video/image/audio) out of the history
  - upload a local image so it can be used as a LoadImage input

Only needs `requests` and the stdlib `websocket-client` package
(`pip install requests websocket-client`).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

try:
    import websocket  # from the `websocket-client` package
except ImportError:  # pragma: no cover
    websocket = None


class ComfyUIError(RuntimeError):
    pass


class ComfyUIClient:
    def __init__(self, server: str = "http://127.0.0.1:8188", client_id: Optional[str] = None,
                 timeout: float = 60.0):
        self.server = server.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self.timeout = timeout

    # -- basic HTTP helpers -------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.server}{path}"

    def get_object_info(self) -> Dict[str, Any]:
        r = requests.get(self._url("/object_info"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def upload_image(self, filepath: str, subfolder: str = "", overwrite: bool = True) -> Dict[str, Any]:
        """
        Uploads a local image to ComfyUI's input directory so it can be
        referenced by name in a LoadImage node. Returns the JSON response,
        e.g. {"name": "myimage.png", "subfolder": "", "type": "input"}.
        """
        with open(filepath, "rb") as f:
            files = {"image": (os.path.basename(filepath), f)}
            data = {"overwrite": "true" if overwrite else "false"}
            if subfolder:
                data["subfolder"] = subfolder
            r = requests.post(self._url("/upload/image"), files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def queue_prompt(self, api_prompt: Dict[str, Any]) -> str:
        """Submit an API-format prompt dict. Returns the prompt_id."""
        payload = {"prompt": api_prompt, "client_id": self.client_id}
        r = requests.post(self._url("/prompt"), json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise ComfyUIError(f"/prompt failed ({r.status_code}): {r.text}")
        data = r.json()
        if "error" in data:
            raise ComfyUIError(f"ComfyUI rejected the prompt: {json.dumps(data['error'], indent=2)}")
        return data["prompt_id"]

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        r = requests.get(self._url(f"/history/{prompt_id}"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_queue(self) -> Dict[str, Any]:
        r = requests.get(self._url("/queue"), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def download_file(self, filename: str, subfolder: str, folder_type: str, out_path: str) -> str:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(self._url("/view"), params=params, timeout=self.timeout, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return out_path

    # -- run-and-wait ---------------------------------------------------

    def run(
        self,
        api_prompt: Dict[str, Any],
        on_progress=None,
        poll_interval: float = 1.0,
        ws_timeout: float = 3600.0,
    ) -> Dict[str, Any]:
        """
        Submit a prompt and block until it finishes (success or error).
        Uses the websocket stream if `websocket-client` is installed and
        reachable; otherwise falls back to polling /history.

        Returns the history entry dict:
            {"prompt": [...], "outputs": {node_id: {...}}, "status": {...}}
        """
        prompt_id = self.queue_prompt(api_prompt)

        if websocket is not None:
            try:
                self._wait_via_websocket(prompt_id, on_progress=on_progress, ws_timeout=ws_timeout)
            except Exception as e:
                if on_progress:
                    on_progress({"type": "websocket_fallback", "error": str(e)})
                self._wait_via_polling(prompt_id, on_progress=on_progress,
                                        poll_interval=poll_interval, timeout=ws_timeout)
        else:
            self._wait_via_polling(prompt_id, on_progress=on_progress,
                                    poll_interval=poll_interval, timeout=ws_timeout)

        history = self.get_history(prompt_id).get(prompt_id)
        if history is None:
            raise ComfyUIError(f"No history found for prompt_id={prompt_id} after completion")

        status = history.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyUIError(f"Prompt {prompt_id} failed: {json.dumps(status, indent=2)}")

        return history

    def _wait_via_websocket(self, prompt_id: str, on_progress=None, ws_timeout: float = 3600.0):
        ws_url = self.server.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
        ws = websocket.create_connection(ws_url, timeout=ws_timeout)
        try:
            start = time.time()
            while True:
                if time.time() - start > ws_timeout:
                    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")
                raw = ws.recv()
                if isinstance(raw, bytes):
                    continue  # binary preview frames, ignore
                msg = json.loads(raw)
                mtype = msg.get("type")
                data = msg.get("data", {})

                if on_progress:
                    on_progress(msg)

                if mtype == "executing" and data.get("prompt_id") == prompt_id and data.get("node") is None:
                    # node == None signals this prompt finished executing
                    return
                if mtype == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise ComfyUIError(f"Execution error: {json.dumps(data, indent=2)}")
        finally:
            ws.close()

    def _wait_via_polling(self, prompt_id: str, on_progress=None, poll_interval: float = 1.0,
                           timeout: float = 3600.0):
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")
            hist = self.get_history(prompt_id)
            if prompt_id in hist:
                if on_progress:
                    on_progress({"type": "poll", "data": {"prompt_id": prompt_id, "done": True}})
                return
            if on_progress:
                on_progress({"type": "poll", "data": {"prompt_id": prompt_id, "done": False}})
            time.sleep(poll_interval)

    # -- output extraction ---------------------------------------------

    @staticmethod
    def list_output_files(history_entry: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Flattens every SaveImage/SaveVideo/SaveAudio-style output referenced
        in a history entry into a list of
            {"filename":..., "subfolder":..., "type":..., "node_id":..., "kind":...}
        `kind` is one of "images", "videos", "gifs", "audio" (whatever key
        ComfyUI used in that node's output dict).
        """
        results = []
        for node_id, node_output in history_entry.get("outputs", {}).items():
            for kind, files in node_output.items():
                if not isinstance(files, list):
                    continue
                for f in files:
                    if not isinstance(f, dict) or "filename" not in f:
                        continue
                    results.append({
                        "filename": f["filename"],
                        "subfolder": f.get("subfolder", ""),
                        "type": f.get("type", "output"),
                        "node_id": node_id,
                        "kind": kind,
                    })
        return results

    def download_all_outputs(self, history_entry: Dict[str, Any], out_dir: str) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        saved = []
        for f in self.list_output_files(history_entry):
            out_path = os.path.join(out_dir, f["filename"])
            self.download_file(f["filename"], f["subfolder"], f["type"], out_path)
            saved.append(out_path)
        return saved
