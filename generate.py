#!/usr/bin/env python3
"""
generate.py -- headless automation for the Qwen -> LTX-2.3 image+video ComfyUI
workflow (Qwen_LTX2_3.json).

Pipeline this drives:
    Qwen text-to-image  --(image)-->  LTX-2.3 image-to-video (+ foley audio)  --> mp4

WHAT IT DOES
    1. Converts your UI-exported workflow JSON to API format (querying the
       live ComfyUI server's /object_info so it stays correct across custom
       node versions -- see workflow_converter.py).
    2. Patches the prompts / seeds / resolution / frame count / fps you pass
       on the command line (or from a batch jobs file) into the graph.
    3. Optionally swaps in your own starting image instead of generating one
       with Qwen (--input-image), skipping that whole branch.
    4. Submits the job, waits for it to finish, downloads the resulting
       .mp4 (and any other outputs) to --out-dir.

REQUIREMENTS
    pip install requests websocket-client

QUICK START
    # single video, generating the start frame with Qwen from a text prompt
    python generate.py \\
        --server https://your-tunnel-or-host:8188 \\
        --image-prompt "a cozy reading nook, warm lamp light, rain on the window" \\
        --video-prompt "slow push in, curtains sway gently, steam rises from a mug" \\
        --filename-prefix "cozy_nook"

    # image-to-video only, using your own starting frame
    python generate.py --server http://127.0.0.1:8188 \\
        --input-image ./my_photo.png \\
        --video-prompt "camera slowly orbits left, hair moves in the breeze"

    # batch mode: many jobs from one file
    python generate.py --server http://127.0.0.1:8188 --jobs jobs.json

JOBS FILE FORMAT (jobs.json)
    [
      {
        "name": "shot01",
        "image_prompt": "...",
        "image_negative_prompt": "...",
        "video_prompt": "...",
        "video_negative_prompt": "...",
        "input_image": "optional/path/to/local/image.png",
        "width": 512, "height": 512, "frames": 49, "fps": 24,
        "image_seed": 12345, "video_seed": 67890
      },
      { "...": "next job" }
    ]
    Any field you omit falls back to the matching --image-prompt/--width/etc.
    CLI default for that run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from comfyui_client import ComfyUIClient, ComfyUIError
from workflow_converter import convert_workflow, fetch_object_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Graph role discovery -- find the "meaningful" nodes generically by class
# type rather than hardcoded ids, so this keeps working even if you re-save
# the workflow and node ids shift around.
# ---------------------------------------------------------------------------

def first_of_type(api_prompt: Dict[str, Any], class_type: str) -> Optional[str]:
    for node_id, node in api_prompt.items():
        if node["class_type"] == class_type:
            return node_id
    return None


def all_of_type(api_prompt: Dict[str, Any], class_type: str) -> List[str]:
    return [nid for nid, n in api_prompt.items() if n["class_type"] == class_type]


def discover_roles(api_prompt: Dict[str, Any]) -> Dict[str, Optional[str]]:
    roles: Dict[str, Optional[str]] = {}

    qwen_ksampler = first_of_type(api_prompt, "KSampler")
    roles["qwen_ksampler"] = qwen_ksampler
    if qwen_ksampler:
        ki = api_prompt[qwen_ksampler]["inputs"]
        roles["qwen_positive"] = ki.get("positive", [None])[0]
        roles["qwen_negative"] = ki.get("negative", [None])[0]
        roles["qwen_latent"] = ki.get("latent_image", [None])[0]
    else:
        roles["qwen_positive"] = roles["qwen_negative"] = roles["qwen_latent"] = None

    ltxv_node = first_of_type(api_prompt, "LTXVImgToVideo")
    roles["ltxv_node"] = ltxv_node
    if ltxv_node:
        li = api_prompt[ltxv_node]["inputs"]
        roles["video_positive"] = li.get("positive", [None])[0]
        roles["video_negative"] = li.get("negative", [None])[0]

    roles["ksampler_advanced"] = first_of_type(api_prompt, "KSamplerAdvanced")
    roles["audio_latent"] = first_of_type(api_prompt, "LTXVEmptyLatentAudio")
    roles["create_video"] = first_of_type(api_prompt, "CreateVideo")
    roles["save_video"] = first_of_type(api_prompt, "SaveVideo")
    roles["preview_images"] = all_of_type(api_prompt, "PreviewImage")
    return roles


# ---------------------------------------------------------------------------
# Job application
# ---------------------------------------------------------------------------

def _coerce(value: str):
    """Best-effort string -> python type coercion for --override values."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(api_prompt: Dict[str, Any], overrides: List[str]) -> None:
    """--override 'CreateVideo.fps=24' or --override '503.fps=24'"""
    for item in overrides:
        if "=" not in item or "." not in item.split("=")[0]:
            print(f"[warn] ignoring malformed --override '{item}' (expected KEY.input=value)")
            continue
        key, value = item.split("=", 1)
        target, input_name = key.rsplit(".", 1)
        value = _coerce(value)

        node_ids = [target] if target in api_prompt else all_of_type(api_prompt, target)
        if not node_ids:
            print(f"[warn] --override target '{target}' not found in graph; skipping")
            continue
        for nid in node_ids:
            api_prompt[nid]["inputs"][input_name] = value


def apply_job(base_api_prompt: Dict[str, Any], roles: Dict[str, Any], job: Dict[str, Any],
              client: Optional[ComfyUIClient]) -> Dict[str, Any]:
    """Return a patched deep copy of base_api_prompt for one job's parameters."""
    api_prompt = copy.deepcopy(base_api_prompt)

    def set_input(node_id, name, value):
        if node_id and node_id in api_prompt and value is not None:
            api_prompt[node_id]["inputs"][name] = value

    input_image = job.get("input_image")
    if input_image:
        if client is None:
            raise RuntimeError("input_image was given but no ComfyUIClient was provided to upload it")
        uploaded = client.upload_image(input_image)
        image_ref = uploaded["name"]
        subfolder = uploaded.get("subfolder", "")
        if subfolder:
            image_ref = f"{subfolder}/{image_ref}"

        load_image_id = "job_load_image"
        api_prompt[load_image_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_ref},
            "_meta": {"title": "LoadImage (job input)"},
        }
        if roles.get("ltxv_node"):
            set_input(roles["ltxv_node"], "image", [load_image_id, 0])

        # No need to run the Qwen image-generation branch at all -- drop any
        # PreviewImage output nodes so ComfyUI doesn't bother executing it.
        for pid in roles.get("preview_images", []):
            api_prompt.pop(pid, None)

    set_input(roles.get("qwen_positive"), "text", job.get("image_prompt"))
    set_input(roles.get("qwen_negative"), "text", job.get("image_negative_prompt"))
    set_input(roles.get("video_positive"), "text", job.get("video_prompt"))
    set_input(roles.get("video_negative"), "text", job.get("video_negative_prompt"))

    width = job.get("width")
    height = job.get("height")
    if roles.get("qwen_latent"):
        set_input(roles["qwen_latent"], "width", width)
        set_input(roles["qwen_latent"], "height", height)
    if roles.get("ltxv_node"):
        set_input(roles["ltxv_node"], "width", width)
        set_input(roles["ltxv_node"], "height", height)

    frames = job.get("frames")
    if roles.get("ltxv_node"):
        set_input(roles["ltxv_node"], "length", frames)
    if roles.get("audio_latent"):
        set_input(roles["audio_latent"], "frames_number", frames)

    if roles.get("qwen_ksampler"):
        set_input(roles["qwen_ksampler"], "seed", job.get("image_seed"))
        set_input(roles["qwen_ksampler"], "steps", job.get("image_steps"))
        set_input(roles["qwen_ksampler"], "cfg", job.get("image_cfg"))

    if roles.get("ksampler_advanced"):
        set_input(roles["ksampler_advanced"], "seed", job.get("video_seed"))
        set_input(roles["ksampler_advanced"], "steps", job.get("video_steps"))
        set_input(roles["ksampler_advanced"], "cfg", job.get("video_cfg"))

    if roles.get("create_video"):
        set_input(roles["create_video"], "fps", job.get("fps"))

    if roles.get("save_video"):
        prefix = job.get("filename_prefix") or job.get("name")
        set_input(roles["save_video"], "filename_prefix", prefix)

    apply_overrides(api_prompt, job.get("overrides", []))
    return api_prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8188",
                     help="ComfyUI server base URL (local or your cloudflared tunnel URL)")
    ap.add_argument("--workflow", default=os.path.join(SCRIPT_DIR, "Qwen_LTX2_3.json"),
                     help="Path to the UI-format workflow JSON exported from ComfyUI")
    ap.add_argument("--api-workflow", default=None,
                     help="Skip conversion and load an already-converted API-format JSON instead")
    ap.add_argument("--save-api-workflow", default=None,
                     help="Write the converted API-format JSON to this path for inspection/reuse")
    ap.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "outputs"))

    # single-job convenience args
    ap.add_argument("--image-prompt", default=None, help="Positive prompt for the Qwen start-frame image")
    ap.add_argument("--image-negative-prompt", default=None)
    ap.add_argument("--video-prompt", default=None, help="Positive/motion prompt for the LTX video")
    ap.add_argument("--video-negative-prompt", default=None)
    ap.add_argument("--input-image", default=None,
                     help="Local image path to use as the start frame instead of generating one with Qwen")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None, help="Video length in frames")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--image-seed", type=int, default=None)
    ap.add_argument("--video-seed", type=int, default=None)
    ap.add_argument("--image-steps", type=int, default=None)
    ap.add_argument("--image-cfg", type=float, default=None)
    ap.add_argument("--video-steps", type=int, default=None)
    ap.add_argument("--video-cfg", type=float, default=None)
    ap.add_argument("--filename-prefix", default=None)
    ap.add_argument("--override", action="append", default=[],
                     help="Generic patch: 'ClassType.input=value' or 'nodeId.input=value'. Repeatable.")

    # batch mode
    ap.add_argument("--jobs", default=None, help="Path to a JSON list of job dicts for batch processing")

    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def load_or_convert_workflow(args) -> Dict[str, Any]:
    if args.api_workflow:
        with open(args.api_workflow, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(args.workflow, "r", encoding="utf-8") as f:
        ui_workflow = json.load(f)

    print(f"Fetching node schemas from {args.server}/object_info ...")
    object_info = fetch_object_info(args.server)
    api_prompt, _meta = convert_workflow(ui_workflow, object_info, verbose=args.verbose)

    if args.save_api_workflow:
        with open(args.save_api_workflow, "w", encoding="utf-8") as f:
            json.dump(api_prompt, f, indent=2)
        print(f"Saved converted workflow to {args.save_api_workflow}")

    return api_prompt


def make_progress_printer(job_name: str):
    last_node = {"value": None}

    def on_progress(msg):
        mtype = msg.get("type")
        data = msg.get("data", {})
        if mtype == "progress":
            v, m = data.get("value"), data.get("max")
            print(f"\r[{job_name}] progress {v}/{m}", end="", flush=True)
        elif mtype == "executing":
            node = data.get("node")
            if node != last_node["value"]:
                last_node["value"] = node
                if node is not None:
                    print(f"\n[{job_name}] running node {node} ...", end="", flush=True)
        elif mtype == "websocket_fallback":
            print(f"\n[{job_name}] websocket unavailable ({data.get('error') if isinstance(data, dict) else msg.get('error')}), polling instead")

    return on_progress


def run_job(client: ComfyUIClient, api_prompt: Dict[str, Any], job_name: str, out_dir: str) -> List[str]:
    print(f"\n=== Submitting job: {job_name} ===")
    history = client.run(api_prompt, on_progress=make_progress_printer(job_name))
    print()
    files = client.download_all_outputs(history, out_dir)
    for f in files:
        print(f"[{job_name}] saved -> {f}")
    if not files:
        print(f"[{job_name}] finished but no output files were found in history.outputs")
    return files


def main():
    args = build_arg_parser().parse_args()

    base_api_prompt = load_or_convert_workflow(args)
    roles = discover_roles(base_api_prompt)
    if args.verbose:
        print("Discovered graph roles:", json.dumps(roles, indent=2))

    client = ComfyUIClient(server=args.server)

    cli_job_defaults = {
        "name": args.filename_prefix or "video",
        "image_prompt": args.image_prompt,
        "image_negative_prompt": args.image_negative_prompt,
        "video_prompt": args.video_prompt,
        "video_negative_prompt": args.video_negative_prompt,
        "input_image": args.input_image,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "image_seed": args.image_seed,
        "video_seed": args.video_seed,
        "image_steps": args.image_steps,
        "image_cfg": args.image_cfg,
        "video_steps": args.video_steps,
        "video_cfg": args.video_cfg,
        "filename_prefix": args.filename_prefix,
        "overrides": args.override,
    }

    if args.jobs:
        with open(args.jobs, "r", encoding="utf-8") as f:
            jobs = json.load(f)
        if not isinstance(jobs, list):
            print("ERROR: --jobs file must contain a JSON list of job objects", file=sys.stderr)
            sys.exit(1)
    else:
        jobs = [{}]  # single job using pure CLI defaults

    exit_code = 0
    for i, job in enumerate(jobs):
        merged = dict(cli_job_defaults)
        merged.update({k: v for k, v in job.items() if v is not None})
        job_name = merged.get("name") or f"job{i+1}"

        try:
            api_prompt = apply_job(base_api_prompt, roles, merged, client)
            run_job(client, api_prompt, job_name, args.out_dir)
        except ComfyUIError as e:
            print(f"\n[{job_name}] FAILED: {e}", file=sys.stderr)
            exit_code = 1
        except Exception as e:
            print(f"\n[{job_name}] FAILED (unexpected error): {e}", file=sys.stderr)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
