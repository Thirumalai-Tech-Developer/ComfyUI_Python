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

    # multi-scene mode: one JSON describing a whole sequence of shots,
    # duration given in seconds instead of raw frame counts
    python generate.py --server http://127.0.0.1:8188 --scenes scenes.json

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

SCENES FILE FORMAT (scenes.json) -- for multi-shot sequences
    {
      "total_scene": 5,
      "scenes": [
        {
          "id": 1,
          "image_prompt_positive": "...",
          "image_prompt_negative": "...",
          "video_prompt_positive": "...",
          "video_prompt_negative": "...",
          "audio_prompt": "...",
          "seed": 1024,
          "duration": 5
        },
        { "id": 2, "...": "..." }
      ]
    }
    - "duration" is in **seconds** and is converted to a valid LTX frame
      count using --gen-fps (default 24), rounded to the nearest value LTX
      accepts (frame counts of the form 8*n+1).
    - "seed" is used for both the image and video sampler unless you also
      give per-scene "image_seed"/"video_seed" (those take priority).
    - "audio_prompt" is folded into the video's positive prompt as a
      "Sound Design Prompt" section (matching how the original workflow's
      LTX-2.3 foley conditioning expects it) -- see --audio-prompt-template
      to customize, or --no-audio-in-prompt to disable and drop it.
    - Any field also accepted by the jobs.json format above (width, height,
      fps, input_image, filename_prefix, overrides, ...) can be added to a
      scene too and will be used for that scene.
    - Output files are named "scene_<id>_..." unless a scene sets "name" or
      "filename_prefix" explicitly.
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
from resilience import ServerResolver, ProgressStore, retry_with_backoff, is_connection_error

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
        set_input(roles["audio_latent"], "frame_rate", job.get("gen_fps"))

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
# Scenes file support (multi-scene sequence -> list of job dicts)
# ---------------------------------------------------------------------------

DEFAULT_AUDIO_PROMPT_TEMPLATE = "{video_prompt}\n\nSound Design Prompt\n{audio_prompt}"


def duration_to_frames(duration_seconds: float, fps: float) -> int:
    """
    Convert a duration in seconds to a frame count LTX will accept.
    LTX video latents are temporally compressed 8x, so valid lengths are of
    the form 8*n + 1 (the workflow's own default, 49, is 8*6 + 1). We pick
    the closest such value to duration_seconds * fps, with a floor of 9
    frames (the minimum LTXVImgToVideo allows).
    """
    raw = max(float(duration_seconds) * float(fps), 9.0)
    n = round((raw - 1) / 8)
    n = max(n, 1)
    return 8 * n + 1


def load_scenes_file(path: str, gen_fps: float, audio_prompt_template: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load a scenes.json (the {"total_scene": N, "scenes": [...]} format) and
    turn it into the internal job-dict format apply_job() understands.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = data.get("scenes")
    if scenes is None:
        raise ValueError(f"{path}: expected a top-level \"scenes\" list")

    declared_total = data.get("total_scene")
    if declared_total is not None and declared_total != len(scenes):
        print(f"[warn] {path}: total_scene={declared_total} but found {len(scenes)} entries in \"scenes\"")

    scenes = sorted(scenes, key=lambda s: s.get("id", 0))

    jobs = []
    for scene in scenes:
        sid = scene.get("id")
        name = scene.get("name") or scene.get("filename_prefix") or (
            f"scene_{sid:03d}" if isinstance(sid, int) else f"scene_{sid}")

        video_prompt = scene.get("video_prompt_positive", "")
        audio_prompt = scene.get("audio_prompt")
        if audio_prompt and audio_prompt_template:
            video_prompt = audio_prompt_template.format(
                video_prompt=video_prompt, audio_prompt=audio_prompt)

        duration = scene.get("duration")
        frames = scene.get("frames")
        if frames is None and duration is not None:
            frames = duration_to_frames(duration, scene.get("gen_fps", gen_fps))

        seed = scene.get("seed")

        job = {
            "name": name,
            "image_prompt": scene.get("image_prompt_positive"),
            "image_negative_prompt": scene.get("image_prompt_negative"),
            "video_prompt": video_prompt or None,
            "video_negative_prompt": scene.get("video_prompt_negative"),
            "input_image": scene.get("input_image"),
            "width": scene.get("width"),
            "height": scene.get("height"),
            "frames": frames,
            "fps": scene.get("fps"),
            "image_seed": scene.get("image_seed", seed),
            "video_seed": scene.get("video_seed", seed),
            "image_steps": scene.get("image_steps"),
            "image_cfg": scene.get("image_cfg"),
            "video_steps": scene.get("video_steps"),
            "video_cfg": scene.get("video_cfg"),
            "filename_prefix": scene.get("filename_prefix") or name,
            "overrides": scene.get("overrides", []),
            "gen_fps": scene.get("gen_fps", gen_fps),
            "_scene_id": sid,
            "_duration": duration,
        }
        jobs.append(job)

    return jobs


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

    # multi-scene mode
    ap.add_argument("--scenes", default=None,
                     help="Path to a scenes.json ({\"total_scene\": N, \"scenes\": [...]}) for a multi-shot sequence")
    ap.add_argument("--gen-fps", type=float, default=24.0,
                     help="FPS used to convert each scene's \"duration\" (seconds) into a frame count, "
                          "and to set the LTX audio latent's frame_rate. Default 24.")
    ap.add_argument("--audio-prompt-template", default=DEFAULT_AUDIO_PROMPT_TEMPLATE,
                     help="Python str.format template combining {video_prompt} and {audio_prompt} into the "
                          "text sent to the video's positive conditioning. Only used in --scenes mode.")
    ap.add_argument("--no-audio-in-prompt", action="store_true",
                     help="In --scenes mode, don't fold audio_prompt into the video positive prompt at all")

    ap.add_argument("--server-discovery-ntfy", default=None,
                     help="ntfy topic to monitor for dynamic server URL updates")
    ap.add_argument("--watch", action="store_true",
                     help="Wait indefinitely for the server to become available instead of failing")
    ap.add_argument("--max-retries", type=int, default=3,
                     help="Maximum number of connection retries before failing")
    ap.add_argument("--retry-delay", type=float, default=5.0,
                     help="Base delay in seconds between connection retries")
    ap.add_argument("--force", action="store_true",
                     help="Force generating jobs that are already marked as done in progress.json")

    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def load_or_convert_workflow(args, server: str) -> Dict[str, Any]:
    if args.api_workflow:
        with open(args.api_workflow, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(args.workflow, "r", encoding="utf-8") as f:
        ui_workflow = json.load(f)

    print(f"Fetching node schemas from {server}/object_info ...")
    object_info = fetch_object_info(server)
    api_prompt, _meta = convert_workflow(ui_workflow, object_info, verbose=args.verbose)

    if args.save_api_workflow:
        with open(args.save_api_workflow, "w", encoding="utf-8") as f:
            json.dump(api_prompt, f, indent=2)
        print(f"Saved converted workflow to {args.save_api_workflow}")

    return api_prompt


def wait_for_server(resolver: ServerResolver, client: ComfyUIClient, max_attempts: int = 0,
                     initial_delay: float = 5.0):
    """Block until the server (following ntfy-discovered URL changes) answers a basic ping."""

    def ping():
        client.server = resolver.server
        client.get_queue()

    def on_retry(attempt, e, delay):
        print(f"[wait] server not reachable yet ({e}); retry {attempt} in {delay:.0f}s ...")
        if resolver.refresh():
            print(f"[wait] discovered new server URL: {resolver.server}")

    retry_with_backoff(ping, is_retryable=is_connection_error, max_attempts=max_attempts,
                        initial_delay=initial_delay, on_retry=on_retry)
    client.server = resolver.server
    print(f"[wait] server is up: {resolver.server}")


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

    resolver = ServerResolver(args.server, args.server_discovery_ntfy)
    if args.server_discovery_ntfy:
        if resolver.refresh():
            print(f"[main] using latest discovered server URL: {resolver.server}")
        else:
            print(f"[main] no ntfy message yet on topic '{args.server_discovery_ntfy}', "
                  f"starting with --server as given: {resolver.server}")

    client = ComfyUIClient(server=resolver.server)
    top_level_attempts = 0 if args.watch else max(args.max_retries, 1)
    wait_for_server(resolver, client, max_attempts=top_level_attempts, initial_delay=args.retry_delay)

    base_api_prompt = retry_with_backoff(
        lambda: load_or_convert_workflow(args, resolver.server),
        is_retryable=is_connection_error,
        max_attempts=top_level_attempts,
        initial_delay=args.retry_delay,
        on_retry=lambda a, e, d: (print(f"[main] workflow fetch failed ({e}); retry {a} in {d:.0f}s ..."),
                                   resolver.refresh()),
    )
    roles = discover_roles(base_api_prompt)
    if args.verbose:
        print("Discovered graph roles:", json.dumps(roles, indent=2))

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

    if args.scenes and args.jobs:
        print("ERROR: use either --scenes or --jobs, not both", file=sys.stderr)
        sys.exit(1)

    if args.scenes:
        template = None if args.no_audio_in_prompt else args.audio_prompt_template
        jobs = load_scenes_file(args.scenes, gen_fps=args.gen_fps, audio_prompt_template=template)
        print(f"Loaded {len(jobs)} scene(s) from {args.scenes}")
        for j in jobs:
            print(f"  id={j.get('_scene_id')!s:>4}  name={j['name']:<20}  "
                  f"duration={j.get('_duration')}s -> frames={j.get('frames')}")
    elif args.jobs:
        with open(args.jobs, "r", encoding="utf-8") as f:
            jobs = json.load(f)
        if not isinstance(jobs, list):
            print("ERROR: --jobs file must contain a JSON list of job objects", file=sys.stderr)
            sys.exit(1)
    else:
        jobs = [{}]  # single job using pure CLI defaults

    progress = ProgressStore(os.path.join(args.out_dir, "progress.json"))

    exit_code = 0
    for i, job in enumerate(jobs):
        merged = dict(cli_job_defaults)
        merged.update({k: v for k, v in job.items() if v is not None})
        job_name = merged.get("name") or f"job{i+1}"

        if not args.force and progress.is_done(job_name):
            print(f"[{job_name}] already completed (see {progress.path}), skipping. Use --force to redo.")
            continue

        def attempt(merged=merged, job_name=job_name):
            client.server = resolver.server
            api_prompt = apply_job(base_api_prompt, roles, merged, client)
            return run_job(client, api_prompt, job_name, args.out_dir)

        def on_retry(attempt_n, e, delay, job_name=job_name):
            print(f"\n[{job_name}] connection issue ({e}); retry {attempt_n} in {delay:.0f}s "
                  f"(server may be restarting) ...")
            if resolver.refresh():
                print(f"[{job_name}] switched to newly discovered server URL: {resolver.server}")

        job_max_attempts = 0 if args.watch else args.max_retries
        try:
            files = retry_with_backoff(attempt, is_retryable=is_connection_error,
                                        max_attempts=job_max_attempts,
                                        initial_delay=args.retry_delay, on_retry=on_retry)
            progress.mark_done(job_name, files)
        except ComfyUIError as e:
            print(f"\n[{job_name}] FAILED: {e}", file=sys.stderr)
            progress.mark_failed(job_name, str(e))
            exit_code = 1
        except Exception as e:
            print(f"\n[{job_name}] FAILED (unexpected error): {e}", file=sys.stderr)
            progress.mark_failed(job_name, str(e))
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()