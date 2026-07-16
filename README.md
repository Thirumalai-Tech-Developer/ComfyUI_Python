# ComfyUI Automation — Qwen Image → LTX-2.3 Video

Automates your `Qwen_LTX2_3.json` workflow (Qwen text-to-image start frame →
LTX-2.3 image-to-video with foley audio → mp4) so you can run it from the
command line / cron / another script instead of clicking through the ComfyUI
web UI.

## How it works

Your workflow JSON is the **UI-format** export (the normal one, with a
`nodes`/`links` graph and a collapsed "subgraph" for the audio latent node).
ComfyUI's `/prompt` API needs a different, flat **API-format** JSON. Rather
than hand-guessing the input names for your custom nodes (`MultiGPU_WorkUnits`,
`VRAM_Debug`, `RIFE VFI`, the LTXV nodes, etc.) — which can change between
versions — `workflow_converter.py` asks your **running ComfyUI server** for
the authoritative schema of every node (`GET /object_info`) and uses that to
convert correctly, every time. It also automatically flattens the collapsed
subgraph node into real nodes.

Files:
- `workflow_converter.py` — UI JSON → API JSON converter (also runnable standalone).
- `comfyui_client.py` — small HTTP + WebSocket client: submit a prompt, wait
  for completion, download the resulting files.
- `generate.py` — the script you actually run. Patches prompts / seed /
  resolution / frame count / fps into the graph, submits it, and saves the
  output video(s).
- `Qwen_LTX2_3.json` — your original workflow (copied in for convenience).
- `jobs.example.json` — example batch file.

## Setup

```bash
pip install -r requirements.txt
```

Your ComfyUI server needs to already be **running and reachable** (this is
just the automation layer around the notebook you shared — the notebook
still installs ComfyUI/models and starts the server + Cloudflare tunnel;
these scripts just talk to it over HTTP).

## 1. One-off video

```bash
python generate.py \
  --server http://127.0.0.1:8188 \
  --image-prompt "a cozy reading nook, warm lamp light, rain on the window" \
  --image-negative-prompt "blurry, low quality, cartoon, watermark" \
  --video-prompt "slow push in, curtains sway gently, steam rises from a mug" \
  --video-negative-prompt "static camera, jittery motion, flickering" \
  --width 512 --height 512 --frames 49 --fps 24 \
  --filename-prefix "cozy_nook"
```

If your ComfyUI is behind the notebook's Cloudflare tunnel, just pass that
URL as `--server` (e.g. `--server https://xxxx.trycloudflare.com`).

Outputs land in `./outputs/` by default (`--out-dir` to change).

## 2. Image-to-video only (skip Qwen, use your own start frame)

```bash
python generate.py \
  --server http://127.0.0.1:8188 \
  --input-image ./my_photo.png \
  --video-prompt "camera slowly orbits left, hair moves in the breeze" \
  --frames 49 --fps 24
```

This uploads your local image to ComfyUI, wires it directly into the LTX
image-to-video node, and drops the preview-image output so the Qwen
generation branch never runs (saves a lot of time/VRAM).

## 3. Batch processing many jobs

```bash
python generate.py --server http://127.0.0.1:8188 --jobs jobs.example.json
```

Each entry in the JSON list can override any of: `image_prompt`,
`image_negative_prompt`, `video_prompt`, `video_negative_prompt`,
`input_image`, `width`, `height`, `frames`, `fps`, `image_seed`,
`video_seed`, `image_steps`, `image_cfg`, `video_steps`, `video_cfg`,
`filename_prefix`, `name`. Anything you omit falls back to the matching
`--flag` you passed on the command line (or the workflow's original default
if you didn't pass that flag either).

## 4. Tweaking anything else

Every input on every node can be poked directly with `--override`, without
editing any code:

```bash
--override 'CreateVideo.fps=30' \
--override 'LoraLoaderModelOnly.strength_model=0.7'
```

Format is `ClassType.input_name=value` (patches the first node of that
type) or `nodeId.input_name=value` (patches a specific node — get ids by
running the converter standalone, see below). This also works per-job via
an `"overrides": ["...", "..."]` list inside a job entry in the jobs file.

## 5. Inspecting the converted graph / node ids

```bash
python workflow_converter.py Qwen_LTX2_3.json --server http://127.0.0.1:8188 -o workflow_api.json
```

Prints every node id, type, and title — handy for `--override` targeting or
debugging. `generate.py` also accepts `--api-workflow workflow_api.json` to
skip re-conversion (and re-hitting `/object_info`) on subsequent runs, and
`--save-api-workflow out.json` to save the converted graph as it builds it.

## Notes / gotchas

- **Seeds**: pass `--image-seed` / `--video-seed` explicitly for
  reproducible runs. The original workflow has both KSamplers set to
  "randomize" mode in the UI — the API format has no such thing, so without
  an explicit seed you'll just get whatever seed happened to be baked into
  the JSON at export time, repeated every run.
- **`frames` also drives the audio node**: `--frames` patches both
  `LTXVImgToVideo.length` and `LTXVEmptyLatentAudio.frames_number` together,
  since they need to stay in sync (this mirrors what your workflow already
  had: 49 / 49).
- If conversion fails with `Node type 'X' not found in this server's
  /object_info`, it means that custom node pack isn't installed/loaded on
  the server you pointed `--server` at — double check the notebook's
  install cells all ran successfully.
- The client tries a WebSocket connection first for live progress and falls
  back to polling `/history` if that's unreachable (e.g. some tunnels only
  proxy HTTP).
