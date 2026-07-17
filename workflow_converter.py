"""
workflow_converter.py

Converts a ComfyUI *UI-format* workflow export (what you get from "Save (API)"
disabled, i.e. the normal "Save" / drag-drop-able workflow with `nodes` +
`links`) into the *API-format* prompt JSON that the ComfyUI `/prompt` HTTP
endpoint expects (a flat dict of `{node_id: {"class_type": ..., "inputs": {...}}}`).

Why this exists instead of hand-writing the API JSON:
    Several nodes in this workflow (MultiGPU_WorkUnits, VRAM_Debug, RIFE VFI,
    the LTXV nodes, etc.) come from custom node packs whose exact widget
    order can change between versions. Instead of guessing, this converter
    asks the *running* ComfyUI server for the authoritative input schema of
    every node type via `GET /object_info`, then positionally maps each
    node's `widgets_values` onto the correct input names. This makes the
    conversion correct no matter which version of the custom nodes you have
    installed -- as long as ComfyUI is up and reachable when you run this.

It also flattens ComfyUI "subgraph" nodes (the collapsed sub-workflow blocks
newer ComfyUI frontends support), since the API format has no concept of
subgraphs -- only real nodes.

Usage (standalone):
    python workflow_converter.py Qwen_LTX2_3.json --server http://127.0.0.1:8188 -o workflow_api.json

Usage (as a library):
    from workflow_converter import convert_workflow
    api_prompt, meta = convert_workflow(ui_workflow_dict, object_info_dict)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

CONTROL_AFTER_GENERATE_VALUES = {"fixed", "increment", "decrement", "randomize"}

# Node types that exist in the UI graph purely for editor convenience and
# have no equivalent / are meaningless in a headless API submission.
SKIP_NODE_TYPES = {
    "Note",
    "MarkdownNote",
    "Reroute",
}


def fetch_object_info(server_url: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Fetch the full node schema catalogue from a running ComfyUI server."""
    url = server_url.rstrip("/") + "/object_info"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Subgraph flattening
# ---------------------------------------------------------------------------

def _index_subgraphs(workflow: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    subgraphs = workflow.get("definitions", {}).get("subgraphs", [])
    return {sg["id"]: sg for sg in subgraphs}


def flatten_subgraphs(workflow: Dict[str, Any]) -> Tuple[List[dict], List[list]]:
    """
    Returns (nodes, links) with every subgraph-instance node replaced by its
    real internal nodes, fully rewired. Handles nesting (a subgraph containing
    another subgraph instance) by recursing.

    Links keep the original list-tuple format:
        [link_id, origin_id, origin_slot, target_id, target_slot, type]
    Node/link ids belonging to an inlined subgraph are namespaced as
    "<outer_node_id>::<inner_id>" to guarantee global uniqueness.
    """
    subgraph_defs = _index_subgraphs(workflow)
    nodes = copy.deepcopy(workflow["nodes"])
    links = copy.deepcopy(workflow.get("links", []))

    def link_lookup(link_list):
        return {l[0]: l for l in link_list}

    changed = True
    while changed:
        changed = False
        link_map = link_lookup(links)
        new_nodes = []
        new_links = list(links)
        nodes_to_remove = set()
        links_to_remove = set()

        for node in nodes:
            ntype = node.get("type")
            if ntype not in subgraph_defs:
                continue

            changed = True
            outer_id = node["id"]
            sg = subgraph_defs[ntype]
            ns = lambda inner_id: f"{outer_id}::{inner_id}"  # noqa: E731

            inner_nodes = copy.deepcopy(sg.get("nodes", []))
            inner_links = copy.deepcopy(sg.get("links", []))
            inner_link_map = {l["id"]: l for l in inner_links}

            # Remap every inner node id -> namespaced id
            for inode in inner_nodes:
                inode["id"] = ns(inode["id"])
                for inp in inode.get("inputs", []):
                    pass  # link ids fixed up below via inner_links rebuild
                for out in inode.get("outputs", []):
                    pass

            id_map = {}
            for orig, renamed in zip(sg.get("nodes", []), inner_nodes):
                id_map[orig["id"]] = renamed["id"]

            # Rebuild inner node input 'link' fields and output 'links' fields
            # using namespaced link ids so they don't collide with outer links.
            def inner_link_id(lid):
                return ns(f"L{lid}")

            for inode, orig in zip(inner_nodes, sg.get("nodes", [])):
                for inp, orig_inp in zip(inode.get("inputs", []), orig.get("inputs", [])):
                    if orig_inp.get("link") is not None:
                        inp["link"] = inner_link_id(orig_inp["link"])
                for out, orig_out in zip(inode.get("outputs", []), orig.get("outputs", [])):
                    if orig_out.get("links"):
                        out["links"] = [inner_link_id(x) for x in orig_out["links"]]

            # Outer input sockets of the subgraph instance node
            outer_inputs = node.get("inputs", [])
            # Subgraph "inputs" definitions describe, for each boundary input,
            # which internal (node,slot) pairs it feeds via linkIds referencing
            # links whose origin_id == -10 (the special input-proxy node).
            for idx, sg_input in enumerate(sg.get("inputs", [])):
                if idx >= len(outer_inputs):
                    continue
                outer_link_id = outer_inputs[idx].get("link")
                if outer_link_id is None:
                    continue  # nothing was connected to this boundary input
                outer_link = link_map.get(outer_link_id)
                if outer_link is None:
                    continue
                origin_id, origin_slot = outer_link[1], outer_link[2]

                for lid in sg_input.get("linkIds", []):
                    ilink = inner_link_map.get(lid)
                    if not ilink or ilink["origin_id"] != -10:
                        continue
                    target_inner_id = id_map[ilink["target_id"]]
                    target_slot = ilink["target_slot"]
                    # Point the (namespaced) inner link's origin at the real
                    # outer producer instead of the -10 proxy node.
                    new_link_id = inner_link_id(lid)
                    new_links.append([new_link_id, origin_id, origin_slot,
                                       target_inner_id, target_slot, outer_link[5]])

            # Subgraph "outputs": for each boundary output, find the internal
            # (node,slot) that feeds the -20 proxy, then redirect every link
            # that used to originate at the outer subgraph node to originate
            # at that internal node/slot instead.
            outer_outputs = node.get("outputs", [])
            for idx, sg_output in enumerate(sg.get("outputs", [])):
                if idx >= len(outer_outputs):
                    continue
                consumer_link_ids = outer_outputs[idx].get("links") or []
                real_origin = None
                for lid in sg_output.get("linkIds", []):
                    ilink = inner_link_map.get(lid)
                    if not ilink or ilink["target_id"] != -20:
                        continue
                    real_origin = (id_map[ilink["origin_id"]], ilink["origin_slot"])
                if real_origin is None:
                    continue
                for lid in consumer_link_ids:
                    for l in new_links:
                        if l[0] == lid:
                            l[1], l[2] = real_origin[0], real_origin[1]

            new_nodes.extend(inner_nodes)
            nodes_to_remove.add(outer_id)
            # remove the now-obsolete links that touched the subgraph
            # instance node directly
            for l in list(new_links):
                if l[1] == outer_id or l[3] == outer_id:
                    links_to_remove.add(l[0])

        if changed:
            nodes = [n for n in nodes if n["id"] not in nodes_to_remove] + new_nodes
            links = [l for l in new_links if l[0] not in links_to_remove]

    return nodes, links


# ---------------------------------------------------------------------------
# API-format conversion
# ---------------------------------------------------------------------------

def _schema_input_order(schema_for_type: Dict[str, Any]) -> List[Tuple[str, Any, bool]]:
    """
    Returns [(name, type_spec, is_required), ...] in declaration order for a
    node type, based on the /object_info payload for that type.
    """
    order = []
    input_block = schema_for_type.get("input", {})
    for name, spec in input_block.get("required", {}).items():
        order.append((name, spec, True))
    for name, spec in input_block.get("optional", {}).items():
        order.append((name, spec, False))
    return order


def _is_widget_spec(spec) -> bool:
    """A spec is a widget (as opposed to a socket-only type) if its first
    element is a list of choices, or a primitive widget type name."""
    if not spec:
        return False
    t = spec[0]
    if isinstance(t, list):
        return True
    return t in {"INT", "FLOAT", "STRING", "BOOLEAN"}


def convert_workflow(
    workflow: Dict[str, Any],
    object_info: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convert a UI-format workflow dict to an API-format prompt dict.

    Returns (api_prompt, meta) where meta contains helpful lookups:
        meta["by_title"]      -> {title: node_id_str}
        meta["by_type"]       -> {class_type: [node_id_str, ...]}
        meta["node_titles"]   -> {node_id_str: title_or_type}
    """
    nodes, links = flatten_subgraphs(workflow)
    link_map = {l[0]: l for l in links}

    api_prompt: Dict[str, Any] = {}
    by_title: Dict[str, str] = {}
    by_type: Dict[str, List[str]] = {}
    node_titles: Dict[str, str] = {}

    for node in nodes:
        ntype = node.get("type")
        if ntype in SKIP_NODE_TYPES:
            continue
        if node.get("mode") == 4:  # muted/bypassed node in the editor
            if verbose:
                print(f"[skip] node {node['id']} ({ntype}) is muted/bypassed")
            continue

        node_id = str(node["id"])
        schema = object_info.get(ntype)
        if schema is None:
            raise ValueError(
                f"Node type '{ntype}' (id={node_id}) not found in this server's "
                f"/object_info. Make sure the matching custom node pack is "
                f"installed and ComfyUI has fully started."
            )

        input_order = _schema_input_order(schema)
        ui_inputs_by_name = {i["name"]: i for i in node.get("inputs", [])}
        widgets_values = list(node.get("widgets_values") or [])
        wv_idx = 0

        api_inputs: Dict[str, Any] = {}

        for name, spec, required in input_order:
            ui_socket = ui_inputs_by_name.get(name)
            has_socket_entry = ui_socket is not None
            is_connected = has_socket_entry and ui_socket.get("link") is not None

            if is_connected:
                link = link_map.get(ui_socket["link"])
                if link is None:
                    continue
                origin_id, origin_slot = link[1], link[2]
                api_inputs[name] = [str(origin_id), origin_slot]
                continue

            if has_socket_entry and not is_connected:
                # Declared as a socket in the UI but nothing plugged in.
                # If it's also a widget-capable type it may still have a
                # value sitting in widgets_values (some nodes keep both).
                if _is_widget_spec(spec) and wv_idx < len(widgets_values):
                    api_inputs[name] = widgets_values[wv_idx]
                    wv_idx += 1
                # otherwise leave unset; ComfyUI will use the node default
                continue

            # Pure widget input -> pull next value from widgets_values
            if wv_idx < len(widgets_values):
                value = widgets_values[wv_idx]
                api_inputs[name] = value
                wv_idx += 1

                # seed / noise_seed widgets are followed in the UI by a
                # "control_after_generate" combo (fixed/increment/.../randomize)
                # that is NOT a real backend input -- skip it if present.
                if name in ("seed", "noise_seed") and wv_idx < len(widgets_values):
                    peek = widgets_values[wv_idx]
                    if isinstance(peek, str) and peek in CONTROL_AFTER_GENERATE_VALUES:
                        wv_idx += 1
            elif required:
                if verbose:
                    print(f"[warn] node {node_id} ({ntype}) missing widget value for "
                          f"required input '{name}'; leaving to server default.")

        title = node.get("title") or ntype
        api_prompt[node_id] = {
            "class_type": ntype,
            "inputs": api_inputs,
            "_meta": {"title": title},
        }
        node_titles[node_id] = title
        by_type.setdefault(ntype, []).append(node_id)
        if node.get("title"):
            by_title[node["title"]] = node_id

    meta = {"by_title": by_title, "by_type": by_type, "node_titles": node_titles}
    return api_prompt, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workflow", help="Path to UI-format workflow JSON (e.g. Qwen_LTX2_3.json)")
    ap.add_argument("--server", default="http://127.0.0.1:8188",
                     help="Running ComfyUI server base URL used to fetch /object_info")
    ap.add_argument("-o", "--output", default="workflow_api.json",
                     help="Where to write the converted API-format JSON")
    ap.add_argument("--object-info-cache", default=None,
                     help="Optional path to a cached object_info.json instead of hitting the server")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.workflow, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    if args.object_info_cache:
        with open(args.object_info_cache, "r", encoding="utf-8") as f:
            object_info = json.load(f)
    else:
        print(f"Fetching node schemas from {args.server}/object_info ...")
        object_info = fetch_object_info(args.server)

    api_prompt, meta = convert_workflow(workflow, object_info, verbose=args.verbose)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(api_prompt, f, indent=2)

    print(f"Wrote {args.output} ({len(api_prompt)} nodes)")
    print("\nNode id -> title (useful for scripting):")
    for nid, title in meta["node_titles"].items():
        print(f"  {nid:>4}  {api_prompt[nid]['class_type']:<28} {title}")


if __name__ == "__main__":
    main()
