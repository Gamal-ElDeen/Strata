#!/usr/bin/env python3
"""Pack a local DFlash2DraftModel snapshot into a DFL2 file the engine can load.

Nothing is downloaded. Pass the snapshot directory you already have:

    python tools/dflash_rt.py /path/to/Qwen3.8-Flash-Next-DFlash2 /path/to/out.dfl2

The Qwen3.8-Flash-Next contract is checked before any weight is copied: architecture
DFlash2DraftModel, hidden 2560, 5 sliding-window layers, block 8, selector rank 256
and top-k 16, and target layers 5, 15, 25, 35, 45.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MAGIC = b"DFL2"
VERSION = 1
FLASH_TARGET_LAYERS = [5, 15, 25, 35, 45]
REQUIRED = {
    "architectures": ["DFlash2DraftModel"],
    "hidden_size": 2560,
    "num_hidden_layers": 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 7680,
    "vocab_size": 248320,
    "sliding_window": 2048,
}


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        nbytes = struct.unpack("<Q", raw_len)[0]
        header = handle.read(nbytes)
        if len(header) != nbytes:
            raise ValueError(f"{path}: truncated safetensors header")
    parsed = json.loads(header)
    parsed.pop("__metadata__", None)
    return parsed


def validate_config(config: dict) -> dict:
    """Return the geometry the pack header stores, or raise ValueError."""
    for key, expected in REQUIRED.items():
        if config.get(key) != expected:
            raise ValueError(f"unsupported DFlash2 {key}: {config.get(key)!r}")
    dflash = config.get("dflash_config") or {}
    expected_dflash = {
        "block_size": 8,
        "selector_rank": 256,
        "selector_top_k": 16,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "mask_token_id": 248077,
        "target_layer_ids": FLASH_TARGET_LAYERS,
    }
    for key, expected in expected_dflash.items():
        if dflash.get(key) != expected:
            raise ValueError(f"unsupported DFlash2 {key}: {dflash.get(key)!r}")
    return {
        "hidden": config["hidden_size"],
        "n_layers": config["num_hidden_layers"],
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],
        "intermediate": config["intermediate_size"],
        "vocab": config["vocab_size"],
        "block_size": dflash["block_size"],
        "selector_rank": dflash["selector_rank"],
        "selector_top_k": dflash["selector_top_k"],
        "conv_kernel": dflash["conv_kernel_size"],
        "conv_group": dflash["conv_group_size"],
        "sliding_window": config["sliding_window"],
        "mask_token": dflash["mask_token_id"],
        "target_layers": list(dflash["target_layer_ids"]),
    }


def _bf16_nbytes(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"bad tensor shape {shape}")
        count *= dim
    return count * 2


def pack_snapshot(snapshot: Path, dest: Path) -> dict:
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f"{snapshot} needs config.json and model.safetensors")
    geometry = validate_config(json.loads(config_path.read_text()))
    header = _safetensors_header(weights_path)
    names = sorted(header)
    if "fc.weight" not in header or "candidate_selector.hidden_projection.weight" not in header:
        raise ValueError("snapshot is missing the feature projection or the selector")
    table = bytearray()
    blob = bytearray()
    records = []
    with weights_path.open("rb") as handle:
        raw_len = struct.unpack("<Q", handle.read(8))[0]
        data_origin = 8 + raw_len
        for name in names:
            info = header[name]
            if info.get("dtype") != "BF16":
                raise ValueError(f"{name}: expected BF16, got {info.get('dtype')}")
            shape = [int(dim) for dim in info["shape"]]
            nbytes = _bf16_nbytes(shape)
            start, end = info["data_offsets"]
            if end - start != nbytes:
                raise ValueError(f"{name}: data span {end - start} != {nbytes}")
            handle.seek(data_origin + start)
            raw = handle.read(nbytes)
            if len(raw) != nbytes:
                raise ValueError(f"{name}: short weight read")
            encoded = name.encode("utf-8")
            offset = len(blob)
            table += _u32(len(encoded)) + encoded + _u32(len(shape))
            dims = shape + [0] * (4 - len(shape))
            table += b"".join(_u64(dim) for dim in dims)
            table += _u64(offset) + _u64(nbytes)
            blob += raw
            records.append({"name": name, "shape": shape, "nbytes": nbytes})
    layers = geometry["target_layers"] + [-1] * (8 - len(geometry["target_layers"]))
    packed = bytearray(MAGIC)
    packed += _u32(VERSION) + _u32(len(names)) + _u32(0)
    for key in ("hidden", "n_layers", "n_heads", "n_kv_heads", "head_dim", "intermediate", "vocab",
                "block_size", "selector_rank", "selector_top_k", "conv_kernel", "conv_group",
                "sliding_window"):
        packed += _u32(geometry[key])
    packed += _u32(geometry["mask_token"] & 0xFFFFFFFF)
    packed += _u32(len(geometry["target_layers"]))
    packed += b"".join(_u32(layer & 0xFFFFFFFF) for layer in layers)
    packed += table
    packed += blob
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(packed)
    return {"path": str(dest), "bytes": len(packed), "tensors": len(records), "geometry": geometry}


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack a local DFlash2 snapshot into a DFL2 file")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    try:
        report = pack_snapshot(args.snapshot, args.dest)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"dflash_rt: {exc}", file=sys.stderr)
        return 1
    print(f"dflash_rt: wrote {report['path']} ({report['tensors']} tensors, {report['bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
