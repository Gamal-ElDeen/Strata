#!/usr/bin/env python3
"""Pack-format checks for tools/dflash_rt.py. No checkpoint and no download."""

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_rt


def test_validate_accepts_flash_contract():
    config = {key: value for key, value in dflash_rt.REQUIRED.items()}
    config["dflash_config"] = {
        "block_size": 8, "selector_rank": 256, "selector_top_k": 16,
        "conv_kernel_size": 2, "conv_group_size": 16, "mask_token_id": 248077,
        "target_layer_ids": [5, 15, 25, 35, 45],
    }
    geometry = dflash_rt.validate_config(config)
    assert geometry["target_layers"] == [5, 15, 25, 35, 45]
    assert geometry["mask_token"] == 248077


def test_validate_rejects_other_targets():
    config = {key: value for key, value in dflash_rt.REQUIRED.items()}
    config["dflash_config"] = {
        "block_size": 8, "selector_rank": 256, "selector_top_k": 16,
        "conv_kernel_size": 2, "conv_group_size": 16, "mask_token_id": 248077,
        "target_layer_ids": [1, 2, 3, 4, 5],
    }
    try:
        dflash_rt.validate_config(config)
    except ValueError as exc:
        assert "target_layer_ids" in str(exc)
    else:
        raise AssertionError("wrong target layers were accepted")


def _write_tiny_safetensors(path: Path):
    tensors = {
        "fc.weight": (b"\x01\x02" * 8, [2, 4]),
        "candidate_selector.hidden_projection.weight": (b"\x03\x04" * 2, [1, 2]),
    }
    header = {}
    blob = bytearray()
    for name, (raw, shape) in tensors.items():
        start = len(blob)
        blob += raw
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, start + len(raw)]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


def test_pack_roundtrip_header():
    config = {key: value for key, value in dflash_rt.REQUIRED.items()}
    config["dflash_config"] = {
        "block_size": 8, "selector_rank": 256, "selector_top_k": 16,
        "conv_kernel_size": 2, "conv_group_size": 16, "mask_token_id": 248077,
        "target_layer_ids": [5, 15, 25, 35, 45],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        snapshot = root / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text(json.dumps(config))
        _write_tiny_safetensors(snapshot / "model.safetensors")
        dest = root / "draft.dfl2"
        report = dflash_rt.pack_snapshot(snapshot, dest)
        raw = dest.read_bytes()
        assert raw[:4] == b"DFL2"
        assert report["tensors"] == 2
        assert raw[4:8] == struct.pack("<I", 1)


if __name__ == "__main__":
    test_validate_accepts_flash_contract()
    test_validate_rejects_other_targets()
    test_pack_roundtrip_header()
    print("test_dflash_rt: ok")
