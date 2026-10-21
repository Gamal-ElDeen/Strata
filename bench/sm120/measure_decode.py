#!/usr/bin/env python3
"""Repeatable SM120 decode/prefill cells for the FP4 MTP server.

Uses the already-built engine binary. Does not rebuild, and does not pass
--expert-profile-save (a bench must not overwrite the learned profile).

Refuses to start if anything is listening on port 12345 or if another process
already holds the GPU. Each config is one process. Within it, every prompt is
sent --repeats times after one warmup, with --prompt-cache 0 and ckpt=0 so a
later repeat is not a prefix-cache hit.

Pass --profile to set STRATA_DECODE_TIMING, STRATA_DRAFT_TIMING and
STRATA_VERIFY_PROFILE. Leave that off for the speed cells: the extra CUDA
events are not part of the served command.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ART = Path("/home/righteous/ai/dflash2-artifacts")
PACK = "/mnt/models/qwen/Strata-Qwen3.8-Flash-FP4-Local"
PROFILE = "/home/righteous/inference/Strata/data/expert-profile-fp4-learned.bin"
LIVE_BIN = "/home/righteous/inference/Strata/engine/strata"
PR_BIN = "/home/righteous/ai/Strata-pr3322/build-dflash2/strata"
DFLASH_GGUF = "/home/righteous/ai/dflash2-artifacts/dflash2-bf16.gguf"

QUALITY_ENV = {
    "STRATA_PREFILL_NVFP4": "w4a8",
    "STRATA_GDN_CHUNKED": "1",
    "STRATA_HC_UPMIX": "1",
    "STRATA_ADAPT_OVERLAP": "1",
    "STRATA_ROUTE_TAIL_SKIP": "0",
    "STRATA_PREFILL_HOST_PLAN": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":0:0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
}


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def gpu_busy() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    return out


def load_ids(name: str) -> str:
    return (ART / f"{name}.ids").read_text().strip()


def requests(repeats: int):
    short = load_ids("short_code")
    long = load_ids("long_agent")
    agent = load_ids("agent25k")
    one = [
        ("short_greedy", f"GEN 200 ckpt=0 {short}"),
        ("long_greedy", f"GEN 200 ckpt=0 {long}"),
        ("agent_greedy", f"GEN 200 ckpt=0 {agent}"),
        ("short_sampled", f"GEN 64 ckpt=0 temperature=0.7 top_p=0.95 top_k=20 seed=1 {short}"),
        ("long_sampled", f"GEN 64 ckpt=0 temperature=0.7 top_p=0.95 top_k=20 seed=1 {long}"),
        ("agent_sampled", f"GEN 64 ckpt=0 temperature=0.7 top_p=0.95 top_k=20 seed=1 {agent}"),
    ]
    out = [("warmup", f"GEN 8 ckpt=0 {short}")]
    for i in range(repeats):
        for name, line in one:
            out.append((f"{name}_r{i}", line))
    return out


def base_cmd(binary: str, extra: list[str], with_mtp: bool) -> list[str]:
    cmd = [
        binary, "--serve",
        "--pack", f"{PACK}/pack",
        "--native", f"{PACK}/dense.gguf",
        "--native-dense-gguf", f"{PACK}/dense.gguf",
        "--ple-gguf", f"{PACK}/ple-nvfp4.gguf",
        "--expert-profile", PROFILE,
        "--expert-cache", "auto",
        "--prefill", "auto:32768",
        "--max-context", "262144",
        "--kv", "int8",
        "--kv-resident", "32768",
        "--vram-reserve-mib", "512",
        "--pool-workers", "19",
    ]
    if with_mtp:
        cmd.extend(["--mtp", f"{PACK}/mtp/rt"])
    cmd.extend([
        "--spec", "4",
        "--spec-min-p", "0.5",
        "--adapt-every", "1",
        "--adapt-swaps", "160",
        "--adapt-decay", "0.97",
        "--prompt-cache", "0",
        *extra,
    ])
    return cmd


def parse_done(line: str) -> dict:
    # DONE gen prompt prompt_ms decode_ms finish accepted offered ...
    parts = line.split()
    if len(parts) < 8 or parts[0] != "DONE":
        return {"line": line}
    gen = int(parts[1])
    prompt = int(parts[2])
    prompt_ms = float(parts[3])
    decode_ms = float(parts[4])
    accepted = int(parts[6])
    offered = int(parts[7])
    steps = gen - accepted
    return {
        "line": line,
        "generated": gen,
        "prompt_tokens": prompt,
        "prompt_ms": prompt_ms,
        "decode_ms": decode_ms,
        "finish": parts[5],
        "accepted": accepted,
        "offered": offered,
        "acceptance": (accepted / offered) if offered else None,
        "mean_tokens_per_step": (gen / steps) if steps > 0 else None,
        "decode_tok_s": (1000.0 * gen / decode_ms) if decode_ms else None,
        "prefill_tok_s": (1000.0 * prompt / prompt_ms) if prompt_ms else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--binary", default=LIVE_BIN)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()
    if port_open(12345):
        print("refusing: 127.0.0.1:12345 is open. The live server is still serving.", file=sys.stderr)
        return 3
    busy = gpu_busy()
    if busy:
        print("refusing: GPU already has a compute process:\n" + busy, file=sys.stderr)
        return 3
    if "--expert-profile-save" in args.extra:
        print("refusing: this harness must not save over the learned expert profile", file=sys.stderr)
        return 3

    log_dir = ART / "phase0" / args.arm
    log_dir.mkdir(parents=True, exist_ok=True)
    reqs = requests(args.repeats)
    cmd = base_cmd(args.binary, args.extra, with_mtp=not args.no_mtp)
    env = os.environ.copy()
    env.update(QUALITY_ENV)
    if args.profile:
        env["STRATA_DECODE_TIMING"] = "1"
        env["STRATA_DRAFT_TIMING"] = "1"
        env["STRATA_VERIFY_PROFILE"] = "1"
    (log_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    (log_dir / "status.txt").write_text("STARTING\n")

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(ART), env=env, start_new_session=True,
    )
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    out_f = stdout_path.open("w", encoding="utf-8", errors="replace")
    err_f = stderr_path.open("w", encoding="utf-8", errors="replace")
    ready = threading.Event()
    done_n = threading.Event()
    lock = threading.Lock()
    lines_out = []
    peak = {"mib": 0}
    stop = threading.Event()
    need = len(reqs)

    def pump(pipe, dest, is_err):
        for raw in iter(pipe.readline, b""):
            text = raw.decode("utf-8", "replace")
            dest.write(text)
            dest.flush()
            if is_err and "expert cache auto:" in text:
                ready.set()
            if not is_err and text.startswith("DONE "):
                with lock:
                    lines_out.append(text.rstrip("\n"))
                    if sum(1 for s in lines_out if s.startswith("DONE ")) >= need:
                        done_n.set()
        dest.flush()

    def poll_vram():
        t0 = time.monotonic()
        with (log_dir / "vram.csv").open("w", encoding="utf-8") as vram_f:
            vram_f.write("t_sec,used_mib\n")
            while not stop.is_set():
                try:
                    used = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                    ).strip().splitlines()[0]
                    mib = int(used)
                except (subprocess.CalledProcessError, ValueError, IndexError):
                    time.sleep(2)
                    continue
                peak["mib"] = max(peak["mib"], mib)
                vram_f.write(f"{time.monotonic() - t0:.1f},{mib}\n")
                vram_f.flush()
                time.sleep(2)

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, out_f, False), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, err_f, True), daemon=True),
        threading.Thread(target=poll_vram, daemon=True),
    ]
    for t in threads:
        t.start()
    if not ready.wait(timeout=300):
        proc.kill()
        stop.set()
        (log_dir / "status.txt").write_text("FAILED ready timeout\n")
        return 1
    (log_dir / "status.txt").write_text("READY\n")
    assert proc.stdin is not None
    # One request at a time. The 24k prompt line is larger than a default pipe,
    # so writing the whole queue would block before the engine sees a newline.
    for index, (name, line) in enumerate(reqs, start=1):
        proc.stdin.write((line + "\n").encode())
        proc.stdin.flush()
        (log_dir / "status.txt").write_text(f"SENT {name}\n")
        deadline = time.monotonic() + 900
        while True:
            with lock:
                got = len(lines_out)
            if got >= index:
                break
            if proc.poll() is not None:
                stop.set()
                (log_dir / "status.txt").write_text(f"FAILED engine exited during {name}\n")
                return 1
            if time.monotonic() > deadline:
                proc.kill()
                stop.set()
                (log_dir / "status.txt").write_text(f"FAILED timeout {name}\n")
                return 1
            time.sleep(0.2)
    proc.stdin.close()
    proc.wait(timeout=120)
    stop.set()
    out_f.close()
    err_f.close()
    parsed = []
    done_lines = [s for s in lines_out if s.startswith("DONE ")]
    for (name, _), line in zip(reqs, done_lines):
        row = parse_done(line)
        row["name"] = name
        parsed.append(row)
    summary = {
        "arm": args.arm,
        "exit": proc.returncode,
        "peak_vram_mib": peak["mib"],
        "binary": args.binary,
        "profile": args.profile,
        "repeats": args.repeats,
        "requests": parsed,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (log_dir / "status.txt").write_text(f"DONE exit={proc.returncode} peak_mib={peak['mib']}\n")
    print(json.dumps({"arm": args.arm, "exit": proc.returncode, "n": len(parsed), "peak_vram_mib": peak["mib"]}))
    return 0 if proc.returncode == 0 else proc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
