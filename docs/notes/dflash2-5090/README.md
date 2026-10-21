# DFlash2 and decode-speed work on Strata (RTX 5090, SM_120)

Status as of 2026-10-21. These are working notes from our own runs, not upstream claims.
A number below is from one run unless the heading says five runs. The five-run cells list every run, in order, after one warmup. Differences under about 3–4% are noise. The first run of a cell is often colder than the next four.

## Goal

Make decode faster on this RTX 5090 (SM_120, 32 GB, driver 615.71.09, CUDA 13.4) for the Qwen3.8-Flash-Next FP4 pack. The first attempt was a DFlash2 drafter in place of MTP. Phase 0 then measured the MTP server that is actually served, plus a few startup-flag changes, and profiled one decode step.

## Branches

- `dflash2` at `4385b137` before this note: loader only (`tools/dflash_rt.py`, `--dflash PATH`, rejects `--mtp`). It validates the pack and stops before serving. No capture or forward code. Nothing from PR #3322 was merged here.
- `decode-sm120` from v0.2.1.4 (`9de04cd4`). Before this note the tip was `e6aee466` (harness fix on top of `1cfdd858`). Phase 0 benchmark only. No engine changes.

Port 12345 is the MTP server from `~/inference/Strata`, not either of these branches. After the Phase 0 queue the unit was started again. `/health` reported `loaded: true`. One completion (`temperature` 0, `max_tokens` 32) returned the word `pong` and finished with `stop`.

## Single run: PR #3322 against itself

Upstream PR #3322, author Jakub Luwierski, branch `codex/dflash2-main`, SHA tested `784114535b9a133896b77baef4275e9e78a0d7cb`. Built unmodified for SM_120 with `STRATA_ENABLE_DFLASH2=ON`. The seven `dflash` ctests passed. No source patch. The drafter was the BF16 GGUF from `tools/dflash_gguf.py` over the official safetensors (`sha256 d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`). The script has no Q8 flag, so Q8_0 was not converted.

Both arms used that PR binary. Context 262144, `--spec 4`, `--spec-min-p 0.5`, 19 pool workers, suffix drafter left on, prompt cache off, learned expert profile, w4a8, tail skip off. One run per cell. With the suffix drafter left on, `--spec 4` raises the verify cap and this tree drafts K=5 unless `--dflash-block 3` is set. The fair width is K=3.

| Single run, decode tok/s | MTP | DFlash2 K=3 | DFlash2 K=5 |
|---|---:|---:|---:|
| 141-token prompt, greedy, 200 tokens | 166.2 | 145.5 | 144.7 |
| 5,269-token prompt, greedy, 200 tokens | 163.0 | 143.1 | 163.7 |
| 141-token prompt, temperature 0.7, 64 tokens | 197.4 | 114.4 | 110.6 |

Acceptance on that single K=3 pair: short greedy 129/175 (MTP) and 129/172 (DFlash2); long greedy 122/162 and 122/157. At the same draft width acceptance was similar and DFlash2 was about 12% slower. Each K=3 step cost roughly 2.2–2.5 ms more than MTP on that binary (about 19.4 vs 16.9 ms on the short prompt, 17.9 vs 15.7 ms on the 5,269-token prompt). Peak VRAM was 32178 MiB for MTP and for K=3, and 32184 MiB for K=5. The drafter held 1210 MiB (1150 MiB of BF16 weights) and the expert cache was 8932 slots against MTP's 9074. At temperature above 0 the PR logs that DFlash2 uses target-only decoding. A comment with this single-run table was posted on the PR.

DFlash2 was not adopted from that comparison. Cherry-picking the 58 commits onto v0.2.1.4 would be a different binary from the one measured.

## Five runs: the served MTP binary

Phase 0 used `/home/righteous/inference/Strata/engine/strata` (built 2026-10-19, engine string 0.1.45.3, tree v0.2.0-beta). Same quality flags as the unit: w4a8, tail skip off, chunked GDN, HC upmix, adaptive overlap, prefill host plan off, `--prompt-cache 0`. The harness does not pass `--expert-profile-save`. Prompts: 141, 5,269, and 24,267 tokens. Greedy generates 200 tokens. Sampled is temperature 0.7, top_p 0.95, top_k 20, seed 1, 64 tokens. Every finished request in these cells ended on the length limit.

This binary is not the PR binary. Do not read these MTP numbers as a second measurement of the table above.

Served cell (`--spec 4`, `--spec-min-p 0.5`, 19 pool workers, suffix drafter on). The log's verify window is 6 tokens because the suffix path adds 2. Exit 0. Peak 32184 MiB.

| Five runs, served MTP | decode tok/s | accepted/offered |
|---|---|---|
| 141 greedy | 162.8, 202.4, 194.7, 205.9, 192.8 | 125/192, 131/175, 126/175, 131/174, 126/190 |
| 5,269 greedy | 175.2, 191.4, 193.4, 186.7, 188.9 | 128/155, 128/159, 128/162, 125/163, 127/161 |
| 24,267 greedy | 194.6, 196.7, 202.5, 188.1, 192.3 | 133/164, 133/163, 132/164, 123/160, 129/156 |
| 141 sampled | 177.6, 190.1, 194.1, 191.0, 191.3 | 43/52, 45/51, 45/46, 45/49, 45/47 |
| 5,269 sampled | 197.5, 165.2, 200.0, 181.7, 193.6 | 44/50, 40/50, 44/53, 43/55, 46/52 |
| 24,267 sampled | 218.7, 201.9, 223.2, 224.7, 210.2 | 46/48, 46/54, 46/47, 47/47, 46/50 |

Prefill tok/s, same five runs: 5,269 greedy 3873.7, 4139.7, 4085.8, 4095.3, 4084.2. 24,267 greedy 7024.1, 7071.4, 7056.6, 7046.4, 7064.2.

`--spec 3` (suffix still on, so the logged window is 5). Exit 0. Peak 32172 MiB. Decode tok/s: 141 greedy 158.0, 192.4, 195.3, 190.1, 198.1. 5,269 greedy 168.5, 175.9, 175.3, 168.3, 177.8. 24,267 greedy 179.1, 187.0, 191.1, 182.4, 183.1. 24,267 sampled 187.6, 193.3, 205.1, 200.8, 197.2. Slower than spec 4 on the long greedy prompts.

`--spec 5` (logged window 7). Exit 0. Peak 32200 MiB. Decode tok/s: 141 greedy 156.5, 202.1, 196.8, 213.3, 195.6. 5,269 greedy 159.2, 177.8, 184.5, 179.9, 188.6, with acceptance 127/187, 128/191, 131/179, 130/192, 134/190. 24,267 greedy 193.4, 211.6, 223.3, 199.2, 200.3. 24,267 sampled 206.7, 229.4, 247.3, 221.3, 220.2. Not a clear win over spec 4. The 5,269-token greedy acceptance is lower.

`--spec 6` did not finish. One run only, and only the first request completed: 141 greedy, 161.1 tok/s, 143/229 accepted. The suffix bump makes the verify window 8 tokens. Capturing that window left about 13 MiB free, and the 5,269-token prefill then failed in `cudaMalloc` inside the prefill MMQ path (`ggml_cuda_host.cu:127`). The process exited during `long_greedy_r0`. `--vram-reserve-mib` was not raised.

`--spec-min-p 0.3` (spec 4, suffix on). Exit 0. Peak 32192 MiB. Decode tok/s: 141 greedy 163.1, 193.2, 187.5, 210.0, 188.9. 5,269 greedy 170.6, 188.6, 181.0, 185.3, 180.3. 24,267 greedy 185.8, 209.1, 204.2, 215.7, 220.5. 24,267 sampled 228.5, 243.1, 242.4, 233.6, 244.3. The sampled 24,267-token cell is faster than the served 0.5 gate. The greedy cells are not.

`--spec-min-p 0.7`. Exit 0. Peak 32184 MiB. Decode tok/s: 141 greedy 159.8, 192.4, 192.7, 194.4, 191.0. 5,269 greedy 166.8, 179.7, 179.1, 162.8, 169.6. 24,267 sampled 207.4, 212.8, 214.2, 204.6, 207.1. Acceptance went up and decode did not. Not a candidate.

13 pool workers (spec 4, min-p 0.5, suffix on). Exit 0. Peak 32184 MiB. The log shows 13 workers. Decode tok/s: 141 greedy 169.8, 206.8, 205.7, 205.0, 205.8. 5,269 greedy 181.1, 191.8, 189.8, 189.9, 180.2. 24,267 greedy 215.1, 197.7, 220.4, 206.6, 212.5. 24,267 sampled 221.0, 218.1, 232.2, 221.5, 222.2. Prefill on the 24,267-token greedy runs: 6986.7, 7066.5, 7062.4, 7040.9, 7055.6. Decode is the same or a bit faster than 19 workers, and prefill did not drop. The live unit was left at 19 workers.

`--suffix-draft 0` (spec 4). Exit 0. Peak 32166 MiB. Logged verify window 4 tokens. Decode tok/s: 141 greedy 165.2, 199.0, 200.0, 206.0, 191.6. 5,269 greedy 171.5, 188.2, 186.8, 181.8, 180.4. 24,267 greedy 207.8, 218.3, 188.9, 199.0, 218.7. 24,267 sampled 210.7, 222.5, 219.6, 220.5, 223.0. Decode did not fall. On these prompts the extra two suffix tokens are not paying for the wider window.

## Five runs: DFlash2 K=3, suffix off, PR binary

Separate binary: the unmodified PR build at `784114535b9a`, not the served MTP engine. `--no-mtp`, `--dflash-block 3`, `--dflash-window 0`, `--suffix-draft 0`. Logged window 4. Exit 0. Peak 32162 MiB. The load line says pass K=3, a 2060-cell K/V ring, and 1150.3 MiB of weights. Sampled requests log `dflash: sampled request uses target-only decoding` and accepted 0/0.

| Five runs, PR DFlash2 K=3, suffix off | decode tok/s | accepted/offered |
|---|---|---|
| 141 greedy | 172.6, 198.4, 208.7, 193.3, 201.9 | 129/162, 130/167, 133/158, 127/172, 130/167 |
| 5,269 greedy | 168.9, 165.4, 170.8, 180.1, 174.6 | 129/163, 124/161, 124/164, 129/164, 126/169 |
| 24,267 greedy | 189.3, 198.3, 193.1, 190.2, 183.5 | 133/157, 135/154, 133/161, 135/167, 131/169 |
| 141 sampled | 101.0, 101.2, 103.2, 100.9, 100.0 | 0/0, 0/0, 0/0, 0/0, 0/0 |
| 5,269 sampled | 108.1, 109.6, 109.5, 109.9, 107.9 | 0/0, 0/0, 0/0, 0/0, 0/0 |
| 24,267 sampled | 112.1, 111.5, 110.1, 109.2, 108.1 | 0/0, 0/0, 0/0, 0/0, 0/0 |

Prefill tok/s: 5,269 greedy 3338.6, 3493.3, 3519.0, 3489.6, 3510.3. 24,267 greedy 5483.2, 5532.5, 5529.2, 5514.3, 5515.1. Against the served MTP cell above, that is a lower prefill rate (about 3500 vs about 4100 on 5,269 tokens, and about 5500 vs about 7050 on 24,267). Greedy decode on the 141-token prompt landed in the same band as the served MTP cell. The 5,269-token greedy decode was slower. This is not a same-binary comparison. The earlier single-run table is the same-binary one, and there DFlash2 lost.

## Profile of one MTP decode step

One run, served flags, `STRATA_DECODE_TIMING=1` and `STRATA_VERIFY_PROFILE=1`. Not nsys and not ncu. `STRATA_DRAFT_TIMING` did not print; that line waits for 300 rounds, and this run had fewer. The profile flag turns off the shared-expert side stream and the fused wait/copy, so these windows are slower than the unprofiled cells. Short greedy in this run was 148.0 tok/s, one run, against about 193–206 tok/s in the unprofiled served cell.

Short greedy, 77 windows, one run: 17.55 ms/window = verify 15.41 (GPU-reach wait 7.94 + per-layer host 6.59, of which CPU experts 6.21) + commit/emit 0.29 + draft 1.78. GPU stage total 14.99 ms/window. On the GDN layers: qkv GEMV 1.01, resident expert GEMM (`VRAM hits`) 1.60, `waitB` 2.38, `waitCPU` 1.17, launch gap 0.02.

24,267 greedy, 66 windows, one run: 15.41 ms/window, draft 1.81, CPU experts 4.20 inside the per-layer host time.

Decode is not launch-gap bound. A megakernel that only removes kernel launches would be chasing 0.02 ms/window. The host spends about 6 ms of a 15 ms verify in the CPU expert pool, and the GPU timeline shows multi-millisecond waits for the PCIe share and the CPU rows. Those waits are partly hidden in the unprofiled build, because the profile flag disables that fusion. The payoff to chase is the expert handshake, not a fused copy of the math kernels. The draft itself is about 1.8 ms at the end of the window.

## Prefill on the live server

Checked from `strata-fp4-local.log` on the 2026-10-21 07:06 boot, before Phase 0 stopped the unit. Same binary and flags, pcie_frac settled at 0.30, expert cache 9075 slots. One fresh read over 100k tokens: 138,790 tokens at 6,670.4 tok/s, one request. The previous boot (2026-10-20 10:13) had 233 fresh reads in the 100k–200k band at 6,701.4 tok/s weighted, median 6,694. Decode for requests with at least 200 generated tokens was 146.8 tok/s on this boot and 147.2 on the previous one. The session-mean prefill looked low because most turns only read 1–2,000 new tokens, at about 826 tok/s, which matches that previous boot's short-read band (785 tok/s). The restart cleared the prefix cache, so the first turn of a conversation was a full read in wall time.

## Not tested

- Q8_0 drafter, block sizes 6 and 7, and acceptance floors 0.7–0.85. Those are the PR author's settings on another GPU. They were not run here.
- Drafter taps were not compared numerically against a Hugging Face forward. The served target is the FP4 pack.
- Coupled-draft sampling. The PR still falls back to target-only when temperature is above 0.
- A same-binary five-run of DFlash2 against MTP inside the PR tree. The five-run DFlash2 cell above is the PR binary; the five-run MTP cells are the served binary.
- nsys and ncu. The profile used the engine timers.

## Result so far

DFlash2 in BF16 did not beat MTP in the same-binary single run, and it was not adopted. The served MTP command is what is running on port 12345. Phase 0's config cells that did not lose decode were 13 pool workers and suffix drafting off. Spec 6 with the suffix bump does not fit in VRAM. The profile says the next engine work is the expert wait, not a launch-gap megakernel.
