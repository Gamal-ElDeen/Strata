# DFlash2 pack

`--dflash PACK` loads a local DFlash2 draft. It does not download a checkpoint, and it does not replace `--mtp`.

Pack a snapshot you already have:

```sh
python tools/dflash_rt.py /path/to/Qwen3.8-Flash-Next-DFlash2 /path/to/draft.dfl2
```

The snapshot must be `DFlash2DraftModel` for Qwen3.8-Flash-Next: hidden size 2560, five layers, block size 8, and target layers 5, 15, 25, 35, 45. The tool copies the BF16 safetensors into one `DFL2` file.

The engine flag checks that file and then stops. The verify window still drafts only with the MTP pack from `tools/mtp_rt.py`. Serving a DFlash2 proposal needs the target to keep the mean of the four hyper-connection streams after those five layers, and a draft forward that consumes them. That forward is not in the decode loop.
