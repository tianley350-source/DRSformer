# LoopDRSformerV2 short-training benchmark

This benchmark is a deterministic development comparison between the original
DRSformer and `LoopDRSformerV2`. Both models receive the same 100 augmented
32x32 crops, AdamW settings, gradient clipping and repository PSNR-Y/SSIM-Y
metric implementations.

Run from the repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
python benchmarks/compare_drsformer_v2.py --steps 100 --patch-size 32
```

The checked-in run used the three available Rain200H training pairs and three
test pairs, an RTX 4060 8 GB, PyTorch 2.5.1+cu124 and seed 100.

| Model | Parameters | PSNR-Y | SSIM-Y |
|---|---:|---:|---:|
| DRSformer | 33,655,424 | 19.3273 | 0.5179 |
| LoopDRSformerV2 | 29,695,357 | **19.4985** | **0.5591** |
| V2 change | **-11.77%** | **+0.1712 dB** | **+0.0413** |

These values establish that the implementation runs and that this ablation is
promising. They are not publication-level quality claims: 100 optimizer steps
and six paired images are far too small to estimate final Rain200H
generalization. A proper conclusion requires full training and evaluation on
the complete benchmark split.
