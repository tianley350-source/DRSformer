# LoopDRSformer short-training benchmark

This benchmark is a deterministic development comparison between the original
DRSformer, `LoopDRSformerV2`, and `LoopDRSformerV3`. All models receive the
same 100 augmented 32x32 crops, AdamW settings, gradient clipping and
repository PSNR-Y/SSIM-Y metric implementations. V3 additionally uses its
configured 0.1-weight intermediate loss; the table reports its current model
weights because a 0.999 EMA is intentionally too slow to be representative
after only 100 updates.

Run from the repository root:

```powershell
$env:PYTHONPATH=(Resolve-Path '.').Path
python benchmarks/compare_drsformer_v2.py --steps 100 --patch-size 32
```

The checked-in V3 run used the three available Rain200H training pairs and
three test pairs, an RTX 4060 Laptop GPU, PyTorch 2.4.1+cu121 and seed 100.

| Model | Parameters | PSNR-Y | SSIM-Y |
|---|---:|---:|---:|
| DRSformer | 33,655,424 | 19.3273 | 0.5179 |
| LoopDRSformerV2 | 29,695,357 | 19.4985 | **0.5591** |
| LoopDRSformerV3 | 32,004,209 | **19.5774** | 0.5462 |
| V3 change vs. DRSformer | **-4.91%** | **+0.2501 dB** | **+0.0284** |

These values establish that the implementation runs and that this ablation is
promising. They are not publication-level quality claims: 100 optimizer steps
and six paired images are far too small to estimate final Rain200H
generalization. A proper conclusion requires full training and evaluation on
the complete benchmark split.

The completed V3 run later reached 32.0573 PSNR-Y / 0.9298 SSIM-Y, versus the
matched original model's best 32.0762 / 0.9298. This confirms why the short
screen must not be presented as a final result.

To compare raw and EMA weights from one full checkpoint under the same metric
implementation, run:

```powershell
python .\benchmarks\evaluate_checkpoint_variants.py `
  --checkpoint .\experiments\Deraining_LoopDRSformerV3\models\net_g_latest.pth `
  --options .\Options\Deraining_V3.yml
```
