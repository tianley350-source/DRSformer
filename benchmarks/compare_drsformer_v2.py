"""Deterministic short-training comparison for DRSformer and V2.

This benchmark is a development check, not a replacement for full-dataset
training. It gives both models the same crops, optimizer, update count and
repository PSNR-Y/SSIM-Y implementations.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.models.archs.DRSformer_arch import DRSformer, LoopDRSformerV2
from basicsr.utils.img_util import tensor2img


MODEL_SEED = 100
CROP_SCHEDULE_SEED = 20260922
GRAD_CLIP_NORM = 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root', type=Path, default=REPOSITORY_ROOT,
        help='DRSformer repository root.')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--output', type=Path, help='Optional JSON output path.')
    return parser.parse_args()


def read_rgb_tensor(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0


def load_pairs(root, split):
    input_dir = root / 'Datasets' / split / 'Rain200H' / 'input'
    target_dir = root / 'Datasets' / split / 'Rain200H' / 'target'
    pairs = []
    for input_path in sorted(input_dir.glob('*.png')):
        target_path = target_dir / input_path.name
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        pairs.append((
            input_path.name,
            read_rgb_tensor(input_path),
            read_rgb_tensor(target_path)))
    if not pairs:
        raise RuntimeError(f'No paired PNG images found under {input_dir}.')
    return pairs


def make_schedule(pairs, steps, patch_size):
    rng = np.random.default_rng(CROP_SCHEDULE_SEED)
    schedule = []
    for _ in range(steps):
        pair_id = int(rng.integers(0, len(pairs)))
        _, lq, _ = pairs[pair_id]
        height, width = lq.shape[-2:]
        if patch_size > min(height, width):
            raise ValueError(
                f'Patch size {patch_size} exceeds image size {height}x{width}.')
        top = int(rng.integers(0, height - patch_size + 1))
        left = int(rng.integers(0, width - patch_size + 1))
        augment = int(rng.integers(0, 8))
        schedule.append((pair_id, top, left, augment))
    return schedule


def augment(tensor, code):
    if code & 1:
        tensor = torch.flip(tensor, dims=(-1,))
    if code & 2:
        tensor = torch.flip(tensor, dims=(-2,))
    if code & 4:
        tensor = tensor.transpose(-2, -1)
    return tensor.contiguous()


def create_model(model_name):
    common = dict(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias')
    if model_name == 'DRSformer':
        return DRSformer(**common)
    if model_name != 'LoopDRSformerV2':
        raise ValueError(f'Unknown model: {model_name}')
    return LoopDRSformerV2(
        **common,
        latent_pre_blocks=1,
        latent_shared_blocks=4,
        latent_post_blocks=1,
        max_loops=2,
        train_loop_range=[2, 2],
        loop_embedding=True,
        loop_adapters=True,
        input_injection=True,
        layer_scale_init=0.1,
        intermediate_supervision=False,
        dynamic_tksa=True,
        rain_gate=True,
        adaptive_exit=False)


def train_model(model_name, train_pairs, schedule, patch_size, device):
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    random.seed(MODEL_SEED)
    model = create_model(model_name).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4,
        betas=(0.9, 0.999))
    losses = []
    started = time.perf_counter()
    for step, (pair_id, top, left, aug) in enumerate(schedule, 1):
        _, lq, gt = train_pairs[pair_id]
        region = (slice(None), slice(top, top + patch_size),
                  slice(left, left + patch_size))
        lq = augment(lq[region], aug).unsqueeze(0).to(device)
        gt = augment(gt[region], aug).unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(lq)
        if isinstance(prediction, list):
            prediction = prediction[-1]
        loss = F.l1_loss(prediction, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 10 == 0:
            print(
                f'{model_name}: step={step}, loss={losses[-1]:.6f}',
                flush=True)
    return model, losses, time.perf_counter() - started


def pad_to_multiple(tensor, multiple=8):
    height, width = tensor.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    return (F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect'),
            height, width)


def metrics_from_tensors(prediction, target):
    prediction_bgr = tensor2img(prediction, rgb2bgr=True)
    target_bgr = tensor2img(target, rgb2bgr=True)
    return {
        'psnr_y': float(calculate_psnr(
            prediction_bgr, target_bgr, crop_border=0,
            test_y_channel=True)),
        'ssim_y': float(calculate_ssim(
            prediction_bgr, target_bgr, crop_border=0,
            test_y_channel=True)),
    }


def evaluate_model(model, test_pairs, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for name, lq, gt in test_pairs:
            padded, height, width = pad_to_multiple(lq.unsqueeze(0).to(device))
            prediction = model(padded)[..., :height, :width]
            metrics = metrics_from_tensors(prediction, gt.unsqueeze(0))
            rows.append({'image': name, **metrics})
    return rows


def evaluate_input(test_pairs):
    return [
        {'image': name,
         **metrics_from_tensors(lq.unsqueeze(0), gt.unsqueeze(0))}
        for name, lq, gt in test_pairs
    ]


def summarize(rows):
    return {
        'psnr_y': float(np.mean([row['psnr_y'] for row in rows])),
        'ssim_y': float(np.mean([row['ssim_y'] for row in rows])),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark.')
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    train_pairs = load_pairs(args.root, 'train')
    test_pairs = load_pairs(args.root, 'test')
    schedule = make_schedule(train_pairs, args.steps, args.patch_size)
    results = {
        'protocol': {
            'steps': args.steps,
            'patch_size': args.patch_size,
            'train_pairs': len(train_pairs),
            'test_pairs': len(test_pairs),
            'model_seed': MODEL_SEED,
            'crop_schedule_seed': CROP_SCHEDULE_SEED,
            'optimizer': 'AdamW',
            'learning_rate': 3e-4,
            'grad_clip_norm': GRAD_CLIP_NORM,
            'deterministic_cuda': True,
        }
    }
    input_rows = evaluate_input(test_pairs)
    results['RainyInput'] = {
        'per_image': input_rows,
        'mean': summarize(input_rows)}

    for model_name in ('DRSformer', 'LoopDRSformerV2'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, losses, elapsed = train_model(
            model_name, train_pairs, schedule, args.patch_size, device)
        rows = evaluate_model(model, test_pairs, device)
        results[model_name] = {
            'params': sum(parameter.numel()
                          for parameter in model.parameters()),
            'first_loss': losses[0],
            'last_loss': losses[-1],
            'train_seconds': elapsed,
            'peak_cuda_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'per_image': rows,
            'mean': summarize(rows),
        }
        del model
        torch.cuda.empty_cache()

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print('RESULT_JSON=' + json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
