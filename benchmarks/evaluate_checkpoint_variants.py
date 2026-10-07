"""Evaluate both raw and EMA weights stored in a DRSformer checkpoint."""

import argparse
import json
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from basicsr.models.archs import define_network
from benchmarks.compare_drsformer_v2 import (
    evaluate_model, load_pairs, summarize)


def get_checkpoint_variants(checkpoint):
    """Return available inference state dictionaries in evaluation order."""
    variants = OrderedDict()
    for key in ('params', 'params_ema'):
        state = checkpoint.get(key)
        if isinstance(state, dict) and state:
            variants[key] = state
    if not variants:
        raise KeyError('Checkpoint contains neither params nor params_ema.')
    return variants


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        '--options', type=Path,
        default=REPOSITORY_ROOT / 'Options' / 'Deraining_V3.yml')
    parser.add_argument(
        '--checkpoint', type=Path,
        default=(REPOSITORY_ROOT / 'experiments' /
                 'Deraining_LoopDRSformerV3' / 'models' /
                 'net_g_latest.pth'))
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for full checkpoint evaluation.')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    with args.options.open('r', encoding='utf-8') as option_file:
        options = yaml.safe_load(option_file)
    network_options = options['network_g']
    checkpoint = load_checkpoint(args.checkpoint)
    variants = get_checkpoint_variants(checkpoint)
    test_pairs = load_pairs(args.root, 'test')
    device = torch.device('cuda')

    results = {
        'checkpoint': str(args.checkpoint.resolve()),
        'options': str(args.options.resolve()),
        'test_pairs': len(test_pairs),
        'metrics': 'repository PSNR-Y and SSIM-Y on uint8 images',
    }
    for key, state in variants.items():
        model = define_network(deepcopy(network_options))
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        rows = evaluate_model(model, test_pairs, device)
        results[key] = {
            'mean': summarize(rows),
            'per_image': rows,
        }
        print(f'{key}: {results[key]["mean"]}', flush=True)
        del model
        torch.cuda.empty_cache()

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print('RESULT_JSON=' + json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
