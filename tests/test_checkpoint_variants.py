import unittest

import torch

from benchmarks.evaluate_checkpoint_variants import get_checkpoint_variants


class CheckpointVariantTests(unittest.TestCase):
    def test_raw_and_ema_weights_are_both_selected(self):
        checkpoint = {
            'params': {'weight': torch.tensor([1.0])},
            'params_ema': {'weight': torch.tensor([2.0])},
        }

        variants = get_checkpoint_variants(checkpoint)

        self.assertEqual(list(variants), ['params', 'params_ema'])


if __name__ == '__main__':
    unittest.main()
