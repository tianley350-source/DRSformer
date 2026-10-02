import unittest
from pathlib import Path

import torch
import yaml

from basicsr.models.archs import define_network


class LoopDRSformerV3Tests(unittest.TestCase):
    @staticmethod
    def _load_options(filename='Deraining_V3.yml'):
        option_path = Path(__file__).parents[1] / 'Options' / filename
        with option_path.open('r', encoding='utf-8') as option_file:
            return yaml.safe_load(option_file)

    def test_training_configuration_enables_quality_features(self):
        options = self._load_options()

        self.assertEqual(options['network_g']['type'], 'LoopDRSformerV3')
        self.assertEqual(options['network_g']['latent_post_blocks'], 2)
        self.assertTrue(options['network_g']['intermediate_supervision'])
        self.assertEqual(options['train']['intermediate_weights'], [0.1, 1.0])
        self.assertEqual(options['train']['ema_decay'], 0.999)

    def test_full_model_stays_below_original_parameter_budget(self):
        model = define_network(self._load_options()['network_g'])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(parameter_count, 32_004_209)
        self.assertLess(parameter_count, 33_655_424)

    def test_training_exposes_auxiliary_and_final_predictions(self):
        model = define_network({
            'type': 'LoopDRSformerV3',
            'dim': 8,
            'num_blocks': [1, 1, 1, 1],
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.0,
            'latent_pre_blocks': 1,
            'latent_shared_blocks': 1,
            'latent_post_blocks': 2,
            'max_loops': 2,
            'train_loop_range': [2, 2],
            'loop_adapters': True,
            'dynamic_tksa': True,
            'rain_gate': True,
            'intermediate_supervision': True,
        })
        input_image = torch.rand(1, 3, 16, 16)

        model.train()
        predictions = model(input_image, force_loops=2)
        model.eval()
        with torch.no_grad():
            output = model(input_image, force_loops=2)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(tuple(predictions[0].shape), (1, 3, 16, 16))
        self.assertEqual(tuple(predictions[1].shape), (1, 3, 16, 16))
        self.assertEqual(tuple(output.shape), (1, 3, 16, 16))


if __name__ == '__main__':
    unittest.main()
