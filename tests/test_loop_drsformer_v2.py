import unittest
from pathlib import Path

import torch
import yaml

from basicsr.models.archs import define_network


class LoopDRSformerV2Tests(unittest.TestCase):
    @staticmethod
    def _small_v2():
        return define_network({
            'type': 'LoopDRSformerV2',
            'dim': 8,
            'num_blocks': [1, 1, 1, 1],
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.0,
            'latent_pre_blocks': 1,
            'latent_shared_blocks': 1,
            'latent_post_blocks': 1,
            'max_loops': 2,
            'train_loop_range': [2, 2],
            'loop_adapters': True,
            'dynamic_tksa': True,
            'rain_gate': True,
        })

    def test_full_v2_stays_below_original_parameter_budget(self):
        option_path = Path(__file__).parents[1] / 'Options' / 'Deraining_V2.yml'
        with option_path.open('r', encoding='utf-8') as option_file:
            network_options = yaml.safe_load(option_file)['network_g']
        model = define_network(network_options)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(parameter_count, 29_695_357)
        self.assertLess(parameter_count, 33_655_424)

    def test_forward_contract_supports_training_and_inference(self):
        model = self._small_v2()
        input_image = torch.rand(1, 3, 16, 16)

        model.train()
        predictions = model(input_image, force_loops=2)
        model.eval()
        with torch.no_grad():
            output = model(input_image, force_loops=2)

        self.assertEqual(
            ([tuple(item.shape) for item in predictions], tuple(output.shape)),
            ([(1, 3, 16, 16), (1, 3, 16, 16)], (1, 3, 16, 16)))

    def test_dynamic_tksa_exposes_per_head_probability_distributions(self):
        model = self._small_v2()
        model.eval()

        with torch.no_grad():
            details = model(
                torch.rand(1, 3, 16, 16), force_loops=2,
                return_details=True)
        router_weights = details['loop_stats']['router_weights']

        self.assertTrue(
            router_weights.shape == (2, 1, 8, 4)
            and torch.allclose(
                router_weights.sum(dim=-1),
                torch.ones_like(router_weights[..., 0]),
                atol=1e-6))

    def test_rain_gate_reports_soft_gate_for_each_loop(self):
        model = self._small_v2()
        model.eval()

        with torch.no_grad():
            details = model(
                torch.rand(1, 3, 16, 16), force_loops=2,
                return_details=True)
        gate_means = details['loop_stats']['rain_gate_means']

        self.assertTrue(
            gate_means.shape == (2,)
            and torch.all(gate_means > 0)
            and torch.all(gate_means < 1))

    def test_loop_adapters_report_finite_update_for_each_block(self):
        model = self._small_v2()
        model.eval()

        with torch.no_grad():
            details = model(
                torch.rand(1, 3, 16, 16), force_loops=2,
                return_details=True)
        update_norms = details['loop_stats']['adapter_update_norms']

        self.assertTrue(
            update_norms.shape == (2, 1)
            and torch.isfinite(update_norms).all())

    def test_v2_training_yaml_builds_the_declared_network(self):
        option_path = Path(__file__).parents[1] / 'Options' / 'Deraining_V2.yml'
        with option_path.open('r', encoding='utf-8') as option_file:
            options = yaml.safe_load(option_file)

        model = define_network(options['network_g'])

        self.assertEqual(model.__class__.__name__, 'LoopDRSformerV2')

    def test_v2_training_yaml_uses_validated_fixed_two_loop_schedule(self):
        option_path = Path(__file__).parents[1] / 'Options' / 'Deraining_V2.yml'
        with option_path.open('r', encoding='utf-8') as option_file:
            options = yaml.safe_load(option_file)
        network_options = options['network_g']

        self.assertEqual(network_options['max_loops'], 2)
        self.assertEqual(network_options['train_loop_range'], [2, 2])
        self.assertFalse(network_options['intermediate_supervision'])
        self.assertTrue(options['train']['use_grad_clip'])
        self.assertEqual(options['train']['grad_clip_norm'], 1.0)


if __name__ == '__main__':
    unittest.main()
