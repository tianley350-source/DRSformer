import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import yaml

from basicsr.models.archs import define_network
from basicsr.models.image_restoration_model import ImageCleanModel


class QualityCurriculumTests(unittest.TestCase):
    @staticmethod
    def _load_options(filename='Deraining_V4.yml'):
        option_path = Path(__file__).parents[1] / 'Options' / filename
        with option_path.open('r', encoding='utf-8') as option_file:
            return yaml.safe_load(option_file)

    @staticmethod
    def _small_network_options():
        return {
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
        }

    @classmethod
    def _small_training_options(cls):
        return {
            'num_gpu': 0,
            'is_train': True,
            'dist': False,
            'network_g': cls._small_network_options(),
            'path': {'pretrain_network_g': None, 'strict_load_g': False},
            'train': {
                'ema_decay': 0,
                'intermediate_weights': [0.1, 1.0],
                'intermediate_weight_schedule': {
                    'type': 'cosine',
                    'start_iter': 0,
                    'end_iter': 100,
                    'initial': 0.1,
                    'final': 0.0,
                    'final_output': 1.0,
                },
                'mixing_augs': {'mixup': False},
                'pixel_opt': {
                    'type': 'L1Loss', 'loss_weight': 1,
                    'reduction': 'mean'},
                'optim_g': {'type': 'Adam', 'lr': 0.0},
                'scheduler': {
                    'type': 'MultiStepLR', 'milestones': [1000],
                    'gamma': 1.0},
                'use_grad_clip': False,
            },
        }

    def test_full_recipe_uses_progressive_context_within_parameter_budget(self):
        options = self._load_options()
        train_data = options['datasets']['train']

        model = define_network(options['network_g'])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertEqual(train_data['gt_sizes'], [128, 160, 192])
        self.assertEqual(sum(train_data['iters']), options['train']['total_iter'])
        self.assertLess(parameter_count, 33_655_424)

    def test_auxiliary_supervision_retires_at_the_end_of_its_schedule(self):
        data = {
            'lq': torch.rand(1, 3, 16, 16),
            'gt': torch.rand(1, 3, 16, 16),
        }
        torch.manual_seed(7)
        early_model = ImageCleanModel(deepcopy(self._small_training_options()))
        early_model.feed_train_data(data)
        early_model.optimize_parameters(current_iter=0)

        torch.manual_seed(7)
        late_model = ImageCleanModel(deepcopy(self._small_training_options()))
        late_model.feed_train_data(data)
        late_model.optimize_parameters(current_iter=100)

        self.assertLess(
            late_model.get_current_log()['l_pix'],
            early_model.get_current_log()['l_pix'])

    def test_finetune_recipe_reuses_ema_with_larger_patches(self):
        options = self._load_options('Deraining_V4_finetune.yml')
        train_data = options['datasets']['train']

        self.assertEqual(options['path']['param_key'], 'params_ema')
        self.assertEqual(options['path']['ema_param_key'], 'params_ema')
        self.assertTrue(options['path']['strict_load_g'])
        self.assertFalse(options['network_g']['intermediate_supervision'])
        self.assertEqual(train_data['gt_sizes'], [160, 192])
        self.assertEqual(sum(train_data['iters']), options['train']['total_iter'])

    def test_finetune_can_initialize_both_models_from_raw_weights(self):
        options = self._small_training_options()
        options['train']['ema_decay'] = 0.9
        source_model = define_network(self._small_network_options())
        raw_state = source_model.state_dict()
        ema_state = {
            name: (value + 1 if torch.is_floating_point(value) else value)
            for name, value in raw_state.items()
        }

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / 'weights.pth'
            torch.save({'params': raw_state, 'params_ema': ema_state},
                       checkpoint_path)
            options['path'].update({
                'pretrain_network_g': str(checkpoint_path),
                'param_key': 'params',
                'ema_param_key': 'params',
                'strict_load_g': True,
            })

            loaded_model = ImageCleanModel(deepcopy(options))
            training_state = loaded_model.get_bare_model(
                loaded_model.net_g).state_dict()
            ema_loaded_state = loaded_model.net_g_ema.state_dict()

        self.assertTrue(all(
            torch.equal(training_state[name], ema_loaded_state[name])
            for name in training_state))


if __name__ == '__main__':
    unittest.main()
