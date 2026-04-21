import unittest
from unittest import mock

import torch

from tests import TimeTrackingTestCase
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.utils import get_featuresdict_from_dataset


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.zeros((3, 4, 4), dtype=torch.uint8)


class DummyFeatureExtractor(FeatureExtractorBase):
    def __init__(self):
        super(DummyFeatureExtractor, self).__init__("dummy-fe", ["feat"])

    @staticmethod
    def get_provided_features_list():
        return ("feat",)

    @staticmethod
    def get_default_feature_layer_for_metric(metric):
        return "feat"

    @staticmethod
    def can_be_compiled():
        return False

    @staticmethod
    def get_dummy_input_for_compile():
        return None

    def forward(self, input):
        return (input.float(),)


class TestNumWorkersOverride(TimeTrackingTestCase):
    def _get_used_num_workers(self, save_cpu_ram, num_workers, cpu_count):
        dataset = DummyDataset()
        feat_extractor = DummyFeatureExtractor()
        captured = {}

        def fake_dataloader(*args, **kwargs):
            captured["num_workers"] = kwargs["num_workers"]
            return [torch.zeros((1, 3, 4, 4), dtype=torch.uint8)]

        with mock.patch("torch_fidelity.utils.multiprocessing.cpu_count", return_value=cpu_count), mock.patch(
            "torch_fidelity.utils.DataLoader", side_effect=fake_dataloader
        ):
            get_featuresdict_from_dataset(
                dataset,
                feat_extractor,
                batch_size=1,
                cuda=False,
                save_cpu_ram=save_cpu_ram,
                verbose=False,
                num_workers=num_workers,
            )

        return captured["num_workers"]

    def test_override_ignores_save_cpu_ram(self):
        self.assertEqual(self._get_used_num_workers(save_cpu_ram=True, num_workers=3, cpu_count=64), 3)

    def test_default_with_save_cpu_ram_uses_zero_workers(self):
        self.assertEqual(self._get_used_num_workers(save_cpu_ram=True, num_workers=None, cpu_count=64), 0)

    def test_default_without_save_cpu_ram_uses_auto_workers(self):
        self.assertEqual(self._get_used_num_workers(save_cpu_ram=False, num_workers=None, cpu_count=1), 2)

    def test_invalid_num_workers_raises(self):
        with self.assertRaises(ValueError):
            self._get_used_num_workers(save_cpu_ram=False, num_workers=-1, cpu_count=1)


if __name__ == "__main__":
    unittest.main()
