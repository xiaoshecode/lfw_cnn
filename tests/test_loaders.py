import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader as TorchDataLoader

from lfw_data import select_samples
from lfw_loader import LFW
from LFW_pytorch_loader import DataLoader


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for label, group in enumerate(("male", "female")):
            names = [f"{group}{i}.png" for i in range(10)]
            for name in names:
                Image.new("RGB", (4, 3), (label * 255, 0, 0)).save(self.images / name)
            (self.root / f"{group}_names.txt").write_text(
                "\n".join(names) + "\n\n", encoding="utf-8"
            )

    def test_dataset_length_tensor_and_labels(self):
        dataset = LFW(self.images, data_count=20, seed=1)
        self.assertEqual(len(dataset), 20)
        self.assertEqual(len(set(name for name, _ in dataset.samples)), 20)
        for image, label in dataset:
            self.assertEqual(image.dtype, torch.uint8)
            self.assertEqual(tuple(image.shape), (3, 3, 4))
            self.assertEqual(image[0, 0, 0].item(), label * 255)
        images, labels = next(iter(TorchDataLoader(dataset, batch_size=4)))
        self.assertEqual(tuple(images.shape), (4, 3, 3, 4))
        self.assertEqual(len(labels), 4)

    def test_custom_transform_and_seed(self):
        first = LFW(self.images, transform=lambda image: image.size, data_count=10, seed=4)
        second = LFW(self.images, data_count=10, seed=4)
        self.assertEqual(first.samples, second.samples)
        self.assertEqual(first[0][0], (4, 3))

    def test_legacy_split_reload_and_instance_isolation(self):
        first = DataLoader(self.images, data_count=20, seed=1)
        second = DataLoader(self.images, data_count=20, seed=2)
        first.loadLFW_label()
        first.loadLFW_facedata()
        self.assertEqual(
            [
                len(first.getTrainData()[0]),
                len(first.getValidationData()[0]),
                len(first.getTestData()[0]),
            ],
            [13, 4, 3],
        )
        self.assertEqual(second.images, [])
        first.loadLFW_facedata()
        self.assertEqual(len(first.images), 20)
        for image, label, onehot in zip(first.images, first.labels, first.labels_onehot):
            self.assertEqual(image[0, 0, 0].item(), label * 255)
            self.assertEqual(int(onehot.argmax()), label)

    def test_bad_counts_and_insufficient_labels(self):
        for count in (0, 1, -2, 22, True, 2.5):
            with self.assertRaises(ValueError):
                select_samples(self.images, data_count=count)

    def test_missing_file(self):
        (self.images / "male0.png").unlink()
        with self.assertRaises(FileNotFoundError):
            LFW(self.images, data_count=20)

    def test_conflicting_labels(self):
        (self.root / "female_names.txt").write_text("male0.png", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "both"):
            LFW(self.images, data_count=2)

    def test_rejects_paths_outside_dataset(self):
        (self.root / "outside.png").write_bytes(b"not an image")
        (self.root / "male_names.txt").write_text("../outside.png", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "within"):
            LFW(self.images, data_count=2)


if __name__ == "__main__":
    unittest.main()
