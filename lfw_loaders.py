"""Implementations shared by the original and compatibility import modules."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor

from lfw_data import select_samples


class LFW(Dataset):
    """Return (RGB image tensor, integer label); custom transforms receive PIL images."""

    def __init__(
        self,
        datapath,
        transform=None,
        *,
        male_file=None,
        female_file=None,
        data_count=500,
        seed=None,
    ):
        self.path_dir = Path(datapath)
        self.transform = transform if transform is not None else PILToTensor()
        self.samples = select_samples(self.path_dir, male_file, female_file, data_count, seed)
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, label = self.samples[index]
        with Image.open(self.path_dir / name) as image:
            return self.transform(image.convert("RGB")), label


class DataLoader:
    """Load explicitly, with all mutable data owned by each instance."""

    def __init__(
        self,
        datapath="easy_trainingdata",
        male_file=None,
        female_file=None,
        data_count=500,
        seed=None,
    ):
        self.dataPath = Path(datapath)
        self.dataCount = data_count
        self.male_file = male_file
        self.female_file = female_file
        self.seed = seed
        self.samples = []
        self.malelable = {}  # Retained spelling for the original API.
        self.femalelabel = {}
        self.images = []
        self.labels = []
        self.labels_onehot = np.empty((0, 2))
        self._split()

    def loadPictensor(self, picFilePath):
        with Image.open(picFilePath) as image:
            return PILToTensor()(image.convert("RGB"))

    def loadLFW_label(self):
        self.samples = select_samples(
            self.dataPath, self.male_file, self.female_file, self.dataCount, self.seed
        )
        self.malelable = {name: label for name, label in self.samples if label == 0}
        self.femalelabel = {name: label for name, label in self.samples if label == 1}

    def print_lable(self):
        for name, label in self.samples:
            print(f"{name}: {label}")

    def loadLFW_facedata(self):
        if not self.samples:
            self.loadLFW_label()
        # Commit together only after all images load; repeated calls replace prior data.
        images = [self.loadPictensor(self.dataPath / name) for name, _ in self.samples]
        self.images = images
        self.labels = [label for _, label in self.samples]
        self.labels_onehot = np.eye(2)[self.labels]
        self._split()

    def _split(self):
        train_end = int(len(self.images) * 0.65)
        validation_end = int(len(self.images) * 0.85)
        for prefix, section in (
            ("train", slice(0, train_end)),
            ("validation", slice(train_end, validation_end)),
            ("test", slice(validation_end, None)),
        ):
            setattr(self, f"{prefix}_images", self.images[section])
            setattr(self, f"{prefix}_labels", self.labels[section])
            setattr(self, f"{prefix}_labels_onehot", self.labels_onehot[section])

    def getTrainData(self):
        return self.train_images, self.train_labels, self.train_labels_onehot

    def getValidationData(self):
        return self.validation_images, self.validation_labels, self.validation_labels_onehot

    def getTestData(self):
        return self.test_images, self.test_labels, self.test_labels_onehot
