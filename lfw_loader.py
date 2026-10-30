"""PyTorch Dataset for explicitly supplied LFW image and label paths."""

from pathlib import Path

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
