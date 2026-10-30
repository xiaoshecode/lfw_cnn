"""Shared label selection for the two LFW loader interfaces."""

import random
from pathlib import Path


def _read_paths(label_file, root):
    """Validate every entry and deduplicate canonical paths before sampling."""
    paths = {}
    for line in label_file.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if not name:
            continue
        relative = Path(name)
        if relative.is_absolute():
            raise ValueError("Image paths must remain within datapath")
        canonical = (root / relative).resolve()
        if not canonical.is_relative_to(root):
            raise ValueError("Image paths must remain within datapath")
        if not canonical.is_file():
            raise FileNotFoundError(canonical)
        # Path equality follows the host platform, including Windows case folding.
        paths.setdefault(canonical, canonical.relative_to(root).as_posix())
    return paths


def select_samples(datapath, male_file=None, female_file=None, data_count=500, seed=None):
    """Select balanced unique relative image paths; label 0/1 follows the source lists."""
    if (
        isinstance(data_count, bool)
        or not isinstance(data_count, int)
        or data_count <= 0
        or data_count % 2
    ):
        raise ValueError("data_count must be a positive even integer")
    root = Path(datapath)
    male_file = Path(male_file) if male_file is not None else root.parent / "male_names.txt"
    female_file = Path(female_file) if female_file is not None else root.parent / "female_names.txt"
    names = [_read_paths(path, root.resolve()) for path in (male_file, female_file)]
    if set(names[0]) & set(names[1]):
        raise ValueError("An image occurs in both label lists")
    count = data_count // 2
    if any(len(group) < count for group in names):
        raise ValueError(f"Each label list must contain at least {count} distinct image paths")
    rng = random.Random(seed)
    samples = [
        (name, label)
        for label, group in enumerate(names)
        for name in rng.sample(list(group.values()), count)
    ]
    rng.shuffle(samples)
    return samples
