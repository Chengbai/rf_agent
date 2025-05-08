from torch.utils.data import Sampler
import random

from src.episode_dataset import EpisodeDataset


class EpisodeBatchRepeatSampler(Sampler):
    def __init__(self, dataset: EpisodeDataset, batch_size: int, repeats: int):
        assert dataset is not None
        assert batch_size > 0
        assert repeats > 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.repeats = repeats

        self.total_samples = batch_size * repeats * len(dataset)

    def __iter__(self):
        """
        dataset_indices: [0, 1, 2, 3]
        batch_size: 2
        repeats: 3
        expected output indices: [0, 1, 0, 1, 0, 1,
                                  2, 3, 2, 3, 2, 3]
        """
        dataset_indices = list(range(len(self.dataset)))
        random.shuffle(dataset_indices)  # Optional: shuffle within each epoch
        indices = []

        for i in range(0, len(dataset_indices), self.batch_size):
            repeat_batch_indices = (
                dataset_indices[i : i + self.batch_size] * self.repeats
            )
            indices.extend(repeat_batch_indices)

        for idx in indices:
            yield idx

    def __len__(self):
        return self.total_samples
