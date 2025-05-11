import numpy as np
import random

from torch.utils.data import Sampler

from src.episode_dataset import EpisodeDataset


class EpisodeBatchRepeatSampler(Sampler):
    def __init__(
        self, dataset: EpisodeDataset, batch_size: int, group_size: int, repeats: int
    ):
        assert dataset is not None
        assert batch_size > 0
        assert group_size > 0
        assert repeats > 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.repeats = repeats

        self.total_samples = len(dataset) * repeats

    def __iter__(self):
        """
        dataset_indices: [0, 1, 2, 3]
        batch_size: 2
        episode_group: 2
        repeats: 3
        expected: [B x G x R]
            - 1. extract B items
            - 2. each item duplicate G times
            - 3. repeat output of 2 R times
        expected output indices: [0C0, 0C1, 1C0, 1C1,
                                  0C0, 0C1, 1C0, 1C1,
                                  0C0, 0C1, 1C0, 1C1,
                                  2C0, 2C1, 3C0, 3C1,
                                  2C0, 2C1, 3C0, 3C1,
                                  2C0, 2C1, 3C0, 3C1,
                                  2C0, 2C1, 3C0, 3C1]
        """
        dataset_indices = list(range(len(self.dataset)))
        # random.shuffle(dataset_indices)  # Optional: shuffle within each epoch
        indices = []

        # Note: for `train`, the `self.dataset` is at size: `config.train_dataset_length x G`!
        # Same to the `test` and `eval`.
        for i in range(0, len(dataset_indices), self.batch_size * self.group_size):
            batch_data_items_indices = np.array(
                dataset_indices[i : i + self.batch_size]
            )
            group_batch_data_items_indices = []
            for dii in batch_data_items_indices:
                group_batch_data_items_indices.extend(
                    range(dii * self.group_size, (dii + 1) * self.group_size, 1)
                )

            repeat_batch_indices = group_batch_data_items_indices * self.repeats
            indices.extend(repeat_batch_indices)

        for idx in indices:
            yield idx

    def __len__(self):
        return self.total_samples
