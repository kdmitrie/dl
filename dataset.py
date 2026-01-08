import numpy as np
from torch.utils.data import Dataset
from typing import Any


def label_to_onehot(label, num_class):
    onehot = np.zeros((num_class))
    if label < num_class:
        onehot[label] = 1
    return onehot


class IndexedDataset(Dataset):
    """Dataset that uses predefined index array"""
    def __init__(self,
                 base_dataset: Dataset,
                 indices: np.ndarray):

        super().__init__()
        self.base_dataset = base_dataset
        self.indices = indices
        self.lazy = base_dataset.lazy

        if not self.lazy:
            self.load()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.base_dataset[self.indices[index]]

    def load(self):
        self.base_dataset.load()

    def unload(self):
        self.base_dataset.unload()

    @property
    def loaded(self):
        return self.base_dataset.loaded


class ShuffledDataset(IndexedDataset):
    """Dataset that shuffles its data"""
    def __init__(self,
                 base_dataset: Dataset,
                 random_seed: int = 42,
                 shuffle: bool = True):
        indices = np.arange(len(base_dataset), dtype=int)
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)
        super().__init__(base_dataset, indices)

    def train_test_split(self, test: float = 0.3) -> \
            tuple[IndexedDataset, IndexedDataset]:

        ll = int(len(self) * (1 - test))
        ind1 = np.arange(ll, dtype=int)
        ind2 = np.arange(ll, len(self), dtype=int)
        return IndexedDataset(self, ind1), IndexedDataset(self, ind2)
