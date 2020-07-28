import copy
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, ann_file, pipeline, data_prefix, test_mode):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    def prepare_train_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_train_data(idx)
        else:
            return self.prepare_test_data(idx)
