from os.path import basename, splitext

from core.data import DatasetBase
from utils.data_utils.misc import (default_loader, to_tensor)


class CDDataset(DatasetBase):
    def __init__(
        self, 
        root, phase,
        transforms,
        repeats, 
        subset
    ):
        super().__init__(root, phase, transforms, repeats, subset)
        self.transforms = list(self.transforms)
        self.transforms += [None]*(3-len(self.transforms))
        self.t1_list, self.t2_list, self.tar_list = self._read_file_paths()
        self.len = len(self.tar_list)

    def __len__(self):
        return self.len * self.repeats

    def fetch_and_preprocess(self, index):
        t1 = self.fetch_image(self.t1_list[index])
        t2 = self.fetch_image(self.t2_list[index])
        tar = self.fetch_target(self.tar_list[index])
        t1, t2, tar = self.preprocess(t1, t2, tar)
        
        if self.phase == 'train':
            return t1, t2, tar
        else:
            return self.get_name(index), t1, t2, tar

    def _read_file_paths(self):
        raise NotImplementedError
        
    def fetch_target(self, target_path):
        return default_loader(target_path)

    def fetch_image(self, image_path):
        return default_loader(image_path)

    def get_name(self, index):
        return splitext(basename(self.tar_list[index]))[0]+'.png'

    def preprocess(self, t1, t2, tar):
        if self.transforms[0] is not None:
            # Applied to all
            t1, t2, tar = self.transforms[0](t1, t2, tar)
        if self.transforms[1] is not None:
            # Solely for images
            t1, t2 = self.transforms[1](t1, t2)
        if self.transforms[2] is not None:
            # Solely for labels
            tar = self.transforms[2](tar)
        
        return to_tensor(t1).float(), to_tensor(t2).float(), to_tensor(tar).long()