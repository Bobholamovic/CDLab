from glob import glob
from os.path import join, basename

import numpy as np

from utils.data_utils.misc import default_loader
from . import CDDataset


class LebedevDataset(CDDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subset='val',
        sets=('real', 'with_shift', 'without_shift')
    ):
        self.sets = sets
        super().__init__(root, phase, transforms, repeats, subset)

    def _read_file_paths(self):
        t1_list, t2_list, tar_list = [], [], []

        for set_ in self.sets:
            # Get subset directory
            if set_ == 'real':
                set_dir = join(self.root, 'Real', 'subset')
            elif set_ == 'with_shift':
                set_dir = join(self.root, 'Model', 'with_shift')
            elif set_ == 'without_shift':
                set_dir = join(self.root, 'Model', 'without_shift')
            else:
                raise RuntimeError("Unrecognized key encountered.")

            pattern = '*.bmp' if (set_ == 'with_shift' and self.subset in ('test', 'val')) else '*.jpg'
            refs = sorted(glob(join(set_dir, self.subset, 'OUT', pattern)))
            t1s = (join(set_dir, self.subset, 'A', basename(ref)) for ref in refs)
            t2s = (join(set_dir, self.subset, 'B', basename(ref)) for ref in refs)

            tar_list.extend(refs)
            t1_list.extend(t1s)
            t2_list.extend(t2s)

        return t1_list, t2_list, tar_list

    def fetch_target(self, target_path):
        # To {0,1}
        return (super().fetch_target(target_path) > 127).astype(np.bool)  