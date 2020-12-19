# C. Benedek and T. Sziranyi, "Change Detection in Optical Aerial Images by a Multilayer Conditional Mixed Markov Model," in IEEE Transactions on Geoscience and Remote Sensing, vol. 47, no. 10, pp. 3416-3430, Oct. 2009, doi: 10.1109/TGRS.2009.2022633.

from ._airchange import _AirChangeDataset


class AC_TiszadobDataset(_AirChangeDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subset='val'
    ):
        super().__init__(root, phase, transforms, repeats, subset)

    @property
    def LOCATION(self):
        return 'Tiszadob'

    @property
    def TRAIN_SAMPLE_IDS(self):
        return (1,2,5,)

    @property
    def VAL_SAMPLE_IDS(self):
        return (4,)

    @property
    def TEST_SAMPLE_IDS(self):
        return (3,)