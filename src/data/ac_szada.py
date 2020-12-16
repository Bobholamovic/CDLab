from ._airchange import _AirChangeDataset


class AC_SzadaDataset(_AirChangeDataset):
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
        return 'Szada'

    @property
    def TEST_SAMPLE_IDS(self):
        return (0,)
        
    @property
    def N_PAIRS(self):
        return 7