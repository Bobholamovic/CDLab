from copy import deepcopy

import numpy as np
import skimage


__all__ = ['CenterCrop', 'Normalize', 'Resize']


def _isseq(x): return isinstance(x, (tuple, list))


class Preprocess:
    def _process(self, x):
        raise NotImplementedError

    def __call__(self, *args, copy=False):
        # NOTE: A Preprocess object deals with 2-D or 3-D numpy ndarrays only, with an optional third dim as the channel dim.
        if copy:
            args = deepcopy(args)
        return self._process(args[0]) if len(args) == 1 else tuple(self._process(x) for x in args)

    def __repr__(self):
        return self.__class__.__name__


class CenterCrop(Preprocess):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size if _isseq(crop_size) else (crop_size, crop_size)

    def _process(self, x):
        h, w = x.shape[:2]

        ch, cw = self.crop_size

        if ch>h or cw>w:
            raise ValueError("Image size is smaller than cropping size.")
        
        offset_up = (h-ch)//2
        offset_left = (w-cw)//2

        return x[offset_up:offset_up+ch, offset_left:offset_left+cw]

    def __repr__(self):
        return super().__repr__()+"\ncrop_size={}".format(self.crop_size)


class Normalize(Preprocess):
    def __init__(self, mu=0.0, sigma=1.0, zscore=False, chn_wise=False):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.zscore = zscore
        self.chn_wise = chn_wise    # Channel_wise

    def _process(self, x):
        if self.zscore:
            if self.chn_wise:
                if x.ndim < 3:
                    raise ValueError("Channel dimension is not found.")
                mu = x.mean((0,1), keepdims=True)
                sigma = x.std((0,1), keepdims=True)
            else:
                mu = x.mean()
                sigma = x.std()
        else:
            mu = self.mu
            sigma = self.sigma
        # dtype of output is determined by x, mu and sigma (when using python float type it is possibly np.float64)
        # Out-of-place operations are used here to enable type promotions
        # This is because we usually do not expect type preservation or data overflow from a *normalize* function
        return (x-mu) / sigma

    def __repr__(self):
        return super().__repr__()+"\nmu={}\nsigma={}\nzscore={}\nchn_wise={}".format(
            self.mu, self.sigma, self.zscore, self.chn_wise
        )


class Resize(Preprocess):
    def __init__(self, size):
        super().__init__()
        self.size = size if _isseq(size) else (size, size)

    def _process(self, x):
        h, w = x.shape[:2]

        nh, nw = self.size

        if nh==h or nw==w:
            return x

        order = 1 if np.issubdtype(x.dtype, np.floating) else 0
        return skimage.transform.resize(x, self.size, order=order, preserve_range=True, anti_aliasing=False).astype(x.dtype)

    def __repr__(self):
        return super().__repr__()+"\nsize={}".format(self.size)