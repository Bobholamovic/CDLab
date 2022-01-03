import random
import math
from functools import wraps
from copy import deepcopy

import numpy as np
import skimage.transform


__all__ = [
    'Compose', 'Choose', 
    'Identity',
    'Scale', 'DiscreteScale', 
    'FlipRotate', 'Flip', 'HorizontalFlip', 'VerticalFlip', 'Rotate', 
    'Crop',
    'Shift', 'XShift', 'YShift',
    'ContrastBrightScale', 'ContrastScale', 'BrightnessShift',
    'AddGaussNoise'
]


rand = random.random
randi = random.randint
choice = random.choice
uniform = random.uniform
# gauss = random.gauss
gauss = random.normalvariate    # This one is thread-safe


def _isseq(x): return isinstance(x, (tuple, list))


class Transform:
    def __init__(self, rand_state=False, prob_apply=1.0):
        self._rand_state = rand_state
        self.prob_apply = prob_apply

    def _transform(self, x, params):
        raise NotImplementedError

    def __call__(self, *args, copy=False):
        # NOTE: A Transform object deals with 2-D or 3-D numpy ndarrays only, with an optional third dim as the channel dim.
        if copy:
            args = deepcopy(args)
        if rand() > self.prob_apply:
            return args[0] if len(args) == 1 else args
        if self._rand_state:
            params = self._get_rand_params()
        else:
            params = None
        return self._transform(args[0], params) if len(args) == 1 else tuple(self._transform(x, params) for x in args)

    def _get_rand_params(self):
        raise NotImplementedError

    def info(self):
        return ""

    def __repr__(self):
        return self.info()+"\nrand_state={}\nprob_apply={}\n".format(self._rand_state, self.prob_apply)


class Compose:
    def __init__(self, *tfs):
        if len(tfs) == 0:
            raise ValueError("The transformation sequence should contain at least one element.")
        self.tfs = tfs

    def __call__(self, *x):
        if len(x) == 1:
            x = x[0]
            for tf in self.tfs: 
                x = tf(x)
        else:
            for tf in self.tfs:
                x = tf(*x)
        return x

    def __repr__(self):
        return "Compose [ "+", ".join(tf.__repr__() for tf in self.tfs)+"]\n"


class Choose:
    def __init__(self, *tfs):
        if len(tfs) < 2:
            raise ValueError("The transformation sequence should contain at least two elements.")
        self.tfs = tfs

    def __call__(self, *x):
        return choice(self.tfs)(*x)

    def __repr__(self):
        return "Choose [ "+", ".join(tf.__repr__() for tf in self.tfs)+"]\n"


class Identity:
    def __call__(self, *x):
        return x if len(x)>0 else x[0]


class Scale(Transform):
    def __init__(self, scale=(0.5, 1.0), prob_apply=1.0):
        super(Scale, self).__init__(rand_state=_isseq(scale), prob_apply=prob_apply)
        if _isseq(scale):
            if len(scale) != 2:
                raise ValueError
            self.scale = tuple(scale)
        else:
            self.scale = scale

    def _transform(self, x, params):
        if self._rand_state:
            scale = params['scale']
        else:
            scale = self.scale
        h, w = x.shape[:2]
        size = (int(h*scale), int(w*scale))
        if size == (h,w):
            return x
        order = 1 if np.issubdtype(x.dtype, np.floating) else 0
        return skimage.transform.resize(x, size, order=order, preserve_range=True, anti_aliasing=False).astype(x.dtype)

    def _get_rand_params(self):
        return {'scale': uniform(*self.scale)}

    def info(self):
        return "Scale\nscaling_factor={}".format(self.scale)
        

class DiscreteScale(Scale):
    def __init__(self, bins=(0.5, 0.75), prob_apply=1.0):
        super(DiscreteScale, self).__init__(scale=(min(bins), max(bins)), prob_apply=prob_apply)
        self.bins = tuple(bins)

    def _get_rand_params(self):
        return {'scale': choice(self.bins)}

    def info(self):
        return "DiscreteScale\nscaling_factors={}".format(self.bins)


class FlipRotate(Transform):
    # Flip or rotate
    _DIRECTIONS = ('ud', 'lr', '90', '180', '270')
    def __init__(self, direction=None, prob_apply=1.0):
        super(FlipRotate, self).__init__(rand_state=(direction is None), prob_apply=prob_apply)
        self.direction = direction

    def _transform(self, x, params):
        if self._rand_state:
            direction = params['direction']
        else:
            direction = self.direction

        if direction == 'ud':
            return np.flip(x, 0)
        elif direction == 'lr':
            return np.flip(x, 1)
        elif direction == '90':
            # Clockwise
            return np.flip(self._T(x), 1)
        elif direction == '180':
            return np.flip(np.flip(x, 0), 1)
        elif direction == '270':
            return np.flip(self._T(x), 0)
        else:
            raise ValueError("Invalid direction")

    def _get_rand_params(self):
        return {'direction': choice(self._DIRECTIONS)}

    @staticmethod
    def _T(x):
        return np.swapaxes(x, 0, 1)

    def info(self):
        return "FlipRotate"


class Flip(FlipRotate):
    _DIRECTIONS = ('ud', 'lr')

    def info(self):
        return "Flip"


class HorizontalFlip(Flip):
    def __init__(self, prob_apply=1.0):
        super(HorizontalFlip, self).__init__(direction='lr', prob_apply=prob_apply)

    def info(self):
        return "HorizontalFlip"


class VerticalFlip(Flip):
    def __init__(self, prob_apply=1.0):
        super(VerticalFlip, self).__init__(direction='ud', prob_apply=prob_apply)
    
    def info(self):
        return "VerticalFlip"


class Rotate(FlipRotate):
    _DIRECTIONS = ('90', '180', '270')

    def info(self):
        return "Rotate"


class Crop(Transform):
    _INNER_BOUNDS = ('bl', 'br', 'tl', 'tr', 't', 'b', 'l', 'r')
    def __init__(self, crop_size=None, bounds=None, prob_apply=1.0):
        _no_bounds = (bounds is None)
        super(Crop, self).__init__(rand_state=_no_bounds, prob_apply=prob_apply)
        if _no_bounds:
            if crop_size is None:
                raise TypeError("crop_size should be specified if bounds is set to None.")
        self.bounds = bounds
        self.crop_size = crop_size if _isseq(crop_size) else (crop_size, crop_size)

    def _transform(self, x, params):
        h, w = x.shape[:2]
        if not self._rand_state:
            bounds = self.bounds
            if bounds == 'bl':
                return x[h//2:,:w//2]
            elif bounds == 'br':
                return x[h//2:,w//2:]
            elif bounds == 'tl':
                return x[:h//2,:w//2]
            elif bounds == 'tr':
                return x[:h//2,w//2:]
            elif bounds == 't':
                return x[:h//2]
            elif bounds == 'b':
                return x[h//2:]
            elif bounds == 'l':
                return x[:,:w//2]
            elif bounds == 'r':
                return x[:,w//2:]
            else:
                left, top, right, lower = bounds
                return x[top:lower, left:right]
        else:
            if self.crop_size > (h, w):
                raise ValueError("Image size is smaller than cropping size.")
            ch, cw = self.crop_size
            if (ch,cw) == (h,w):
                return x
            cx, cy = int((w-cw+1)*params['rel_pos_x']), int((h-ch+1)*params['rel_pos_y'])
            return x[cy:cy+ch, cx:cx+cw]

    def _get_rand_params(self):
        return {'rel_pos_x': rand(),
                'rel_pos_y': rand()}

    def info(self):
        return "Crop\ncrop_size={}\nbounds={}".format(self.crop_size, self.bounds)


class Shift(Transform):
    def __init__(self, xshift=(-0.0625, 0.0625), yshift=(-0.0625, 0.0625), circular=False, prob_apply=1.0):
        super(Shift, self).__init__(rand_state=_isseq(xshift) or _isseq(yshift), prob_apply=prob_apply)

        if _isseq(xshift):
            self.xshift = tuple(xshift)
        else:
            self.xshift = xshift

        if _isseq(yshift):
            self.yshift = tuple(yshift)
        else:
            self.yshift = yshift

        self.circular = circular

    def _transform(self, x, params):
        h, w = x.shape[:2]
        if self._rand_state:
            xshift = params['xshift']
            yshift = params['yshift']
        else:
            xshift = self.xshift
            yshift = self.yshift
        xsh = -int(xshift*w)
        ysh = -int(yshift*h)
        if self.circular:
            # Shift along the x-axis
            x_shifted = np.concatenate((x[:, xsh:], x[:, :xsh]), axis=1)
            # Shift along the y-axis
            x_shifted = np.concatenate((x_shifted[ysh:], x_shifted[:ysh]), axis=0)
        else:
            zeros = np.zeros(x.shape, dtype=x.dtype)
            x1, x2 = (zeros, x) if xsh < 0 else (x, zeros)
            x_shifted = np.concatenate((x1[:, xsh:], x2[:, :xsh]), axis=1)
            x1, x2 = (zeros, x_shifted) if ysh < 0 else (x_shifted, zeros)
            x_shifted = np.concatenate((x1[ysh:], x2[:ysh]), axis=0)

        return x_shifted
        
    def _get_rand_params(self):
        return {'xshift': uniform(*self.xshift) if isinstance(self.xshift, tuple) else self.xshift,
                'yshift': uniform(*self.yshift) if isinstance(self.yshift, tuple) else self.yshift}

    def info(self):
        return "Shift\nxshift={}\nyshift={}".format(self.xshift, self.yshift)


class XShift(Shift):
    def __init__(self, shift=(-0.0625, 0.0625), circular=False, prob_apply=1.0):
        super(XShift, self).__init__(shift, 0.0, circular, prob_apply)


class YShift(Shift):
    def __init__(self, shift=(-0.0625, 0.0625), circular=False, prob_apply=1.0):
        super(YShift, self).__init__(0.0, shift, circular, prob_apply)


# Color jittering and transformation
# Partially refer to https://github.com/albu/albumentations/
class _ValueTransform(Transform):
    def __init__(self, rand_state, prob_apply, limit):
        super(_ValueTransform, self).__init__(rand_state, prob_apply)
        self.limit = limit
        self.limit_range = limit[1] - limit[0]

    @staticmethod
    def keep_range(tf):
        @wraps(tf)
        def wrapper(obj, x, params):
            dtype = x.dtype
            # NOTE: The calculations are done with floating type to prevent overflow.
            # This is a simple yet stupid way.
            if not np.can_cast(dtype, np.float64):
                raise ValueError("dtype={} is not supported.".format(dtype))
            x = x.astype(np.float64, copy=False)
            x = tf(obj, np.clip(x, *obj.limit, out=x), params)
            # Convert back to the original type
            # TODO: Round instead of truncate if dtype is integer
            return np.clip(x, *obj.limit, out=x).astype(dtype, copy=False)
        return wrapper
        

class ContrastBrightScale(_ValueTransform):
    def __init__(self, alpha=(0.2, 0.8), beta=(-0.2, 0.2), prob_apply=1.0, limit=(0, 255)):
        super(ContrastBrightScale, self).__init__(_isseq(alpha) or _isseq(beta), prob_apply, limit)

        if _isseq(alpha):
            self.alpha = tuple(alpha)
        else:
            self.alpha = alpha

        if _isseq(beta):
            self.beta = tuple(beta)
        else:
            self.beta = beta
    
    @_ValueTransform.keep_range
    def _transform(self, x, params):
        alpha = params['alpha'] if self._rand_state else self.alpha
        beta = params['beta'] if self._rand_state else self.beta
        if not math.isclose(alpha, 1.0):
            x *= alpha
        if not math.isclose(beta, 0.0):
            x += beta*np.mean(x)
        return x
    
    def _get_rand_params(self):
        return {'alpha': uniform(*self.alpha) if isinstance(self.alpha, tuple) else self.alpha,
                'beta': uniform(*self.beta) if isinstance(self.beta, tuple) else self.beta}

    def info(self):
        return "ContrastBrightScale\nalpha={}\nbeta={}\nlimit={}".format(self.alpha, self.beta, self.limit)


class ContrastScale(ContrastBrightScale):
    def __init__(self, alpha=(0.2, 0.8), prob_apply=1.0, limit=(0, 255)):
        super(ContrastScale, self).__init__(alpha=alpha, beta=0.0, prob_apply=prob_apply, limit=limit)
        

class BrightnessShift(ContrastBrightScale):
    def __init__(self, beta=(-0.2, 0.2), prob_apply=1.0, limit=(0, 255)):
        super(BrightnessShift, self).__init__(alpha=1.0, beta=beta, prob_apply=prob_apply, limit=limit)
        

class AddGaussNoise(_ValueTransform):
    def __init__(self, mu=0.0, sigma=0.1, prob_apply=1.0, limit=(0, 255)):
        super().__init__(True, prob_apply, limit)
        self.mu = mu
        self.sigma = sigma

    @_ValueTransform.keep_range
    def _transform(self, x, params):
        x += np.random.randn(*x.shape)*self.sigma + self.mu
        return x

    def _get_rand_params(self):
        return {}

    def info(self):
        return "AddGaussNoise\nmu={}\nsigma={}\nlimit={}".format(self.mu, self.sigma, self.limit)