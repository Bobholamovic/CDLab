from functools import partial

import numpy as np
from sklearn import metrics


class Meter:
    def __init__(self, callback=None, calc_avg=True):
        super().__init__()
        if callback is not None:
            self.calculate = callback
        self.calc_avg = calc_avg
        self.reset()

    def calculate(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            raise ValueError

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        if self.calc_avg:
            self.avg = 0

    def update(self, *args, n=1):
        self.val = self.calculate(*args)
        self.sum += self.val * n
        self.count += n
        if self.calc_avg:
            self.avg = self.sum / self.count

    def __repr__(self):
        if self.calc_avg:
            return "val: {} avg: {} cnt: {}".format(self.val, self.avg, self.count)
        else:
            return "val: {} cnt: {}".format(self.val, self.count)


# These metrics only for numpy arrays
class Metric(Meter):
    __name__ = 'Metric'
    def __init__(self, n_classes=2, mode='separ', reduction='binary'):
        self._cm = Meter(partial(metrics.confusion_matrix, labels=np.arange(n_classes)), False)
        self.mode = mode
        if reduction == 'binary' and n_classes != 2:
            raise ValueError("Binary reduction only works in 2-class cases.")
        self.reduction = reduction
        super().__init__(None, mode!='accum')
    
    def _calculate_metric(self, cm):
        raise NotImplementedError

    def calculate(self, pred, true, n=1):
        self._cm.update(true.ravel(), pred.ravel())
        if self.mode == 'accum':
            cm = self._cm.sum
        elif self.mode == 'separ':
            cm = self._cm.val
        else:
            raise ValueError("Invalid working mode")

        if self.reduction == 'none':
            # Do not reduce size
            return self._calculate_metric(cm)
        elif self.reduction == 'mean':
            # Macro averaging
            return self._calculate_metric(cm).mean()
        elif self.reduction == 'binary':
            # The pos_class be 1
            return self._calculate_metric(cm)[1]
        else:
            raise ValueError("Invalid reduction type")

    def reset(self):
        super().reset()
        # Reset the confusion matrix
        self._cm.reset()

    def __repr__(self):
        return self.__name__+" "+super().__repr__()


class Precision(Metric):
    __name__ = 'Prec.'
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=0))


class Recall(Metric):
    __name__ = 'Recall'
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=1))


class Accuracy(Metric):
    __name__ = 'OA'
    def __init__(self, n_classes=2, mode='separ'):
        super().__init__(n_classes=n_classes, mode=mode, reduction='none')
        
    def _calculate_metric(self, cm):
        return np.nan_to_num(np.diag(cm).sum()/cm.sum())


class F1Score(Metric):
    __name__ = 'F1'
    def _calculate_metric(self, cm):
        prec = np.nan_to_num(np.diag(cm)/cm.sum(axis=0))
        recall = np.nan_to_num(np.diag(cm)/cm.sum(axis=1))
        return np.nan_to_num(2*(prec*recall) / (prec+recall))