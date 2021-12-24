from .escnet import ESCNet
from .utils import FeatureConverter


def build_escnet(in_ch, out_ch, n_iters, n_spixels, n_filters, eta_pos, gamma_clr, offsets):
    feat_cvrter = FeatureConverter(eta_pos, gamma_clr, offsets)
    return ESCNet(feat_cvrter, n_iters, n_spixels, n_filters, in_ch, out_ch)