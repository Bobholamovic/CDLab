from .escnet import ESCNet, ESCNet_Detach, ESCNet_Pixel, ESCNet_SPixel
from .utils import FeatureConverter


def build_escnet(arch, in_ch, out_ch, n_iters, n_spixels, n_filters, alpha, eta_pos, gamma_clr, offsets):
    feat_cvrter = FeatureConverter(eta_pos, gamma_clr, offsets)
    if arch == 'full':
        return ESCNet(feat_cvrter, n_iters, n_spixels, n_filters, in_ch, out_ch, alpha)
    elif arch == 'nospixel':
        return ESCNet_Pixel(feat_cvrter, n_iters, n_spixels, n_filters, in_ch, out_ch, alpha)
    elif arch == 'noe2e':
        return ESCNet_Detach(feat_cvrter, n_iters, n_spixels, n_filters, in_ch, out_ch, alpha)
    elif arch == 'norefine':
        return ESCNet_SPixel(feat_cvrter, n_iters, n_spixels, n_filters, in_ch, out_ch, alpha)
    else:
        raise ValueError