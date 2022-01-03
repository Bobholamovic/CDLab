# Custom model builders

from core.misc import MODELS


@MODELS.register_func('UNet_model')
def build_unet_model(C):
    from models.unet import UNet
    return UNet(6, 2)


@MODELS.register_func('UNet_OSCD_model')
def build_unet_oscd_model(C):
    from models.unet import UNet
    return UNet(26, 2)


@MODELS.register_func('SiamUNet-diff_model')
def build_siamunet_diff_model(C):
    from models.siamunet_diff import SiamUNet_diff
    return SiamUNet_diff(3, 2)


@MODELS.register_func('SiamUNet-diff_OSCD_model')
def build_siamunet_diff_oscd_model(C):
    from models.siamunet_diff import SiamUNet_diff
    return SiamUNet_diff(13, 2)


@MODELS.register_func('SiamUNet-conc_model')
def build_siamunet_conc_model(C):
    from models.siamunet_conc import SiamUNet_conc
    return SiamUNet_conc(3, 2)


@MODELS.register_func('SiamUNet-conc_OSCD_model')
def build_siamunet_conc_oscd_model(C):
    from models.siamunet_conc import SiamUNet_conc
    return SiamUNet_conc(13, 2)


@MODELS.register_func('CDNet_model')
def build_cdnet_model(C):
    from models.cdnet import CDNet
    return CDNet(6, 2)


@MODELS.register_func('IFN_model')
def build_ifn_model(C):
    from models.ifn import DSIFN
    return DSIFN()


@MODELS.register_func('SNUNet_model')
def build_snunet_model(C):
    from models.snunet import SNUNet
    return SNUNet(3, 2, 32)


@MODELS.register_func('STANet_model')
def build_stanet_model(C):
    from models.stanet import STANet
    return STANet(**C['stanet_model'])


@MODELS.register_func('LUNet_model')
def build_lunet_model(C):
    from models.lunet import LUNet
    return LUNet(3, 2)


@MODELS.register_func('P2V_model')
def build_p2v_model(C):
    from models.p2v import P2VNet
    return P2VNet(**C['p2v_model'])


@MODELS.register_func('DSAMNet_model')
def build_dsamnet_model(C):
    from models.dsamnet import DSAMNet
    return DSAMNet(**C['dsamnet_model'])


@MODELS.register_func('BIT_model')
def build_bit_model(C):
    from models.bit import BIT
    return BIT(**C['bit_model'])