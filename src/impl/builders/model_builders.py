# Custom model builders

from core.misc import MODELS


@MODELS.register_func('UNet_model')
def build_UNet_model(C):
    from models.unet import UNet
    return UNet(6, 2)


@MODELS.register_func('UNet_OSCD_model')
def build_UNet_OSCD_model(C):
    from models.unet import UNet
    return UNet(26, 2)


@MODELS.register_func('SiamUNet_diff_model')
def build_SiamUNet_diff_model(C):
    from models.siamunet_diff import SiamUNet_diff
    return SiamUNet_diff(3, 2)


@MODELS.register_func('SiamUNet_diff_OSCD_model')
def build_SiamUNet_diff_OSCD_model(C):
    from models.siamunet_diff import SiamUNet_diff
    return SiamUNet_diff(13, 2)


@MODELS.register_func('SiamUNet_conc_model')
def build_SiamUNet_conc_model(C):
    from models.siamunet_conc import SiamUNet_conc
    return SiamUNet_conc(3, 2)


@MODELS.register_func('SiamUNet_conc_OSCD_model')
def build_SiamUNet_conc_OSCD_model(C):
    from models.siamunet_conc import SiamUNet_conc
    return SiamUNet_conc(13, 2)


@MODELS.register_func('CDNet_model')
def build_CDNet_model(C):
    from models.cdnet import CDNet
    return CDNet(6, 2)


@MODELS.register_func('CDNet_OSCD_model')
def build_CDNet_OSCD_model(C):
    from models.cdnet import CDNet
    return CDNet(26, 2)


@MODELS.register_func('I2V_model')
def build_I2V_model(C):
    from models.i2v import I2VNet
    return I2VNet(3, 1)