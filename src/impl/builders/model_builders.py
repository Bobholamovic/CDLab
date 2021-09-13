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
    return I2VNet(3)


@MODELS.register_func('CiDL_Dis_model')
def build_CiDL_Dis_model(C):
    from models.drit import Dis, gaussian_weights_init
    model = Dis(
        **C['cidl_model']['dis']
    )
    model.apply(gaussian_weights_init)
    return model


@MODELS.register_func('CiDL_Enc_Con_model')
def build_CiDL_Enc_Con_model(C):
    from models.drit import E_content, gaussian_weights_init
    model = E_content(
        **C['cidl_model']['enc_con']
    )
    model.apply(gaussian_weights_init)
    return model


@MODELS.register_func('CiDL_Enc_Sty_model')
def build_CiDL_Enc_Sty_model(C):
    from models.drit import E_attr, gaussian_weights_init
    model = E_attr(
        **C['cidl_model']['enc_sty']
    )
    model.apply(gaussian_weights_init)
    return model


@MODELS.register_func('CiDL_Dec_model')
def build_CiDL_Dec_model(C):
    from models.drit import G, gaussian_weights_init
    model = G(
        **C['cidl_model']['dec']
    )
    model.apply(gaussian_weights_init)
    return model


@MODELS.register_func('DnD_model')
def build_DnD_model(C):
    from models.dndnet import DnDNet, BaselineModel, DnDNet_align, BaselineModel_align
    if C['dnd_arch'] == 'dnd':
        return DnDNet(**C['dnd_model'])
    elif C['dnd_arch'] == 'baseline':
        return BaselineModel(**C['dnd_model'])
    elif C['dnd_arch'] == 'align':
        return DnDNet_align(**C['dnd_model'])
    elif C['dnd_arch'] == 'baseline_align':
        return BaselineModel_align(**C['dnd_model'])
    else:
        raise ValueError
