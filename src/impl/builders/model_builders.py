# Custom model builders

from torchvision.models import vgg
from core.misc import MODELS


@MODELS.register_func('Unet_model')
def build_unet_model(C):
    from models.unet import Unet
    return Unet(6, 2)


@MODELS.register_func('Unet_OSCD_model')
def build_unet_oscd_model(C):
    from models.unet import Unet
    return Unet(26, 2)


@MODELS.register_func('SiamUnet-diff_model')
def build_siamunet_diff_model(C):
    from models.siamunet_diff import SiamUnet_diff
    return SiamUnet_diff(3, 2)


@MODELS.register_func('SiamUnet-diff_OSCD_model')
def build_siamunet_diff_oscd_model(C):
    from models.siamunet_diff import SiamUnet_diff
    return SiamUnet_diff(13, 2)


@MODELS.register_func('SiamUnet-conc_model')
def build_siamunet_conc_model(C):
    from models.siamunet_conc import SiamUnet_conc
    return SiamUnet_conc(3, 2)


@MODELS.register_func('SiamUnet-conc_OSCD_model')
def build_siamunet_conc_oscd_model(C):
    from models.siamunet_conc import SiamUnet_conc
    return SiamUnet_conc(13, 2)


@MODELS.register_func('CDNet_model')
def build_cdnet_model(C):
    from models.cdnet import CDNet
    return CDNet(6, 2)


@MODELS.register_func('P2V_model')
def build_p2v_model(C):
    from models.p2v import P2VNet
    return P2VNet(**C['p2v_model'])


@MODELS.register_func('P2V-2DOnly_model')
def build_p2v_2donly_model(C):
    from models.p2v import P2VNet_2DOnly
    return P2VNet_2DOnly(**C['p2v_model'])


@MODELS.register_func('P2V-NoTemporal_model')
def build_p2v_notemporal_model(C):
    from models.p2v import P2VNet_NoTemporal
    return P2VNet_NoTemporal(**C['p2v_model'])


@MODELS.register_func('P2V-LateFusion_model')
def build_p2v_latefusion_model(C):
    from models.p2v import P2VNet_LateFusion
    return P2VNet_LateFusion(**C['p2v_model'])


@MODELS.register_func('P2V-Halfhalf_model')
def build_p2v_halfhalf_model(C):
    from models.p2v import P2VNet_Halfhalf
    return P2VNet_Halfhalf(**C['p2v_model'])


@MODELS.register_func('P2V-Alter_model')
def build_p2v_alter_model(C):
    from models.p2v import P2VNet_Alter
    return P2VNet_Alter(**C['p2v_model'])


@MODELS.register_func('P2V-Perm_model')
def build_p2v_perm_model(C):
    from models.p2v import P2VNet_Perm
    return P2VNet_Perm(**C['p2v_model'])


@MODELS.register_func('P2V-Decouple_model')
def build_p2v_decouple_model(C):
    from models.p2v import P2VNet_Decouple
    return P2VNet_Decouple(**C['p2v_model'])


@MODELS.register_func('SNUNet_model')
def build_snunet_model(C):
    from models.snunet import SNUNet_ECAM
    return SNUNet_ECAM(3, 2)


@MODELS.register_func('LUNet_model')
def build_lunet_model(C):
    from models.lunet import U_Net
    return U_Net(**C['lunet_model'])


@MODELS.register_func('STANet_model')
def build_stanet_model(C):
    from models.stanet import create_model
    return create_model(**C['stanet_model'])


@MODELS.register_func('IFN_model')
def build_ifn_model(C):
    from models.ifn import vgg16_base, DSIFN
    vgg16_model = vgg16_base()
    for p in vgg16_model.parameters():
        p.requires_grad = False
    return DSIFN(vgg16_model, vgg16_model)
