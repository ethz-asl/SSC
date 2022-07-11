from .PALNet import SSC_PALNet
from .DDRNet import SSC_RGBD_DDRNet
from .AICNet import SSC_RGBD_AICNet
from .GRFNet import SSC_RGBD_GRFNet
from .PALNet_ours import SSC_PALNet_Ours


def make_model(modelname, num_classes):
    if modelname == 'palnet':
        return SSC_PALNet(num_classes)
    if modelname == 'ddrnet':
        return SSC_RGBD_DDRNet(num_classes)
    if modelname == 'aicnet':
        return SSC_RGBD_AICNet(num_classes)
    if modelname == 'grfnet':
        return SSC_RGBD_GRFNet(num_classes)
    if modelname == 'palnet_ours':
        return SSC_PALNet_Ours(num_classes)


__all__ = ["make_model"]
