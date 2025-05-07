import torch

def create_model(opt):
    from .pix2pixHD_model import Pix2PixHDModel
    model = Pix2PixHDModel()
    model.initialize(opt)
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model, model.module.optimizer_G, model.module.optimizer_D
