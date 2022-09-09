import random
import os
import numpy as np
import torch
from torch import nn


# 固定随机种子
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def Mu_Law(image_tensor, mu=5000):
    device = image_tensor.device
    mu = torch.tensor([mu]).to(device)
    return torch.log(1 + image_tensor*mu) / torch.log(1 + mu)


def Gamma_Correction(image_tensor, gamma=2.2):
    return torch.pow(image_tensor, 1.0/gamma)


def PSNR(image, label):
    mse = nn.functional.mse_loss(image, label)
    return 10 * torch.log10(1 / mse)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)