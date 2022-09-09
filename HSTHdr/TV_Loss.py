import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, weight: float = 1) -> None:
        """Total Variation Loss
        Args:
            weight (float): weight of TV loss
        """
        super().__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TV_loss = TVLoss().to(device)
    input = torch.randn(7, 3, 256, 256).to(device)

    out = TV_loss(input)
    print(out)