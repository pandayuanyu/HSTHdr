import cv2 as cv
import matplotlib.pyplot as plt
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Structure_Tensor(nn.Module):
    def __init__(self):
        super(Structure_Tensor, self).__init__()
        self.gradient_X = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            padding_mode='reflect'
        )
        self.X_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 1, 3)
        self.gradient_X.weight.data = self.X_kernel

        self.gradient_Y = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0),
            padding_mode='reflect'
        )
        self.Y_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 3, 1)
        self.gradient_Y.weight.data = self.Y_kernel

    def forward(self, x):
        # 计算灰度图
        r, g, b = x.unbind(dim=-3)
        gray = (0.2989 * r + 0.587 * g + 0.114 * b)
        gray = gray.unsqueeze(dim=-3) * 255.0

        # 计算梯度
        Ix = self.gradient_X(gray)
        Iy = self.gradient_Y(gray)

        Ix2 = torch.pow(Ix, 2)
        Iy2 = torch.pow(Iy, 2)
        Ixy = Ix * Iy

        # 计算行列式和迹
        #  Ix2, Ixy
        #  Ixy, Iy2
        H = Ix2 + Iy2
        K = Ix2 * Iy2 - Ixy * Ixy

        # Flat平坦区域：H = 0;
        # Edge边缘区域：H > 0 & & K = 0;
        # Corner角点区域：H > 0 & & K > 0;

        h_ = 100

        Flat = torch.zeros_like(H)
        Flat[H < h_] = 1.0

        Edge = torch.zeros_like(H)
        Edge[(H >= h_) * (K.abs() <= 1e-6)] = 1.0

        Corner = torch.zeros_like(H)
        Corner[(H >= h_) * (K.abs() > 1e-6)] = 1.0

       # return 1.0 - Flat
        return 1.0 - Flat


test_transform = A.Compose(
        [
            A.ToFloat(max_value=255.0),
            ToTensorV2(p=1.0),
        ],
        additional_targets={
            "image1": "image",
            "image2": "image",
        },
    )

def image_gradient(input, gray=False):
    input_device = input.device
    base_kernel = torch.tensor([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=torch.float32).to(input_device)
    # base_kernel = torch.tensor([[0,  1,  0],
    #                             [1, -4,  1],
    #                             [0,  1,  0]], dtype=torch.float32).to(device)
    if gray:
        conv_op = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(input_device)

        kernel = base_kernel.reshape((1, 1, 3, 3))
        conv_op.weight.data = kernel
        return conv_op(input)
    else:
        conv_op = nn.Conv2d(3, 1, kernel_size=3, bias=False, padding=1).to(input_device)

        kernel = torch.zeros((1, 3, 3, 3), dtype=torch.float32).to(input_device)
        for i in range(3):
            kernel[:, i] = base_kernel
        conv_op.weight.data = kernel
        return conv_op(input)


def gradient_Gray(input):
    input_device = input.device
    conv_op = nn.Conv2d(1, 1, kernel_size=3, bias=False, padding=1).to(input_device)
    kernel = torch.tensor([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=torch.float32).to(input_device)
    # kernel = torch.tensor([[0,  1, 0],
    #                        [1, -4, 1],
    #                        [0,  1, 0]], dtype=torch.float32).to(input_device)
    kernel = kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = kernel

    return conv_op(input)
plt.rcParams['figure.figsize']=(19.405, 14.41)
# 显示图片
def show_numpy(image, name):
    plt.figure(name)
   # plt.figure(figsize=(15.0, 10.0), dpi=100)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    plt.axis('off')  # 关掉坐标轴为off
    plt.savefig("F:\\SwinTransHDR\\ManStanding\\gt_st.png", bbox_inches='tight', pad_inches=0)
  #  plt.title('text title')  # 图像标题
    plt.show()



if __name__ == '__main__':
    ST = Structure_Tensor()

    a = ST.gradient_Y.weight.data
    print(a)

    # 原始图片
    image = cv.imread('F:\\SwinTransHDR\ManStanding\\gt.jpg', cv.IMREAD_UNCHANGED)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    print(image.shape)


    # 转为张量
    img_tensor = test_transform(image=image)['image'].unsqueeze(0)
    img_tensor.requires_grad = True
    print(img_tensor.requires_grad)
    print(img_tensor.shape, img_tensor.max(), img_tensor.min())

    Ix = ST(img_tensor)
    Ix = Ix[0, 0, :, :]
    print(Ix.shape)

   # G = image_gradient(img_tensor, gray=False)
   # G = G[0, 0, :, :]

    img_tensor_num = Ix.detach().numpy()
   # img_tensor_num2 = G.detach().numpy()

    print(img_tensor_num.shape, img_tensor_num.max(), img_tensor_num.min())
    show_numpy(img_tensor_num, name='2')
  # show_numpy(img_tensor_num2, name='1')







