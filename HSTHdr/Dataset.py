import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
import numpy as np
import os
import cv2
import imageio
imageio.plugins.freeimage.download()


class HDR_Dataset(Dataset):
    def __init__(self, dataset_path, is_Training=True, patch_size=256):
        self.dataset_path = os.path.join(dataset_path, 'Training/') if is_Training else os.path.join(dataset_path, 'Test2/second/')
        data_list = os.listdir(self.dataset_path)
        # 通过检查是否有hdr文件来过滤无效文件or文件夹
        self.data_list = [x for x in data_list if os.path.exists(os.path.join(self.dataset_path, x, 'HDRImg.hdr'))]
        self.ToFloat32 = A.ToFloat(max_value=65535.0)
        self.train_transform = A.Compose(
            [
                A.RandomCrop(width=patch_size, height=patch_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(p=1.0),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
            },
        )
        self.test_transform = A.Compose(
            [
                A.CenterCrop(height=512, width=512),
                ToTensorV2(p=1.0),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
            },
        )
        self.transform = self.train_transform if is_Training else self.test_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_root = os.path.join(self.dataset_path, self.data_list[idx] + '/')
        file_list = sorted(os.listdir(data_root))

        # 子文件路径
        image_path1 = os.path.join(data_root, file_list[0])
        image_path2 = os.path.join(data_root, file_list[1])
        label_path = os.path.join(data_root, file_list[2])
        txt_path = os.path.join(data_root, file_list[3])

        # 读取输入的TIFF图像
        image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = self.ToFloat32(image=image1)['image']

        image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = self.ToFloat32(image=image2)['image']

        # 读取曝光时间
        expoTimes = np.power(2, np.loadtxt(txt_path))
        expoTimes = torch.from_numpy(expoTimes)

        # 读取HDR图像
        hdr_image = imageio.imread(label_path, format='HDR-FI')
        hdr_image = np.array(hdr_image)
        hdr_image = hdr_image[:, :, (2, 1, 0)]

        # 数据增强
        augmentations = self.transform(
            image=hdr_image,
            image1=image1,
            image2=image2,
        )
        image1 = augmentations['image1']
        image2 = augmentations['image2']
        hdr_image = augmentations['image']

        # 数据处理1 伽马映射
        H_image1 = utils.Gamma_Correction(image1, gamma=2.2)
        H_image2 = utils.Gamma_Correction(image2, gamma=2.2)

        X1 = torch.cat([image1, H_image1], dim=0)
        X2 = torch.cat([image2, H_image2], dim=0)

        sample = {
            'I1': image1,
            'I2': image2,
            'HDR': hdr_image
        }

        return sample


from args_file import set_args
from torch.utils.data import DataLoader
from structure_tensor import Structure_Tensor
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    args = set_args()

    log_name = 'Flat-_50_1e-6'
    writer = SummaryWriter('./ST/logs/' + log_name)

    HDR_set = HDR_Dataset(
        dataset_path=args.dataset_path,
        is_Training=True,
    )
    HDR_set.transform = A.Compose(
        [
            A.CenterCrop(height=512, width=512),
            ToTensorV2(p=1.0),
        ],
        additional_targets={
            "image1": "image",
            "image2": "image",
        },
    )

    loader = DataLoader(
        dataset=HDR_set,
        batch_size=9,
        num_workers=4,
        shuffle=False
    )

    device = torch.device('cuda')

    ST = Structure_Tensor().to(device)

    for i, sample in enumerate(loader):
        print('=-'*30)
        print('out ', (i+1))
        X1 = sample['X1'].to(device)
        X2 = sample['X2'].to(device)
        H = sample['HDR'].to(device)

        # 处理
        Hu = utils.Mu_Law(H)

        # 计算结构张量
        st_1 = ST(X1)
        st_2 = ST(X2)
        st_h = ST(H)
        st_hu = ST(Hu)

        img_grid_I1 = make_grid(st_1, normalize=False, nrow=3)
        img_grid_I2 = make_grid(st_2, normalize=False, nrow=3)
        img_grid_H = make_grid(st_h, normalize=False, nrow=3)
        img_grid_Hu = make_grid(st_hu, normalize=False, nrow=3)

        writer.add_image('Input1', img_grid_I1, global_step=i)
        writer.add_image('Input2', img_grid_I2, global_step=i)
        writer.add_image('HDR', img_grid_H, global_step=i)
        writer.add_image('HDR_u_law', img_grid_Hu, global_step=i)






















