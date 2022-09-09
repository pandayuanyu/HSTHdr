import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler, AdamW
from torch.utils.tensorboard import SummaryWriter
import torchvision

from datetime import datetime
import os
from tqdm import tqdm

import utils
from args_file import set_args
from utils import seed_torch
from Dataset import HDR_Dataset
from SwinT2 import AHDRNet
from structure_tensor import Structure_Tensor
from TV_Loss import TVLoss


def main(args):
    # 设定随机种子
    seed_torch(args.seed)

    # 获取计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-当前计算设备：{}".format(torch.cuda.get_device_name(0)))

    # 导入数据集
    train_dataset = HDR_Dataset(
        dataset_path=args.dataset_path,
        is_Training=True,
        patch_size=args.patch_size
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    test_dataset = HDR_Dataset(
        dataset_path=args.dataset_path,
        is_Training=False
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )

    # 构建神经网络
    model = AHDRNet().to(device)
    print("-Train_network构建完成，参数量为： {} ".format(sum(x.numel() for x in model.parameters())))

    # 损失函数和迭代器
    optimizer = AdamW(model.parameters(), args.learning_rate, weight_decay=5E-2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ST = Structure_Tensor().to(device)
   # TV_loss = TVLoss().to(device)
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    print('-损失函数及优化器构建完成')

    start_epoch = 0
    if args.warm_start_path is not None:
        checkpoint = torch.load(args.warm_start_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.param_groups[0]['lr'] = args.learning_rate
        print('--权重载入完成...')
    else:
        utils.initialize_weights(model)
        print('--权重初始化完成...')

    # 训练记录
    train_time = datetime.now().strftime("%m-%d_%H-%M")
    logs_name = train_time + '_epoch{}'.format(args.epochs + start_epoch)
    logs_dir = os.path.join('./logs/', logs_name)
    writer = SummaryWriter(logs_dir)
    print('-日志保存路径：' + logs_dir)
    print('--使用该指令查看训练过程：tensorboard --logdir=./')
    with open(os.path.join(logs_dir, 'info.txt'), 'a') as f:
        f.write(train_time + '\n')
        f.write('=-' * 30 + '\n')
        for arg in vars(args):
            f.write('--' + str(arg) + ':' + str(getattr(args, arg)) + '\n')
        f.write('=-' * 30 + '\n')

    # 保存权重的主路径
    save_path = os.path.join('./checkpoint/', logs_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 开始训练
    print('-开始训练...')
    step = 0  # 参数更新次数
    max_psnr = 0
    for epoch in range(start_epoch, args.epochs + start_epoch):
        loop = tqdm(train_loader)
        for _, sample_batch in enumerate(loop):
            I1 = sample_batch['I1'].to(device)
            I2 = sample_batch['I2'].to(device)
            H = sample_batch['HDR'].to(device)

            # 数据处理
            I1_GC = utils.Gamma_Correction(I1)
            I2_GC = utils.Gamma_Correction(I2)
            I1_ST = ST(I1)
            I2_ST = ST(I2)

            X1 = torch.cat([I1, I1_GC, I1_ST], dim=1)
            X2 = torch.cat([I2, I2_GC, I2_ST], dim=1)

            H_merge = model(X1, X2)
            H_u = utils.Mu_Law(H)
            Hmerge_u = utils.Mu_Law(H_merge)

            optimizer.zero_grad()
            mse_loss_value = MSE_loss(H_u, Hmerge_u)
          #  TV_loss_value = TV_loss(H_merge)
            ST_loss_value = L1_loss(ST(H_u), ST(Hmerge_u))

            total_loss_value = mse_loss_value + 0.003 * ST_loss_value


            total_loss_value.backward()
            optimizer.step()

            loop.set_description(f"Train Epoch [{epoch + 1}/{args.epochs + start_epoch}]")
            loop.set_postfix(
                Total_loss=total_loss_value.item(),
                MSE_loss=mse_loss_value.item(),
                ST_loss=ST_loss_value.item(),
             #   TV_loss=TV_loss_value.item(),
            )

            step += 1

        # 测试环节
        psnr_list = []
        loop1 = tqdm(test_loader)
        for i, sample_batch in enumerate(loop1):
            I1 = sample_batch['I1'].to(device)
            I2 = sample_batch['I2'].to(device)
            H = sample_batch['HDR'].to(device)

            # 数据处理
            I1_GC = utils.Gamma_Correction(I1)
            I2_GC = utils.Gamma_Correction(I2)
            I1_ST = ST(I1)
            I2_ST = ST(I2)

            X1 = torch.cat([I1, I1_GC, I1_ST], dim=1)
            X2 = torch.cat([I2, I2_GC, I2_ST], dim=1)
            with torch.no_grad():
                H_merge = model(X1, X2)

            psnr = utils.PSNR(H, H_merge)
            psnr_list.append(psnr.item())

            if epoch == 0:
                X = torch.cat([X1, X2], dim=0)
                I = X[:, :3, :, :]
                img_grid_I = torchvision.utils.make_grid(
                    I, normalize=False, nrow=3
                )
                writer.add_image('TIFF_Image_{}'.format(i), img_grid_I, global_step=1)

            H_list = torch.cat([H, utils.Mu_Law(H), H_merge, utils.Mu_Law(H_merge)], dim=0)
            img_grid_H = torchvision.utils.make_grid(
                    H_list, normalize=False, nrow=2
            )
            writer.add_image('HDR_Image_{}'.format(i), img_grid_H, global_step=epoch + 1)

            loop1.set_description(f"Test Epoch [{epoch + 1}/{args.epochs + start_epoch}]")

        mean_test_psnr = sum(psnr_list) / len(psnr_list)

        # 学习率记录
        writer.add_scalar('Total_Loss', total_loss_value.item(), global_step=epoch + 1)
        writer.add_scalar('MSE_Loss', mse_loss_value.item(), global_step=epoch + 1)
        writer.add_scalar('ST_Loss', ST_loss_value.item(), global_step=epoch + 1)
      #  writer.add_scalar('TV_Loss', TV_loss_value.item(), global_step=epoch + 1)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch + 1)
        writer.add_scalar('Test_PSNR', mean_test_psnr, global_step=epoch + 1)
        scheduler.step()

        # 保存权重
        if epoch % args.save_frequency == 0 or epoch == (args.epochs + start_epoch - 1):
            save_name = os.path.join(save_path + '/',
                                     'epoch{}_{}.pt'.format(args.epochs + start_epoch, epoch + 1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': (epoch)
                }, save_name
            )
            print('第{}个epoch的权重保存至：'.format(epoch+1) + save_name)

        if max_psnr < mean_test_psnr:
            max_psnr = mean_test_psnr
            save_name = os.path.join(save_path + '/',
                                     'epoch{}_psnr{}.pt'.format(epoch + 1, max_psnr))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': (epoch)
                }, save_name
            )
            print('第{}个epoch的权重保存至：'.format(epoch+1) + save_name)



if __name__ == '__main__':
    args = set_args()
    main(args)

