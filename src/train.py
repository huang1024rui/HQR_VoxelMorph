# -*- coding: utf-8 -*—
# Date: 2022/3/1 0001
# Time: 9:43
# Author: 

import os
import glob
import SimpleITK as sitk
import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model.config import args
from Model.model import U_Network,SpatialTransformer
from Model.losses import ncc_loss, mse_loss, gradient_loss
from Model.datagenerators import Dataset

def make_dir():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train():
    # 1. 创建文件夹，创建保存文档
    make_dir()
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    log_name = str(args.n_iter) + "_" + str(args.lr) + "_" + str(args.alpha)
    print("The log_name is:{}".format(log_name))
    f = open(os.path.join(args.logs_dir, log_name + '.txt'), 'w')

    # 2. 读取输入图片
    f_img = sitk.ReadImage(args.atlas_files)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # 记录三维图像大小

    # 对输入图像进行重置并转程tensor文件
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0)
    input_fixed = torch.from_numpy(input_fixed).to(device).float()
    print("The shape of the input_fixed is:{}, type is:{}".format(input_fixed.shape, input_fixed.type()))

    # 3. 创建网络
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 16, 16]
    Unet = U_Network(len(vol_size), enc_nf=nf_enc, dec_nf=nf_dec).to(device)
    ST = SpatialTransformer(vol_size).to(device)
    print("The Unet is:{}".format(Unet))
    print("The ST is:{}".format(ST))

    # 4. 优化器和损失函数
    optimizar = torch.optim.Adam(Unet.parameters(), lr=args.lr)
    sim_loss_fn = ncc_loss if args.sim_loss == "ncc" else mse_loss
    grad_loss_fn = gradient_loss
    # 在SummaryWriter上显示
    writer = SummaryWriter('../Logs')

    # 5. 得到训练数据
    train_files = glob.glob(os.path.join(args.train_dir, "*.nii.gz"))
    val_files = glob.glob(os.path.join(args.valid_dir, "*.nii.gz"))
    t_Data = Dataset(train_files)
    v_Data = Dataset(val_files)
    print("The number of train data:{}; The number of validate data:{}".format(len(t_Data), len(v_Data)))
    train_Data = DataLoader(t_Data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    valid_Data = DataLoader(v_Data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    # 6.开始训练
    train_step = 0
    val_step = 0
    for epoch in range(args.n_iter):
        print("-------------Start the {} training-------------".format(epoch))
        # 训练
        Unet.train()
        ST.train()
        for data in train_Data:
            input_moving = data
            input_moving = input_moving.to(device).float()
            # print("The input_moving shape is:{}".format(input_moving.shape))
            # 计算损失
            train_flow_n2f = Unet(input_moving, input_fixed)
            train_m2f = ST(input_moving, train_flow_n2f)
            train_sim_loss = sim_loss_fn(train_m2f, input_fixed)
            train_grad_loss = grad_loss_fn(train_flow_n2f)
            train_loss = train_sim_loss + args.alpha * train_grad_loss
            # 优化器
            optimizar.zero_grad()
            train_loss.backward()
            optimizar.step()

            # 显示loss
            train_step = train_step + 1
            if train_step % args.n_save_iter == 0:
                print("The training epoch is:{}. The sim_loss is:{}, grad_loss is:{} and loss is:{}".format(
                    train_step, train_sim_loss.item(), train_grad_loss.item(), train_loss.item()))
                print("The training epoch is: %d. The loss is:%.3f, sim_loss is:%.3f, grad_loss is:%.3f" % (
                    train_step, train_loss.item(), train_sim_loss.item(), train_grad_loss.item()), file=f)
                writer.add_scalar('The train sim_loss', train_sim_loss.item(), train_step)
                writer.add_scalar('The train grad_loss', train_grad_loss.item(), train_step)
                writer.add_scalar('The train loss', train_loss.item(), train_step)

                # 保存模型
                save_file_name = os.path.join(args.model_dir, 'Train_model_%d.pth' % train_step)
                torch.save(Unet.state_dict(), save_file_name)
                # Save images
                m_name = str(train_step) + "_m.nii.gz"
                m2f_name = str(train_step) + "_m2f.nii.gz"
                save_image(input_moving, f_img, m_name)
                save_image(train_m2f, f_img, m2f_name)
                # print("Training warped images have saved.")

        # 验证开始
        Unet.eval()
        ST.eval()
        total_val_loss = 0
        with torch.no_grad():
            for v_data in valid_Data:
                val_img = v_data
                # print("The val_img shape is:{}, type is:{}".format(val_img.shape,
                #                                                    val_img.type))
                val_img = val_img.to(device).float()
                # 计算损失函数
                val_flow_n2f = Unet(val_img, input_fixed)
                val_m2f = ST(val_img, val_flow_n2f)
                val_sim_loss = sim_loss_fn(val_m2f, input_fixed)
                val_grad_loss = grad_loss_fn(val_flow_n2f)
                val_loss = val_sim_loss + args.alpha * val_grad_loss
                total_val_loss = total_val_loss + val_loss
        # 打印出测试结果
        val_step = val_step + 1
        print("The validation epoch is:{}. The total_val_loss is:{}".format(val_step, total_val_loss.item()))
        print("The validation epoch is:%d. The total_val_loss is:%.5f" % (val_step, total_val_loss.item()), file=f)
        writer.add_scalar('The Validation loss', total_val_loss.item(), val_step)

        # 保存模型
        save_file_name = os.path.join(args.model_dir, 'Validate_model_%d.pth' % val_step)
        torch.save(Unet.state_dict(), save_file_name)
        # Save images
        m_name = str(val_step) + "_m.nii.gz"
        m2f_name = str(val_step) + "_m2f.nii.gz"
        save_image(input_moving, f_img, m_name)
        save_image(val_m2f, f_img, m2f_name)
        print("Validation warped images have saved.")

    f.close()
    writer.close()


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
