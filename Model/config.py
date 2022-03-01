# -*- coding: utf-8 -*—
# Date: 2022/3/1 0001
# Time: 9:20
# Author: 

import argparse

parse = argparse.ArgumentParser()

# 整体参数
parse.add_argument("--gpu", type=str, help="gpu id",
                   dest="gpu", default="0")
parse.add_argument("--model", type=str, help="The model of VoxelMorph for 1 or 2",
                   dest="model", choices=["vm1", "vm2"], default="vm2")
parse.add_argument("--sim_loss", type=str, help="Image similarity loss: mse or ncc",
                   dest="sim_loss", default="ncc")

# 训练参数
parse.add_argument("--model_dir", type=str, help="The model directory file",
                   dest="model_dir", default="../Checkpoints")
parse.add_argument("--logs_dir", type=str, help="The Logs directory file",
                   dest="logs_dir", default="../Logs")
parse.add_argument("--result_dir", type=str, help="The result directory file",
                   dest="result_dir", default="../Result")
parse.add_argument("--n_iter", type=str, help="The number of iteration",
                   dest="n_iter", default=15000)
parse.add_argument("--lr", type=float, help="The learning rate",
                   dest="lr", default=4e-4)
parse.add_argument("--alpha", type=float, help="The regularization parameter",
                   dest="alpha", default=4.0)
parse.add_argument("--atlas_files", type=str, help="The fixed image files",
                   dest="atlas_files", default="../LPBA40/fixed.nii.gz")
parse.add_argument("--batch_size", type=int, help="The batch size",
                   dest="batch_size", default=1)
parse.add_argument("--train_dir", type=str, help="The training directory file",
                   dest="train_dir", default="../LPBA40/train")
parse.add_argument("--valid_dir", type=str, help="The validation directory files",
                   dest="valid_dir", default="../LPBA40/validate")
parse.add_argument("--n_save_iter", type=int, help="frequency of model saves",
                    dest="n_save_iter", default=15)
# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='./LPBA40/test')
parser.add_argument("--label_dir", type=str, help="label data directory",
                    dest="label_dir", default='./LPBA40/label')
parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default="./Checkpoint/LPBA40.pth")

args = parse.parse_args()


