# HQR_VoxelMorph
The VoxelMorph with pytorch for myself.

----

- `Checkpoints`：存放训练好的模型的文件夹；
- `Log`：存放日志文件的文件夹，记录各个参数下各种loss值的变化和tensorboard的记录；
- `Model`
  - `config.py`：模型配置文件，用来指定学习率、训练次数、loss权重、batch size、保存间隔等，分为三部分，分别是公共参数、训练时参数和测试时参数；
  - `datagenerators.py`：根据图像文件的路径提供数据，使用了torch.utils.data包来实现的；
  - `losses.py`：各种损失函数的计算，包括形变场平滑性损失、MSE、DSC、NCC、CC、雅克比行列式中负值个数等；
  - `model.py`：配准网络（U-Net）和空间变换网络（STN）的实现，并且将两者进行了模块化分离。
- `Result`：存放训练和测试过程中产生的图像数据的文件夹；
- `train.py`：训练代码，36个测试，4个进行验证。

---
数据及简介
LPBA40数据集使用上面的参数得到同样的训练结果，数据集链接如下：https://pan.baidu.com/s/1RgjlNiTVq75-TI8p9q2bkw ，提取码：jyud 。对原始数据的处理简单，只做了裁剪和灰度值局部归一化。数据集目录结构如下：

- `train`：训练图像数据，包括S11-S36共36个图像，只进行过裁剪到160 \* 192 \* 160 大小，和局部归一化操作；
- `validate`：训练图像数据，包括S37-S40共4个图像，只进行过裁剪到160 \* 192 \* 160 大小，和局部归一化操作；
- `test`：测试图像数据，包括S02-S10共9个图像，只进行过裁剪到160 \* 192 \* 160大小，和局部归一化操作；
- `label`：标签数据，包括S01-S40共40个图像，只进行过裁剪到160 \* 192 \* 160 大小的操作，无归一化；
- `fixed.nii.gz`：即S01图像，作为固定图像。


-----
参考：https://github.com/zuzhiang/VoxelMorph-torch
