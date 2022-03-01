# HQR_VoxelMorph
The VoxelMorph with pytorch for myself.

----

- `Checkpoint`：存放训练好的模型的文件夹；
- `Log`：存放日志文件的文件夹，记录各个参数下各种loss值的变化；
- `Model`
  - `config.py`：模型配置文件，用来指定学习率、训练次数、loss权重、batch size、保存间隔等，分为三部分，分别是公共参数、训练时参数和测试时参数；
  - `datagenerators.py`：根据图像文件的路径提供数据，使用了torch.utils.data包来实现的；
  - `losses.py`：各种损失函数的计算，包括形变场平滑性损失、MSE、DSC、NCC、CC、雅克比行列式中负值个数等；
  - `model.py`：配准网络（U-Net）和空间变换网络（STN）的实现，并且将两者进行了模块化分离。
- `Result`：存放训练和测试过程中产生的图像数据的文件夹；
- `test.py`：测试代码；
- `train.py`：训练代码。
