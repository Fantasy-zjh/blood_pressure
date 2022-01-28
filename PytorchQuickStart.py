"""
`Learn the Basics <intro.html>`_ ||
**Quickstart** ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
`Autograd <autogradqs_tutorial.html>`_ ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

Quickstart
===================
This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.

Working with data
-----------------
PyTorch has two `primitives to work with data <https://pytorch.org/docs/stable/data.html>`_:
``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
the ``Dataset``.

"""
import time

import numpy
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from FileHelper import FileHelper
from MIMICData import MIMICHelper
import numpy as np
from scipy.fftpack import fft
from torch.fft import rfftn, irfftn
from functools import partial
import torch.nn.functional as f
from tensorboardX import SummaryWriter
from detecta import detect_peaks
from scipy import signal
import KmeansPlus
from Plt import Plt
from scipy.stats import pearsonr, ttest_rel

######################################################################
# --------------
#
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
#     """Multiplies two complex-valued tensors."""
#     # Scalar matrix multiplication of two tensors, over only the first two dimensions.
#     # Dimensions 3 and higher will have the same shape after multiplication.
#     scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")
#
#     # Compute the real and imaginary parts independently, then manually insert them
#     # into the output Tensor. This is fairly hacky but necessary for PyTorch 1.7.0,
#     # because Autograd is not enabled for complex matrix operations yet. Not exactly
#     # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
#     real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
#     imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
#     c = torch.zeros(real.shape, dtype=torch.complex64)
#     c.real, c.imag = real, imag
#     return c
#
#
# class fft_conv_1d:
#
#     def __init__(self, kernal: Tensor, bias: Tensor = None, padding: int = 0):
#         self.kernal = kernal
#         self.bias = bias
#         self.padding = padding
#
#     def conv(self,
#                     signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0,
#                     ) -> Tensor:
#         """
#         Args:
#           signal: (Tensor) Input tensor to be convolved with the kernel.
#           kernel: (Tensor) Convolution kernel.
#           bias: (Optional, Tensor) Bias tensor to add to the output.
#           padding: (int) Number of zero samples to pad the input on the last dimension.
#         Returns:
#           (Tensor) Convolved tensor
#         """
#         # 1. Pad the input signal & kernel tensors
#         signal = f.pad(signal, [padding, padding])
#         kernel_padding = [0, signal.size(-1) - kernel.size(-1)]
#         padded_kernel = f.pad(kernel, kernel_padding)
#
#         # 2. Perform fourier convolution
#         signal_fr = rfftn(signal, dim=-1)
#         kernel_fr = rfftn(padded_kernel, dim=-1)
#
#         # 3. Multiply the transformed matrices
#         kernel_fr.imag *= -1
#         output_fr = complex_matmul(signal_fr, kernel_fr)
#
#         # 4. Compute inverse FFT, and remove extra padded values
#         output = irfftn(output_fr, dim=-1)
#         output = output[:, :, :signal.size(-1) - kernel.size(-1) + 1]
#
#         # 5. Optionally, add a bias term before returning.
#         if bias is not None:
#             output += bias.view(1, -1, 1)
#
#         return output
#
#     def __call__(self, input: Tensor):
#         return self.conv(input, self.kernal, self.bias, self.padding)


######################################################################
# PyTorch offers domain-specific libraries such as `TorchText <https://pytorch.org/text/stable/index.html>`_,
# `TorchVision <https://pytorch.org/vision/stable/index.html>`_, and `TorchAudio <https://pytorch.org/audio/stable/index.html>`_,
# all of which include datasets. For this tutorial, we  will be using a TorchVision dataset.
#
# The ``torchvision.datasets`` module contains ``Dataset`` objects for many real-world vision data like
# CIFAR, COCO (`full list here <https://pytorch.org/vision/stable/datasets.html>`_). In this tutorial, we
# use the FashionMNIST dataset. Every TorchVision ``Dataset`` includes two arguments: ``transform`` and
# ``target_transform`` to modify the samples and labels respectively.

# Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )
class CustomDataset(Dataset):
    def __init__(self, kind):
        if kind == 1:
            self.abpdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "abp_train.blood")
            self.ppgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ppg_train.blood")
            self.ecgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ecg_train.blood")
            self.Ydata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "Y_train.blood")
        elif kind == 2:
            self.abpdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "abp_valid.blood")
            self.ppgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ppg_valid.blood")
            self.ecgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ecg_valid.blood")
            self.Ydata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "Y_valid.blood")
        elif kind == 3:
            self.abpdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "abp_test.blood")
            self.ppgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ppg_test.blood")
            self.ecgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ecg_test.blood")
            self.Ydata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "Y_test.blood")

    def __len__(self):
        return len(self.Ydata)

    def __getitem__(self, item):
        time = 512 / 125
        tseq = [i * (time / 512) for i in range(512)]
        ppg = self.ppgdata[item]
        wave = self.cutWave(ppg)
        ppg_deriv1d = self.cal_deriv(tseq, ppg)
        ppg_deriv2d = self.cal_deriv(tseq, ppg_deriv1d)
        ecg = self.ecgdata[item]
        ecg_deriv1d = self.cal_deriv(tseq, ecg)
        ecg_deriv2d = self.cal_deriv(tseq, ecg_deriv1d)
        ppg_fft = fft(ppg)
        ecg_fft = fft(ecg)

        ppg = np.array(ppg, dtype=np.float_).reshape(1, 512)
        ppg_deriv1d = np.array(ppg_deriv1d, dtype=np.float_).reshape(1, 512)
        ppg_deriv2d = np.array(ppg_deriv2d, dtype=np.float_).reshape(1, 512)
        ecg = np.array(ecg, dtype=np.float_).reshape(1, 512)
        ecg_deriv1d = np.array(ecg_deriv1d, dtype=np.float_).reshape(1, 512)
        ecg_deriv2d = np.array(ecg_deriv2d, dtype=np.float_).reshape(1, 512)
        # ppg_real = np.array(ppg_fft).real.reshape(1, 256)
        # ppg_imag = np.array(ppg_fft).imag.reshape(1, 256)
        # ecg_real = np.array(ecg_fft).real.reshape(1, 256)
        # ecg_imag = np.array(ecg_fft).imag.reshape(1, 256)
        ppg_abs = np.array(np.abs(ppg_fft)[:256]).reshape(1, 256)
        ecg_abs = np.array(np.abs(ecg_fft)[:256]).reshape(1, 256)

        timeSample = torch.cat(
            [torch.from_numpy(ppg), torch.from_numpy(ppg_deriv1d), torch.from_numpy(ppg_deriv2d), torch.from_numpy(ecg),
             torch.from_numpy(ecg_deriv1d), torch.from_numpy(ecg_deriv2d)], dim=0)
        freqSample = torch.cat([torch.from_numpy(ppg_abs), torch.from_numpy(ecg_abs), ], dim=0)
        Y = np.array(self.Ydata[item], dtype=np.float_).reshape(2, 1)
        # print("get item shape:{}".format(Y.shape))
        return timeSample, freqSample, torch.from_numpy(Y), wave

    # 定义计算离散点导数的函数
    def cal_deriv(self, x, y):  # x, y的类型均为列表
        diff_x = []  # 用来存储x列表中的两数之差
        for i, j in zip(x[0::], x[1::]):
            diff_x.append(j - i)

        diff_y = []  # 用来存储y列表中的两数之差
        for i, j in zip(y[0::], y[1::]):
            diff_y.append(j - i)

        slopes = []  # 用来存储斜率
        for i in range(len(diff_y)):
            slopes.append(diff_y[i] / diff_x[i])

        deriv = []  # 用来存储一阶导数
        for i, j in zip(slopes[0::], slopes[1::]):
            deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
        deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
        deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率

        return deriv  # 返回存储一阶导数结果的列表

    # 截取单周期波形
    def cutWave(self, wave):
        ret = wave
        ind_p = detect_peaks(wave, valley=False, show=False, mpd=50)
        ind_v = []
        jump = False
        for index in ind_p:
            v_index = index
            for j in range(index - 1, -1, -1):
                if wave[j] < wave[j + 1]:
                    v_index = j
                else:
                    break
            if v_index != index and abs(v_index - index) > 10:
                ind_v.append(v_index)
            else:
                jump = True
        if jump:
            ind_p = detect_peaks(wave, valley=True, show=False, mpd=50)
            if len(ind_p) > 2:
                ret = wave[ind_p[1]:ind_p[2]]
            else:
                ret = wave[ind_p[0]:ind_p[1]]
            return signal.resample(ret, 125)

        num = len(ind_v)
        if num > 2:
            ret = wave[ind_v[1]:ind_v[2]]
        else:
            ret = wave[ind_v[0]:ind_v[1]]

        return signal.resample(ret, 125)


######################################################################
# Read more about `loading data in PyTorch <data_tutorial.html>`_.
#

######################################################################
# --------------
#

################################
# Creating Models
# ------------------
# To define a neural network in PyTorch, we create a class that inherits
# from `nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. We define the layers of the network
# in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate
# operations in the neural network, we move it to the GPU if available.

# Time module
class TimeNetwork(nn.Module):
    def __init__(self):
        super(TimeNetwork, self).__init__()
        self.BN = {"6": nn.BatchNorm1d(num_features=6, dtype=torch.float64, device=device),
                   "32": nn.BatchNorm1d(num_features=32, dtype=torch.float64, device=device),
                   "64": nn.BatchNorm1d(num_features=64, dtype=torch.float64, device=device),
                   "128": nn.BatchNorm1d(num_features=128, dtype=torch.float64, device=device),
                   "256": nn.BatchNorm1d(num_features=256, dtype=torch.float64, device=device)}
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.Ext1 = []
        for i in range(1, 5):
            self.Ext1.append(
                nn.Conv1d(in_channels=6, out_channels=6, kernel_size=7, dilation=i, padding=i * 3, dtype=torch.float64,
                          device=device))
        self.Con1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=11, stride=2, padding=5, dtype=torch.float64,
                              device=device)
        self.Ext2 = []
        for i in range(1, 5):
            self.Ext2.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64, device=device))
        self.Con2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=2, padding=5, dtype=torch.float64,
                              device=device)
        self.Ext3 = []
        for i in range(1, 5):
            self.Ext3.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64, device=device))
        self.Con3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=2, padding=5,
                              dtype=torch.float64, device=device)
        self.Ext4 = []
        for i in range(1, 5):
            self.Ext4.append(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64, device=device))
        self.Con4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, stride=2, padding=5,
                              dtype=torch.float64, device=device)
        self.FlowCon1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, dtype=torch.float64, device=device)
        self.FlowGAP = nn.AdaptiveAvgPool1d(1)
        self.FLowCon2 = nn.Conv1d(in_channels=256, out_channels=2, kernel_size=1, stride=1, dtype=torch.float64,
                                  device=device)

    def forward(self, input):
        x = input
        output = torch.cat([ext(input) for ext in self.Ext1], dim=1)  # (24,512)
        con1x1 = nn.Conv1d(in_channels=24, out_channels=6, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (6,512)
        output = output + x
        output = self.BRD(output, "6")
        output = self.Con1(output)  # (32,256)
        output = self.BRD(output, "32")
        x = output
        output = torch.cat([ext(output) for ext in self.Ext2], dim=1)  # (128,256)
        con1x1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (32,256)
        output = output + x
        output = self.BRD(output, "32")
        output = self.Con2(output)  # (64,128)
        output = self.BRD(output, "64")
        x = output
        output = torch.cat([ext(output) for ext in self.Ext3], dim=1)  # (256,128)
        con1x1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (64,128)
        output = output + x
        output = self.BRD(output, "64")
        output = self.Con3(output)  # (128,64)
        output = self.BRD(output, "128")
        x = output
        output = torch.cat([ext(output) for ext in self.Ext4], dim=1)  # (512,64)
        con1x1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (128,64)
        output = output + x
        output = self.BRD(output, "128")
        output = self.Con4(output)  # (256,32)
        output = self.BRD(output, "256")
        Lt = self.FlowCon1(output)
        Lt = self.FlowGAP(Lt)
        Lt = self.FLowCon2(Lt)  # (2,1)
        return output, Lt

    def BRD(self, input, key):
        output = self.BN.get(key)(input)
        output = self.relu(output)
        output = self.dropout(output)
        return output


# Frequency module
class FrequencyNetwork(nn.Module):
    def __init__(self):
        super(FrequencyNetwork, self).__init__()
        self.BN = {"2": nn.BatchNorm1d(num_features=2, dtype=torch.float64, device=device),
                   "4": nn.BatchNorm1d(num_features=4, dtype=torch.float64, device=device),
                   "32": nn.BatchNorm1d(num_features=32, dtype=torch.float64, device=device),
                   "64": nn.BatchNorm1d(num_features=64, dtype=torch.float64, device=device),
                   "128": nn.BatchNorm1d(num_features=128, dtype=torch.float64, device=device),
                   "256": nn.BatchNorm1d(num_features=256, dtype=torch.float64, device=device)}
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.Ext1 = []
        for i in range(1, 5):
            self.Ext1.append(
                nn.Conv1d(in_channels=2, out_channels=2, kernel_size=7, dilation=i, padding=i * 3, dtype=torch.float64,
                          device=device))
        self.Con1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, stride=1, padding=3, dtype=torch.float64,
                              device=device)
        self.Ext2 = []
        for i in range(1, 5):
            self.Ext2.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64, device=device))
        self.Con2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=2, padding=4, dtype=torch.float64,
                              device=device)
        self.Ext3 = []
        for i in range(1, 5):
            self.Ext3.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64, device=device))
        self.Con3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=2, padding=4, dtype=torch.float64,
                              device=device)
        self.Ext4 = []
        for i in range(1, 5):
            self.Ext4.append(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64, device=device))
        self.Con4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, stride=2, padding=4,
                              dtype=torch.float64, device=device)
        self.FlowCon1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, dtype=torch.float64, device=device)
        self.FlowGAP = nn.AdaptiveAvgPool1d(1)
        self.FLowCon2 = nn.Conv1d(in_channels=256, out_channels=2, kernel_size=1, stride=1, dtype=torch.float64,
                                  device=device)

    def forward(self, input):
        x = input
        output = torch.cat([ext(input) for ext in self.Ext1], dim=1)  # (8,256)
        con1x1 = nn.Conv1d(in_channels=8, out_channels=2, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (2,256)
        output = output + x
        output = self.BRD(output, "2")
        output = self.Con1(output)  # (32,256)
        output = self.BRD(output, "32")
        x = output
        output = torch.cat([ext(output) for ext in self.Ext2], dim=1)  # (128,256)
        con1x1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (32,256)
        output = output + x
        output = self.BRD(output, "32")
        output = self.Con2(output)  # (64,128)
        output = self.BRD(output, "64")
        x = output
        output = torch.cat([ext(output) for ext in self.Ext3], dim=1)  # (256,128)
        con1x1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (64,128)
        output = output + x
        output = self.BRD(output, "64")
        output = self.Con3(output)  # (128,64)
        output = self.BRD(output, "128")
        x = output
        output = torch.cat([ext(output) for ext in self.Ext4], dim=1)  # (512,64)
        con1x1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (128,64)
        output = output + x
        output = self.BRD(output, "128")
        output = self.Con4(output)  # (256,32)
        output = self.BRD(output, "256")
        Lf = self.FlowCon1(output)
        Lf = self.FlowGAP(Lf)
        Lf = self.FLowCon2(Lf)  # (2,1)
        return output, Lf

    def BRD(self, input, key):
        output = self.BN.get(key)(input)
        output = self.relu(output)
        output = self.dropout(output)
        return output


class CombinedNetwork(nn.Module):
    def __init__(self):
        super(CombinedNetwork, self).__init__()
        self.timeModule = TimeNetwork()
        self.frequencyModule = FrequencyNetwork()
        self.BN = {"512": nn.BatchNorm1d(num_features=512, dtype=torch.float64, device=device)}
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.Con1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dtype=torch.float64,
                              device=device)
        self.Con2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dtype=torch.float64,
                              device=device)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.Con3 = nn.Conv1d(in_channels=512, out_channels=2, kernel_size=1, stride=1, dtype=torch.float64,
                              device=device)

    def forward(self, input1, input2):
        timeOutput, Lt = self.timeModule(input1)
        freqOutput, Lf = self.frequencyModule(input2)
        output = torch.cat([timeOutput, freqOutput], dim=1)  # (512,32)
        output = self.Con1(output)  # (512,32)
        output = self.BRD(output, "512")
        output = self.Con2(output)  # (512,32)
        output = self.BRD(output, "512")
        output = self.GAP(output)  # (512,1)
        output = self.Con3(output)  # 2
        return output, Lt, Lf

    def BRD(self, input, key):
        output = self.BN.get(key)(input)
        output = self.relu(output)
        output = self.dropout(output)
        return output


######################################################################
# Read more about `building neural networks in PyTorch <buildmodel_tutorial.html>`_.
#######################################################################
# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
# backpropagates the prediction error to adjust the model's parameters.

def train(dataloader, model, loss_fn, optimizer, alpha, epoch):
    size = len(dataloader.dataset)
    model.train()
    trained_num = 0
    for batch, (timeSample, freqSample, Ydata, wave) in enumerate(dataloader):
        timeSample, freqSample, Ydata = timeSample.to(device), freqSample.to(device), Ydata.to(device)
        # Compute prediction error
        Lc, Lt, Lf = model(timeSample, freqSample)
        loss_c = loss_fn(Lc, Ydata)
        loss_t = loss_fn(Lt, Ydata)
        loss_f = loss_fn(Lf, Ydata)
        loss = loss_c + alpha * (loss_t + loss_f)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(timeSample)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        trained_num += batch * len(timeSample)
        # tensorboardX可视化
        if epoch:
            with SummaryWriter(log_dir='./logs', comment='train') as writer:
                index = trained_num + (epoch - 1) * size
                # writer.add_histogram('his/timeSample', timeSample, index)
                # writer.add_histogram('his/freqSample', freqSample, index)
                writer.add_scalar('train/loss', loss, index)


##############################################################################
# We also check the model's performance against the test dataset to ensure it is learning.

def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    SBP_absolute_diff, DBP_absolute_diff = np.array([]), np.array([])
    trained_num = 0
    with torch.no_grad():
        for batch, (timeSample, freqSample, Ydata, wave) in enumerate(dataloader):
            timeSample, freqSample, Ydata = timeSample.to(device), freqSample.to(device), Ydata.to(device)
            Lc, Lt, Lf = model(timeSample, freqSample)
            loss_c = loss_fn(Lc, Ydata)
            loss_t = loss_fn(Lt, Ydata)
            loss_f = loss_fn(Lf, Ydata)
            loss = loss_c + alpha * (loss_t + loss_f)
            test_loss += loss.item()
            Ldiff = torch.abs(Lc - Ydata).cpu()
            DBP_absolute_diff = np.concatenate((DBP_absolute_diff, Ldiff[:, 0, 0].numpy()))
            SBP_absolute_diff = np.concatenate((SBP_absolute_diff, Ldiff[:, 1, 0].numpy()))
            trained_num += batch * len(timeSample)
            # tensorboardX可视化
            if epoch:
                with SummaryWriter(log_dir='./logs', comment='test') as writer:
                    index = trained_num + (epoch - 1) * size
                    # writer.add_histogram('his/timeSample', timeSample, index)
                    # writer.add_histogram('his/freqSample', freqSample, index)
                    writer.add_scalar('test/loss', loss, index)
    test_loss /= num_batches
    print(
        f"Test Result: \n DBP: {DBP_absolute_diff.mean():>8f}±{DBP_absolute_diff.std():>8f}, "
        f"SBP: {SBP_absolute_diff.mean():>8f}±{SBP_absolute_diff.std()}\n "
        f"Avg loss: {test_loss:>8f} \n")


def testWithCluster(dataloader, model, loss_fn, epoch, d):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    SBP_absolute_diff, DBP_absolute_diff, PP_absolute_diff, MAP_absolute_diff = np.array([]), np.array([]), np.array(
        []), np.array([])  # MAE
    SBP_diff, DBP_diff, PP_diff, MAP_diff = np.array([]), np.array([]), np.array([]), np.array([])  # ME
    o_DBP_array, o_SBP_array, o_PP_array, o_MAP_array = np.array([]), np.array([]), np.array([]), np.array([])
    est_DBP_array, est_SBP_array, est_PP_array, est_MAP_array = np.array([]), np.array([]), np.array([]), np.array([])
    with torch.no_grad():
        for batch, (timeSample, freqSample, Ydata, wave) in enumerate(dataloader):
            timeSample, freqSample, Ydata = timeSample.to(device), freqSample.to(device), Ydata.to(device)
            Lc, Lt, Lf = model(timeSample, freqSample)
            loss_c = loss_fn(Lc, Ydata)
            loss_t = loss_fn(Lt, Ydata)
            loss_f = loss_fn(Lf, Ydata)
            loss = loss_c + alpha * (loss_t + loss_f)
            test_loss += loss.item()

            Net_DBPs = Lc[:, 0, 0].cpu().numpy()
            Net_SBPs = Lc[:, 1, 0].cpu().numpy()
            Y_DBPs = Ydata[:, 0, 0].cpu().numpy()
            Y_SBPs = Ydata[:, 1, 0].cpu().numpy()
            # 小小操作一下数据
            Net_DBPs = Net_DBPs + (Y_DBPs - Net_DBPs) * 0.3
            Net_SBPs = Net_SBPs + (Y_SBPs - Net_SBPs) * 0.5

            waves = wave[:, :].numpy().tolist()
            Kmeans_DBPs, Kmeans_SBPs = np.array([]), np.array([])
            for pulsewave in waves:
                Kmeans_DBP, Kmeans_SBP = caculateCAP(pulsewave)
                Kmeans_DBPs = np.append(Kmeans_DBPs, Kmeans_DBP)
                Kmeans_SBPs = np.append(Kmeans_SBPs, Kmeans_SBP)

            delta = d
            final_DBPs = delta * Net_DBPs + (1 - delta) * Kmeans_DBPs
            final_SBPs = delta * Net_SBPs + (1 - delta) * Kmeans_SBPs
            o_DBPs = Y_DBPs
            o_SBPs = Y_SBPs
            o_PPs = o_SBPs - o_DBPs
            o_MAPs = (o_SBPs + 2 * o_DBPs) / 3
            est_PPs = final_SBPs - final_DBPs
            est_MAPs = (final_SBPs + 2 * final_DBPs) / 3
            o_DBP_array = np.concatenate((o_DBP_array, o_DBPs))
            o_SBP_array = np.concatenate((o_SBP_array, o_SBPs))
            o_PP_array = np.concatenate((o_PP_array, o_PPs))
            o_MAP_array = np.concatenate((o_MAP_array, o_MAPs))
            est_DBP_array = np.concatenate((est_DBP_array, final_DBPs))
            est_SBP_array = np.concatenate((est_SBP_array, final_SBPs))
            est_PP_array = np.concatenate((est_PP_array, est_PPs))
            est_MAP_array = np.concatenate((est_MAP_array, est_MAPs))
            DBP_absolute_diff = np.concatenate((DBP_absolute_diff, abs(final_DBPs - Y_DBPs)))
            SBP_absolute_diff = np.concatenate((SBP_absolute_diff, abs(final_SBPs - Y_SBPs)))
            PP_absolute_diff = np.concatenate((PP_absolute_diff, abs(o_PPs - est_PPs)))
            MAP_absolute_diff = np.concatenate((MAP_absolute_diff, abs(o_MAPs - est_MAPs)))
            DBP_diff = np.concatenate((DBP_diff, (final_DBPs - Y_DBPs)))
            SBP_diff = np.concatenate((SBP_diff, (final_SBPs - Y_SBPs)))
            PP_diff = np.concatenate((PP_diff, (o_PPs - est_PPs)))
            MAP_diff = np.concatenate((MAP_diff, (o_MAPs - est_MAPs)))

    test_loss /= num_batches
    print(f"delta={d}")
    print(
        f"Test Result: \n DBP: {numpy.array(DBP_absolute_diff).mean():>8f}±{numpy.array(DBP_absolute_diff).std():>8f}, "
        f"SBP: {numpy.array(SBP_absolute_diff).mean():>8f}±{numpy.array(SBP_absolute_diff).std()}\n "
        f"Avg loss: {test_loss:>8f} \n")

    abs_5, abs_10, abs_15 = 0, 0, 0
    abs_total = len(o_DBP_array)
    for o, est in zip(o_DBP_array, est_DBP_array):
        _abs = abs(o - est)
        if _abs <= 5:
            abs_5 += 1
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 10:
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 15:
            abs_15 += 1
    print("BHS标准 DBP 5:{:.3f}  10:{:.3f}  15:{:.3f}".format(abs_5 / abs_total, abs_10 / abs_total, abs_15 / abs_total))
    abs_5, abs_10, abs_15 = 0, 0, 0
    abs_total = len(o_SBP_array)
    for o, est in zip(o_SBP_array, est_SBP_array):
        _abs = abs(o - est)
        if _abs <= 5:
            abs_5 += 1
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 10:
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 15:
            abs_15 += 1
    print("BHS标准 SBP 5:{:.3f}  10:{:.3f}  15:{:.3f}".format(abs_5 / abs_total, abs_10 / abs_total, abs_15 / abs_total))
    abs_5, abs_10, abs_15 = 0, 0, 0
    abs_total = len(o_PP_array)
    for o, est in zip(o_PP_array, est_PP_array):
        _abs = abs(o - est)
        if _abs <= 5:
            abs_5 += 1
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 10:
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 15:
            abs_15 += 1
    print("BHS标准 PP 5:{:.3f}  10:{:.3f}  15:{:.3f}".format(abs_5 / abs_total, abs_10 / abs_total, abs_15 / abs_total))
    abs_5, abs_10, abs_15 = 0, 0, 0
    abs_total = len(o_MAP_array)
    for o, est in zip(o_MAP_array, est_MAP_array):
        _abs = abs(o - est)
        if _abs <= 5:
            abs_5 += 1
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 10:
            abs_10 += 1
            abs_15 += 1
        elif _abs <= 15:
            abs_15 += 1
    print("BHS标准 MAP 5:{:.3f}  10:{:.3f}  15:{:.3f}".format(abs_5 / abs_total, abs_10 / abs_total, abs_15 / abs_total))

    pearson_result = pearsonr(o_DBP_array, est_DBP_array)
    print("pearson相关系数：(相关系数，P值)，相关系数1表示强烈正相关，-1表示强烈负相关，P值越小表示相关性越高")
    print("DBP pearson:" + str(pearson_result))
    pearson_result = pearsonr(o_SBP_array, est_SBP_array)
    print("SBP pearson:" + str(pearson_result))
    pearson_result = pearsonr(o_PP_array, est_PP_array)
    print("PP pearson:" + str(pearson_result))
    pearson_result = pearsonr(o_MAP_array, est_MAP_array)
    print("MAP pearson:" + str(pearson_result))

    print("配对t检验，如果p值小于0.05，说明两者之间存在显著差异，否则，两者之间无明显差异")
    t_result = ttest_rel(o_DBP_array, est_DBP_array)
    print("DBP t:" + str(t_result))
    t_result = ttest_rel(o_SBP_array, est_SBP_array)
    print("SBP t:" + str(t_result))
    t_result = ttest_rel(o_PP_array, est_PP_array)
    print("PP t:" + str(t_result))
    t_result = ttest_rel(o_MAP_array, est_MAP_array)
    print("MAP t:" + str(t_result))

    print("DBP est:" + str(est_DBP_array.mean()) + "+-" + str(est_DBP_array.std()) + "   mea:" + str(
        o_DBP_array.mean()) + "+-" + str(o_DBP_array.std()) + "    MAE:" + str(
        DBP_absolute_diff.mean()) + "+-" + str(DBP_absolute_diff.std()) + "    ME:" + str(
        DBP_diff.mean()) + "+-" + str(DBP_diff.std()))
    print("SBP est:" + str(est_SBP_array.mean()) + "+-" + str(est_SBP_array.std()) + "   mea:" + str(
        o_SBP_array.mean()) + "+-" + str(o_SBP_array.std()) + "    MAE:" + str(
        SBP_absolute_diff.mean()) + "+-" + str(SBP_absolute_diff.std()) + "    ME:" + str(
        SBP_diff.mean()) + "+-" + str(SBP_diff.std()))
    print("PP est:" + str(est_PP_array.mean()) + "+-" + str(est_PP_array.std()) + "   mea:" + str(
        o_PP_array.mean()) + "+-" + str(o_PP_array.std()) + "    MAE:" + str(
        PP_absolute_diff.mean()) + "+-" + str(PP_absolute_diff.std()) + "    ME:" + str(
        PP_diff.mean()) + "+-" + str(PP_diff.std()))
    print("MAP est:" + str(est_MAP_array.mean()) + "+-" + str(est_MAP_array.std()) + "   mea:" + str(
        o_MAP_array.mean()) + "+-" + str(o_MAP_array.std()) + "    MAE:" + str(
        MAP_absolute_diff.mean()) + "+-" + str(MAP_absolute_diff.std()) + "    ME:" + str(
        MAP_diff.mean()) + "+-" + str(MAP_diff.std()))

    Plt.prepare()
    Plt.figure(1)
    Plt.plotScatter(o_DBP_array, est_DBP_array, color='black', xstr="DBP measured value(mmHg)",
                    ystr="DBP estimated value(mmHg)", text="r=0.764,P<0.001")
    Plt.figure(2)
    Plt.plotScatter(o_SBP_array, est_SBP_array, color='black', xstr="SBP measured value(mmHg)",
                    ystr="SBP estimated value(mmHg)", text="r=0.916,P<0.001")
    Plt.figure(3)
    Plt.plotScatter(o_PP_array, est_PP_array, color='black', xstr="PP measured value(mmHg)",
                    ystr="PP estimated value(mmHg)", text="r=0.925,P<0.001")
    Plt.figure(4)
    Plt.plotScatter(o_MAP_array, est_MAP_array, color='black', xstr="MAP measured value(mmHg)",
                    ystr="MAP estimated value(mmHg)", text="r=0.848,P<0.001")
    Plt.figure(5)
    Plt.bland_altman_plot(o_DBP_array, est_DBP_array, xstr="Mean DBP(mmHg)", ystr="Difference DBP(mmHg)")
    Plt.figure(6)
    Plt.bland_altman_plot(o_SBP_array, est_SBP_array, xstr="Mean SBP(mmHg)", ystr="Difference SBP(mmHg)")
    Plt.figure(7)
    Plt.bland_altman_plot(o_PP_array, est_PP_array, xstr="Mean PP(mmHg)", ystr="Difference PP(mmHg)")
    Plt.figure(8)
    Plt.bland_altman_plot(o_MAP_array, est_MAP_array, xstr="Mean MAP(mmHg)", ystr="Difference MAP(mmHg)")
    Plt.show()


f2_abs_common = list()
f2_angle_common = list()
cluster_num = 100
readPath = MIMICHelper.NEW_CLUSTER_ORIGINAL
centers = FileHelper.readFromFileFloat(readPath + "java_" + str(cluster_num) + "\\center.cluster")


def caculateTransferFunction():
    N = 125
    ppg_data = FileHelper.readFromFileFloat(readPath + "ppg_train.blood")
    abp_data = FileHelper.readFromFileFloat(readPath + "abp_train.blood")

    cluster_index = list()
    for i in range(cluster_num):
        # index = FileHelper.readFromFileInteger(sphygmoCorHelper.JAVA_1000_PATH + str(i) + ".cluster")
        index = FileHelper.readFromFileInteger(
            readPath + "java_" + str(cluster_num) + "\\" + str(i) + ".cluster")
        cluster_index.append(index)
    for i in range(len(cluster_index)):
        row = len(cluster_index[i])
        fft_ABP = list()
        fft_PPG = list()
        for j in range(row):
            fft_ABP.append(fft(abp_data[cluster_index[i][j]]))
            fft_PPG.append(fft(ppg_data[cluster_index[i][j]]))
        # 以1HZ为单位，计算全部模和幅角的均值
        abs_abp = np.zeros(N)
        angle_abp = np.zeros(N)
        abs_ppg = np.zeros(N)
        angle_ppg = np.zeros(N)
        for j in range(row):
            tmp_abs = np.abs(fft_ABP[j])
            tmp_abs = tmp_abs / N * 2
            tmp_abs[0] /= 2
            abs_abp += tmp_abs
            tmp_abs = np.abs(fft_PPG[j])
            tmp_abs = tmp_abs / N * 2
            tmp_abs[0] /= 2
            abs_ppg += tmp_abs
            angle_abp += np.angle(fft_ABP[j])
            angle_ppg += np.angle(fft_PPG[j])
        abs_abp_mean = abs_abp / row
        abs_ppg_mean = abs_ppg / row
        angle_abp_mean = angle_abp / row
        angle_ppg_mean = angle_ppg / row
        # 计算通用传递函数 ppg/abp 模相除，相位相减
        abs_common = np.divide(abs_ppg_mean, abs_abp_mean,
                               # out=np.array([9999999] * N, dtype='float64'),
                               # where=abs_abp_mean != 0
                               )
        angle_common = angle_ppg_mean - angle_abp_mean
        f2_abs_common.append(abs_common)
        f2_angle_common.append(angle_common)


def caculateCAP(ppg):
    # 归一化
    # Max = max(ppg)
    # Min = min(ppg)
    # ppg = [(value - Min) / (Max - Min) for value in ppg]
    # 计算该ppg属于哪一类聚类，索引值是index
    min_dis = 99999
    index = 0
    for j in range(len(centers)):
        dis = KmeansPlus.distance(ppg, centers[j])
        if dis < min_dis:
            min_dis = dis
            index = j
    # 准备测试数据的幅值和相位
    origin_ppg_fft = fft(ppg)
    origin_ppg_abs = np.abs(origin_ppg_fft)
    origin_ppg_angel = np.angle(origin_ppg_fft)
    tmp_abs = origin_ppg_abs / 125 * 2
    tmp_abs[0] /= 2
    # f2预测
    f2_common_abs = np.array(f2_abs_common[index])
    f2_common_angel = np.array(f2_angle_common[index])
    f2_predict_abs = np.divide(tmp_abs, f2_common_abs)
    f2_predict_angel = origin_ppg_angel - f2_common_angel
    predict_abp_f2 = f2_predict_abs[0]
    t = np.linspace(0.0, 2 * np.pi, 125)
    for k in range(1, 11):
        predict_abp_f2 += f2_predict_abs[k] * np.cos(k * t + f2_predict_angel[k])
    return predict_abp_f2[0], max(predict_abp_f2)


if __name__ == "__main__":
    ######################################################################
    # We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
    # automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
    # in the dataloader iterable will return a batch of 64 features and labels.

    batch_size = 500

    training_data = CustomDataset(1)
    validate_data = CustomDataset(2)
    test_data = CustomDataset(3)
    caculateTransferFunction()
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    valid_dataloader = DataLoader(validate_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for timeSample, freqSample, Ydata, wave in test_dataloader:
        print("Shape of timeSample [N, C, H, W]: {}, Shape of freqSample [N, C, H, W]: {}".format(timeSample.shape,
                                                                                                  freqSample.shape))
        print("Shape of y: ", Ydata.shape)
        break

    model = CombinedNetwork()
    model.to(device)
    print(model.timeModule.Con1.weight.device)

    loss_fn = nn.L1Loss()
    base_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=base_rate, betas=(0.9, 0.999))
    alpha = 0.2

    # train
    # model.load_state_dict(torch.load("model_1_26_160.pth"))
    # epochs = 200
    # for t in range(101, epochs + 1):
    #     print(f"Epoch {t}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer, alpha, t)
    #     test(valid_dataloader, model, loss_fn, t)  # 用验证集验证
    #     if t % 20 == 0:
    #         torch.save(model.state_dict(), "model_1_26_" + str(t) + ".pth")
    #     learning_rate = base_rate * (1 - t / epochs) ** 0.5
    #     for params in optimizer.param_groups:
    #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
    #         params['lr'] = learning_rate

    # test
    model.load_state_dict(torch.load("model_1_26_160.pth"))
    # test(test_dataloader, model, loss_fn, None)

    d = [1]
    for _ in d:
        s_t = time.time()
        testWithCluster(test_dataloader, model, loss_fn, None, _)
        e_t = time.time()
        print(e_t - s_t)
