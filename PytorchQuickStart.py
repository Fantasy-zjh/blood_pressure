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
        ppg_deriv1d = self.cal_deriv(tseq, ppg)
        ppg_deriv2d = self.cal_deriv(tseq, ppg_deriv1d)
        ecg = self.ecgdata[item]
        ecg_deriv1d = self.cal_deriv(tseq, ecg)
        ecg_deriv2d = self.cal_deriv(tseq, ecg_deriv1d)
        ppg_fft = fft(ppg)[:256]
        ecg_fft = fft(ecg)[:256]

        ppg = np.array(ppg, dtype=np.float_).reshape(1, 512)
        ppg_deriv1d = np.array(ppg_deriv1d, dtype=np.float_).reshape(1, 512)
        ppg_deriv2d = np.array(ppg_deriv2d, dtype=np.float_).reshape(1, 512)
        ecg = np.array(ecg, dtype=np.float_).reshape(1, 512)
        ecg_deriv1d = np.array(ecg_deriv1d, dtype=np.float_).reshape(1, 512)
        ecg_deriv2d = np.array(ecg_deriv2d, dtype=np.float_).reshape(1, 512)
        ppg_real = np.array(ppg_fft).real.reshape(1, 256)
        ppg_imag = np.array(ppg_fft).imag.reshape(1, 256)
        ecg_real = np.array(ecg_fft).real.reshape(1, 256)
        ecg_imag = np.array(ecg_fft).imag.reshape(1, 256)

        timeSample = torch.cat(
            [torch.from_numpy(ppg), torch.from_numpy(ppg_deriv1d), torch.from_numpy(ppg_deriv2d), torch.from_numpy(ecg),
             torch.from_numpy(ecg_deriv1d), torch.from_numpy(ecg_deriv2d)], dim=0)
        freqSample = torch.cat([torch.from_numpy(ppg_real), torch.from_numpy(ppg_imag), torch.from_numpy(ecg_real),
                                torch.from_numpy(ecg_imag)], dim=0)
        Y = np.array(self.Ydata[item], dtype=np.float_).reshape(2, 1)
        # print("get item shape:{}".format(Y.shape))
        return timeSample, freqSample, torch.from_numpy(Y)

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
        output = torch.cat([ext(input) for ext in self.Ext1], dim=1)  # (24,512)
        con1x1 = nn.Conv1d(in_channels=24, out_channels=6, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (6,512)
        output = self.BRD(output, "6")
        output = self.Con1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = torch.cat([ext(output) for ext in self.Ext2], dim=1)  # (128,256)
        con1x1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = self.Con2(output)  # (64,128)
        output = self.BRD(output, "64")
        output = torch.cat([ext(output) for ext in self.Ext3], dim=1)  # (256,128)
        con1x1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (64,128)
        output = self.BRD(output, "64")
        output = self.Con3(output)  # (128,64)
        output = self.BRD(output, "128")
        output = torch.cat([ext(output) for ext in self.Ext4], dim=1)  # (512,64)
        con1x1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (128,64)
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
                nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, dilation=i, padding=i * 3, dtype=torch.float64,
                          device=device))
        self.Con1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=9, stride=1, padding=4, dtype=torch.float64,
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
        output = torch.cat([ext(input) for ext in self.Ext1], dim=1)  # (16,256)
        con1x1 = nn.Conv1d(in_channels=16, out_channels=4, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (4,256)
        output = self.BRD(output, "4")
        output = self.Con1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = torch.cat([ext(output) for ext in self.Ext2], dim=1)  # (128,256)
        con1x1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = self.Con2(output)  # (64,128)
        output = self.BRD(output, "64")
        output = torch.cat([ext(output) for ext in self.Ext3], dim=1)  # (256,128)
        con1x1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (64,128)
        output = self.BRD(output, "64")
        output = self.Con3(output)  # (128,64)
        output = self.BRD(output, "128")
        output = torch.cat([ext(output) for ext in self.Ext4], dim=1)  # (512,64)
        con1x1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, dtype=torch.float64, device=device)
        output = con1x1(output)  # (128,64)
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

def train(dataloader, model, loss_fn, optimizer, alpha):
    size = len(dataloader.dataset)
    model.train()
    for batch, (timeSample, freqSample, Ydata) in enumerate(dataloader):
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


##############################################################################
# We also check the model's performance against the test dataset to ensure it is learning.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, SBP_MAE, DBP_MAE = 0, 0, 0
    with torch.no_grad():
        for timeSample, freqSample, Ydata in dataloader:
            timeSample, freqSample, Ydata = timeSample.to(device), freqSample.to(device), Ydata.to(device)
            Lc, Lt, Lf = model(timeSample, freqSample)
            loss_c = loss_fn(Lc, Ydata)
            loss_t = loss_fn(Lt, Ydata)
            loss_f = loss_fn(Lf, Ydata)
            loss = loss_c + alpha * (loss_t + loss_f)
            test_loss += loss.item()
            Ldiff = torch.sum(torch.abs(Lc - Ydata), dim=0)
            DBP_MAE += Ldiff[0].item()
            SBP_MAE += Ldiff[1].item()
    test_loss /= num_batches
    DBP_MAE /= size
    SBP_MAE /= size
    print(f"Test Error: \n DBP: {DBP_MAE:>8f}%, SBP: {SBP_MAE:>8f}, Avg loss: {test_loss:>8f} \n")


##############################################################################
# The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
# parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
# accuracy increase and the loss decrease with every epoch.

# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t + 1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

######################################################################
# Read more about `Training your model <optimization_tutorial.html>`_.
#

######################################################################
# --------------
#

######################################################################
# Saving Models
# -------------
# A common way to save a model is to serialize the internal state dictionary (containing the model parameters).

# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

######################################################################
# Loading Models
# ----------------------------
#
# The process for loading a model includes re-creating the model structure and loading
# the state dictionary into it.

# model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pth"))

#############################################################
# This model can now be used to make predictions.

# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')

######################################################################
# Read more about `Saving & Loading your model <saveloadrun_tutorial.html>`_.
#
if __name__ == "__main__":
    ######################################################################
    # We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
    # automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
    # in the dataloader iterable will return a batch of 64 features and labels.

    batch_size = 100

    training_data = CustomDataset(1)
    validate_data = CustomDataset(2)
    test_data = CustomDataset(3)
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    valid_dataloader = DataLoader(validate_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for timeSample, freqSample, Ydata in test_dataloader:
        print("Shape of timeSample [N, C, H, W]: {}, Shape of freqSample [N, C, H, W]: {}".format(timeSample.shape,
                                                                                                  freqSample.shape))
        print("Shape of y: ", Ydata.shape)
        break

    model = CombinedNetwork()
    # model.to(device)
    model = model.cuda()
    print(model.timeModule.Con1.weight.device)
    #####################################################################
    # Optimizing the Model Parameters
    # ----------------------------------------
    # To train a model, we need a `loss function <https://pytorch.org/docs/stable/nn.html#loss-functions>`_
    # and an `optimizer <https://pytorch.org/docs/stable/optim.html>`_.

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=[0.9, 0.999])
    alpha = 0.2

    epochs = 800
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, alpha)
        test(valid_dataloader, model, loss_fn)  # 用验证集验证
