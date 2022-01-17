import numpy

from MIMICData import MIMICHelper
import numpy as np
from WaveletDenoising import wavelet_noising
from scipy import signal
import random
from Plt import Plt
import os, sys, shutil
from scipy.stats import pearsonr
from scipy.stats import ttest_rel
import math
import torch
import torch.nn.functional as f
from torch import nn, Tensor
from torch.fft import rfftn, irfftn
from functools import partial
from scipy.fftpack import fft
from FileHelper import FileHelper


def complex_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first two dimensions.
    # Dimensions 3 and higher will have the same shape after multiplication.
    scalar_matmul = partial(torch.einsum, "ab..., cb... -> ac...")

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor. This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet. Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64)
    c.real, c.imag = real, imag
    return c


class fft_conv_1d:

    def __init__(self, kernal: Tensor, bias: Tensor = None, padding: int = 0):
        self.kernal = kernal
        self.bias = bias
        self.padding = padding

    def conv(self,
             signal: Tensor, kernel: Tensor, bias: Tensor = None, padding: int = 0,
             ) -> Tensor:
        """
        Args:
          signal: (Tensor) Input tensor to be convolved with the kernel.
          kernel: (Tensor) Convolution kernel.
          bias: (Optional, Tensor) Bias tensor to add to the output.
          padding: (int) Number of zero samples to pad the input on the last dimension.
        Returns:
          (Tensor) Convolved tensor
        """
        # 1. Pad the input signal & kernel tensors
        signal = f.pad(signal, [padding, padding])
        kernel_padding = [0, signal.size(-1) - kernel.size(-1)]
        padded_kernel = f.pad(kernel, kernel_padding)

        # 2. Perform fourier convolution
        signal_fr = rfftn(signal, dim=-1)
        kernel_fr = rfftn(padded_kernel, dim=-1)

        # 3. Multiply the transformed matrices
        kernel_fr.imag *= -1
        output_fr = complex_matmul(signal_fr, kernel_fr)

        # 4. Compute inverse FFT, and remove extra padded values
        output = irfftn(output_fr, dim=-1)
        output = output[:, :, :signal.size(-1) - kernel.size(-1) + 1]

        # 5. Optionally, add a bias term before returning.
        if bias is not None:
            output += bias.view(1, -1, 1)

        return output

    def __call__(self, input: Tensor):
        return self.conv(input, self.kernal, self.bias, self.padding)


# Time module
class TimeNetwork(nn.Module):
    def __init__(self):
        super(TimeNetwork, self).__init__()
        self.BN = {"6": nn.BatchNorm1d(num_features=6, dtype=torch.float64),
                   "32": nn.BatchNorm1d(num_features=32, dtype=torch.float64),
                   "64": nn.BatchNorm1d(num_features=64, dtype=torch.float64),
                   "128": nn.BatchNorm1d(num_features=128, dtype=torch.float64),
                   "256": nn.BatchNorm1d(num_features=256, dtype=torch.float64)}
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.Ext1 = []
        for i in range(1, 5):
            self.Ext1.append(
                nn.Conv1d(in_channels=6, out_channels=6, kernel_size=7, dilation=i, padding=i * 3, dtype=torch.float64))
        self.Con1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=11, stride=2, padding=5, dtype=torch.float64)
        self.Ext2 = []
        for i in range(1, 5):
            self.Ext2.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64))
        self.Con2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=2, padding=5, dtype=torch.float64)
        self.Ext3 = []
        for i in range(1, 5):
            self.Ext3.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64))
        self.Con3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=2, padding=5,
                              dtype=torch.float64)
        self.Ext4 = []
        for i in range(1, 5):
            self.Ext4.append(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64))
        self.Con4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, stride=2, padding=5,
                              dtype=torch.float64)
        self.FlowCon1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, dtype=torch.float64)
        self.FlowGAP = nn.AdaptiveAvgPool1d(1)
        self.FLowCon2 = nn.Conv1d(in_channels=256, out_channels=2, kernel_size=1, stride=1, dtype=torch.float64)

    def forward(self, input):
        output = torch.cat([ext(input) for ext in self.Ext1], dim=1)  # (24,512)
        con1x1 = nn.Conv1d(in_channels=24, out_channels=6, kernel_size=1, dtype=torch.float64)
        output = con1x1(output)  # (6,512)
        output = self.BRD(output, "6")
        output = self.Con1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = torch.cat([ext(output) for ext in self.Ext2], dim=1)  # (128,256)
        con1x1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, dtype=torch.float64)
        output = con1x1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = self.Con2(output)  # (64,128)
        output = self.BRD(output, "64")
        output = torch.cat([ext(output) for ext in self.Ext3], dim=1)  # (256,128)
        con1x1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dtype=torch.float64)
        output = con1x1(output)  # (64,128)
        output = self.BRD(output, "64")
        output = self.Con3(output)  # (128,64)
        output = self.BRD(output, "128")
        output = torch.cat([ext(output) for ext in self.Ext4], dim=1)  # (512,64)
        con1x1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, dtype=torch.float64)
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
        self.BN = {"2": nn.BatchNorm1d(num_features=2, dtype=torch.float64),
                   "4": nn.BatchNorm1d(num_features=4, dtype=torch.float64),
                   "32": nn.BatchNorm1d(num_features=32, dtype=torch.float64),
                   "64": nn.BatchNorm1d(num_features=64, dtype=torch.float64),
                   "128": nn.BatchNorm1d(num_features=128, dtype=torch.float64),
                   "256": nn.BatchNorm1d(num_features=256, dtype=torch.float64)}
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.Ext1 = []
        for i in range(1, 5):
            self.Ext1.append(
                nn.Conv1d(in_channels=2, out_channels=2, kernel_size=7, dilation=i, padding=i * 3, dtype=torch.float64))
        self.Con1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=7, stride=1, padding=3, dtype=torch.float64)
        self.Ext2 = []
        for i in range(1, 5):
            self.Ext2.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64))
        self.Con2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=2, padding=4, dtype=torch.float64)
        self.Ext3 = []
        for i in range(1, 5):
            self.Ext3.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64))
        self.Con3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=2, padding=4, dtype=torch.float64)
        self.Ext4 = []
        for i in range(1, 5):
            self.Ext4.append(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, dilation=i, padding=i * 3,
                                       dtype=torch.float64))
        self.Con4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9, stride=2, padding=4,
                              dtype=torch.float64)
        self.FlowCon1 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, dtype=torch.float64)
        self.FlowGAP = nn.AdaptiveAvgPool1d(1)
        self.FLowCon2 = nn.Conv1d(in_channels=256, out_channels=2, kernel_size=1, stride=1, dtype=torch.float64)

    def forward(self, input):
        output = torch.cat([ext(input) for ext in self.Ext1], dim=1)  # (8,256)
        con1x1 = nn.Conv1d(in_channels=8, out_channels=2, kernel_size=1, dtype=torch.float64)
        output = con1x1(output)  # (2,256)
        output = self.BRD(output, "2")
        output = self.Con1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = torch.cat([ext(output) for ext in self.Ext2], dim=1)  # (128,256)
        con1x1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1, dtype=torch.float64)
        output = con1x1(output)  # (32,256)
        output = self.BRD(output, "32")
        output = self.Con2(output)  # (64,128)
        output = self.BRD(output, "64")
        output = torch.cat([ext(output) for ext in self.Ext3], dim=1)  # (256,128)
        con1x1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=1, dtype=torch.float64)
        output = con1x1(output)  # (64,128)
        output = self.BRD(output, "64")
        output = self.Con3(output)  # (128,64)
        output = self.BRD(output, "128")
        output = torch.cat([ext(output) for ext in self.Ext4], dim=1)  # (512,64)
        con1x1 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, dtype=torch.float64)
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
        self.BN = {"512": nn.BatchNorm1d(num_features=512, dtype=torch.float64)}
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.Con1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dtype=torch.float64)
        self.Con2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, dtype=torch.float64)
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.Con3 = nn.Conv1d(in_channels=512, out_channels=2, kernel_size=1, stride=1, dtype=torch.float64)

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


class TestClass:

    def __init__(self):
        self.id = 1

    def __getitem__(self, item):
        return "a"

    def __setitem__(self, key, value):
        return "a"


if __name__ == "__main__":
    # module = CombinedNetwork()
    # a = np.array([i for i in range(6 * 512 * 100)], dtype=np.float_).reshape(100, 6, 512)
    # a = torch.from_numpy(a)
    # b = np.array([i for i in range(2 * 256 * 100)], dtype=np.float_).reshape(100, 2, 256)
    # b = torch.from_numpy(b)
    # output, Lt, Lf = module.forward(a, b)
    # print(a.shape)

    abpdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "abp_train.blood")[0]
    print("len of abpdata = {}".format(len(abpdata)))
    fft_data = fft(abpdata)
    abs_data = np.abs(fft_data)
    print("len of abs = {}   type=".format(len(abs_data), type(abs_data)))
    draw_data = abs_data
    Plt.prepare()
    Plt.figure(1)
    Plt.plotLiner(x_data=[index for index in range(len(draw_data))], y_data=draw_data)
    Plt.show()
