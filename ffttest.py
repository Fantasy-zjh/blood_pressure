from pylab import *
from MIMICData import MIMICHelper
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from SphygmoCorData import SphygmoCorHelper
from scipy import signal
from FileHelper import FileHelper

if __name__ == "__main__":
    ppgData = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ppg.blood")
    abpData = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "abp.blood")

    for k in range(len(abpData)):
        abp = abpData[k]
        N = len(abp)
        t = np.linspace(0.0, 2 * np.pi, N)
        _fft = fft(abp)
        # fft10 = [0] * N
        # for i in range(11):
        #     if i == 0:
        #         fft10[i] = fft[i]
        #     else:
        #         fft10[i] = fft[i]
        #         fft10[N - i] = fft[N - i]
        # ifft10 = ifft(fft10)

        _abs = np.abs(_fft)
        _abs = _abs / N * 2
        _abs[0] /= 2
        _angle = np.angle(_fft)

        y = _abs[0]
        for i in range(1, 11):
            y += _abs[i] * np.cos(i * t + _angle[i])

        # print("直流信号振幅：" + str(abs / N) + " 相位：" + str(angle))
        # print("直流信号振幅2：" + str((np.sqrt(real ** 2 + imag ** 2)) / N) + " 相位：" + str(np.arctan2(imag, real)))
        # print(fft)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        # plt.figure(1)
        plt.ylabel('P/mmHg', fontsize=20)
        plt.plot(abp, label="脉搏波", color='blue')
        plt.plot(t, y, label='傅里叶逆变换', color='red')
        plt.legend(loc='upper right', fontsize=20)

        plt.subplot(2, 1, 2)
        # plt.figure(2)
        plt.ylabel('幅值', fontsize=20)
        plt.xlabel('频率/Hz', fontsize=20)
        plt.stem((np.arange(N) / 2 * np.pi)[:125 // 4], _abs[:125 // 4])
        # plt.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.show()
