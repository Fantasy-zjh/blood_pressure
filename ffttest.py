from pylab import *
from MIMICData import MIMICHelper
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from SphygmoCorData import SphygmoCorHelper
from scipy import signal

if __name__ == "__main__":
    mimicHelper = MIMICHelper()
    sphygmoCorHelper = SphygmoCorHelper()
    bbp_data, abp_data = sphygmoCorHelper.readSphygmoCorData()
    # resample至125个点
    abp_data_125 = list()
    bbp_data_125 = list()
    for i in range(len(abp_data)):
        abp_125 = signal.resample(abp_data[i], 125).tolist()
        bbp_125 = signal.resample(bbp_data[i], 125).tolist()
        abp_data_125.append(abp_125)
        bbp_data_125.append(bbp_125)

    for k in range(len(bbp_data_125)):
        bbp = bbp_data_125[k]
        N = len(bbp)
        t = np.linspace(0.0, 2 * np.pi, N)
        _fft = fft(bbp)
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
        y_list = []
        nums = [11]  # 4Hz-15Hz
        for num in nums:
            y = _abs[0]
            for i in range(1, num):
                y += _abs[i] * np.cos(i * t + _angle[i])
            y_list.append(y)

        # print("直流信号振幅：" + str(abs / N) + " 相位：" + str(angle))
        # print("直流信号振幅2：" + str((np.sqrt(real ** 2 + imag ** 2)) / N) + " 相位：" + str(np.arctan2(imag, real)))
        # print(fft)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

        # plt.subplot(2, 1, 1)
        plt.title("pic")
        plt.ylabel('P/mmHg')
        plt.plot(t, bbp, label="bbp", color='black')
        for i in range(len(nums)):
            plt.plot(t, y_list[i], label=str(nums[i]))
        plt.legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.show()
