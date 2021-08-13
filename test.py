import matplotlib.pyplot as plt
from MIMICData import MIMICHelper
import numpy as np

if __name__ == "__main__":
    mimicHelper = MIMICHelper()
    abp_data, ppg_data = mimicHelper.readMIMICData()

    for i in range(len(abp_data)):
        abp = abp_data[i]
        ppg = ppg_data[i]
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

        plt.figure(1)
        plt.ylabel('P/mmHg', fontsize=20)
        plt.plot(np.arange(len(abp)), abp, label="ABP", color='black')
        plt.legend(loc='upper right', fontsize=20)

        plt.figure(2)
        # plt.ylabel('P/mmHg', fontsize=20)
        plt.plot(np.arange(len(ppg)), ppg, label="PPG", color='black')
        plt.legend(loc='upper right', fontsize=20)

        plt.tight_layout()
        plt.show()
