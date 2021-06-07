import os
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal


from MIMICData import MIMICHelper

SPHYGMOCOR_FILE_PATH = "E:\\毕业论文\\blood_data\\data5\\"

class SphygmoCorHelper:
    @staticmethod
    def readSphygmoCorData():
        bbp = []
        abp = []
        for file in os.listdir(SPHYGMOCOR_FILE_PATH):
            path = SPHYGMOCOR_FILE_PATH + file
            with open(path, 'r') as f:
                nums1 = []
                nums2 = []
                for line in f.readlines():
                    line = line.strip()
                    num1, num2 = line.split(' ')
                    nums1.append(float(num1))
                    nums2.append(float(num2))
                new_nums1 = signal.resample(nums1[:], 125)
                new_nums2 = signal.resample(nums2[:], 125)
                bbp.append(new_nums1)
                abp.append(new_nums2)

        return bbp, abp


if __name__ == "__main__":
    # start_time = time.time()
    bbp_data, abp_data = SphygmoCorHelper.readSphygmoCorData()
    # end_time = time.time()
    # print("行：" + str(len(abp_data)))  11808
    # print("列：" + str(len(abp_data[0])))  1000
    # print("读取数据耗时：" + str(end_time - start_time))  45.54142904281616

    for i in range(len(bbp_data)):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(211)
        plt.title('bbp')
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(bbp_data[i], label="bbp")  # t是横坐标，bbp是纵坐标

        # plt.text(t[index_tB] + 0.25, bbp[index_tB] + 1, 'B', ha='center', va='bottom', fontsize=10.5)

        plt.subplot(212)
        plt.title('abp')
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(abp_data[i], label="abp")

        plt.tight_layout()
        plt.legend()
        plt.show()
