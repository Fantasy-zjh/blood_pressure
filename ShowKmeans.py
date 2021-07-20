from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
from scipy import signal
from SphygmoCorData import SphygmoCorHelper

if __name__ == "__main__":
    mimicHelper = MIMICHelper()
    sphygmoCorHelper = SphygmoCorHelper()

    # 读取原始的未处理的abp和ppg波形
    # abp_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_abp.blood")
    # ppg_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_ppg.blood")
    bbp_data, abp_data = sphygmoCorHelper.readSphygmoCorData()

    # 读取ppg聚类中心波形
    # centers = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_1000_PATH + "center.cluster")
    centers = mimicHelper.readFromFileFloat(sphygmoCorHelper.SPHYGMOCOR_500_PATH + "center.cluster")

    # 读取子类索引
    # cluster_index = list()
    # for i in range(1000):
    #     index = mimicHelper.readFromFileInteger(mimicHelper.MIMIC_ONE_1000_PATH + str(i) + ".cluster")
    #     cluster_index.append(index)
    cluster_index = list()
    for i in range(500):
        index = mimicHelper.readFromFileInteger(sphygmoCorHelper.SPHYGMOCOR_500_PATH + str(i) + ".cluster")
        cluster_index.append(index)

    # resample至125个点
    # abp_data_125 = list()
    # ppg_data_125 = list()
    # for i in range(len(ppg_data)):
    #     abp_125 = signal.resample(abp_data[i], 125).tolist()
    #     ppg_125 = signal.resample(ppg_data[i], 125).tolist()
    #     abp_data_125.append(abp_125)
    #     ppg_data_125.append(ppg_125)
    abp_data_125 = list()
    bbp_data_125 = list()
    for i in range(len(abp_data)):
        abp_125 = signal.resample(abp_data[i], 125).tolist()
        bbp_125 = signal.resample(bbp_data[i], 125).tolist()
        abp_data_125.append(abp_125)
        bbp_data_125.append(bbp_125)

    # 聚类中心展示
    # plt.figure(1, figsize=(12, 8))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # plt.title("centers")
    # plt.ylabel('P/mmHg')
    # for i in range(len(centers)):
    #     plt.plot(centers[i])
    #     # plt.pause(0.1)
    # plt.show()

    # 子类展示
    for i in range(len(cluster_index)):
        num = len(cluster_index[i])
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.title("ppg")
        plt.ylabel('P/mmHg')
        for j in range(num):
            bbp_data[cluster_index[i][j]] = signal.resample(bbp_data[cluster_index[i][j]], 125).tolist()
            plt.plot(bbp_data[cluster_index[i][j]], label="bbp")
        plt.plot(centers[i], label="center", linestyle='--')
        plt.subplot(2, 1, 2)
        plt.title("abp")
        plt.ylabel("p/mmHg")
        for j in range(num):
            abp_data[cluster_index[i][j]] = signal.resample(abp_data[cluster_index[i][j]], 125).tolist()
            plt.plot(abp_data[cluster_index[i][j]])

        plt.tight_layout()
        plt.legend(loc='upper right', fontsize=6)
        plt.show()
