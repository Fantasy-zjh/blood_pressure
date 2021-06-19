from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
import KmeansPlus
from scipy import signal

if __name__ == "__main__":
    mimicHelper = MIMICHelper()
    start_time = time.time()
    abp_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_abp.blood")
    ppg_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_ppg.blood")
    centers = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_1000_PATH + "center.cluster")
    cluster_index = list()
    for i in range(1000):
        index = mimicHelper.readFromFileInteger(mimicHelper.MIMIC_ONE_1000_PATH + str(i) + ".cluster")
        cluster_index.append(index)
    end_time = time.time()
    # print("行：" + str(len(abp_data)))  #11808
    # print("列：" + str(len(abp_data[0])))  #1000
    print("读取数据耗时：" + str(end_time - start_time))  # 45.54142904281616

    # 聚类中心展示
    # plt.figure(1, figsize=(12, 8))
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # plt.title("centers")
    # plt.ylabel('P/mmHg')
    # for i in range(len(centers)):
    #     plt.plot(centers[i])
    # plt.pause(0.01)

    # 子类展示
    for i in range(len(cluster_index)):
        num = len(cluster_index[i])
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.title("ppg")
        plt.ylabel('P/mmHg')
        for j in range(num):
            ppg_data[cluster_index[i][j]] = signal.resample(ppg_data[cluster_index[i][j]], 125).tolist()
            plt.plot(ppg_data[cluster_index[i][j]], label="ppg")
        plt.plot(centers[i], label="center", linestyle='--')
        plt.subplot(2, 1, 2)
        plt.title("abp")
        plt.ylabel("p/mmHg")
        for j in range(num):
            abp_data[cluster_index[i][j]] = signal.resample(abp_data[cluster_index[i][j]], 125).tolist()
            plt.plot(abp_data[cluster_index[i][j]])

        plt.tight_layout()
        plt.legend()
        plt.show()
