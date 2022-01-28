import sys

import KmeansPlus
from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
from scipy import signal
from SphygmoCorData import SphygmoCorHelper
from FileHelper import FileHelper


def DaviesBouldinIndexScore(data, centers, clusterIndexs):
    N = len(centers)  # 类别个数
    o6s = []
    for i in range(N):
        n = len(clusterIndexs[i])  # 类内元素个数
        totalDistance = 0
        for j in range(n):
            totalDistance += KmeansPlus.distance(centers[i], data[clusterIndexs[i][j]])
        o6s.append(totalDistance / n)  # 类簇内所有点到中心的平均距离
    DBIScore = 0
    for i in range(N):
        maxV = 0
        for j in range(N):
            if i == j:
                continue  # i不等于j
            V = (o6s[i] + o6s[j]) / KmeansPlus.distance(centers[i], centers[j])
            if V > maxV:
                maxV = V
        DBIScore += maxV
    return DBIScore / N


if __name__ == "__main__":
    readPath = MIMICHelper.NEW_CLUSTER_ORIGINAL
    # 读取原始的未处理的abp和ppg波形
    ppg_data = FileHelper.readFromFileFloat(readPath + "ppg_train.blood")
    abp_data = FileHelper.readFromFileFloat(readPath + "abp_train.blood")

    # 读取ppg聚类中心波形
    N = 100
    centers = FileHelper.readFromFileFloat(readPath + "java_" + str(N) + "\\center.cluster")

    # 读取子类索引
    cluster_index = list()
    for i in range(N):
        index = FileHelper.readFromFileInteger(readPath + "java_" + str(N) + "\\" + str(i) + ".cluster")
        cluster_index.append(index)

    scores = []
    ns = [10, 50, 100, 150, 200, 250, 300]
    for n in ns:
        centers = FileHelper.readFromFileFloat(readPath + "java_" + str(n) + "\\center.cluster")
        cluster_index = list()
        for i in range(n):
            index = FileHelper.readFromFileInteger(readPath + "java_" + str(n) + "\\" + str(i) + ".cluster")
            cluster_index.append(index)
        DBISore = DaviesBouldinIndexScore(ppg_data, centers, cluster_index)
        scores.append(DBISore)
        print("类别数={}, DBI score = {}".format(n, DBISore))
    xyFont = {
        'family': 'Times New Roman',
        'size': 20
    }
    plt.plot(ns, scores)
    plt.xlabel('Cluster number', xyFont)
    plt.ylabel('DBI Score', xyFont)
    plt.show()
    # resample至125个点
    # abp_data_125 = list()
    # ppg_data_125 = list()
    # for i in range(len(ppg_data)):
    #     abp_125 = signal.resample(abp_data[i], 125).tolist()
    #     ppg_125 = signal.resample(ppg_data[i], 125).tolist()
    #     abp_data_125.append(abp_125)
    #     ppg_data_125.append(ppg_125)
    # abp_data_125 = list()
    # bbp_data_125 = list()
    # for i in range(len(abp_data)):
    #     abp_125 = signal.resample(abp_data[i], 125).tolist()
    #     bbp_125 = signal.resample(bbp_data[i], 125).tolist()
    #     abp_data_125.append(abp_125)
    #     bbp_data_125.append(bbp_125)

    # 聚类中心展示
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.figure(1, figsize=(12, 8))
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # plt.title("centers")
    # plt.ylabel('P/mmHg')
    # for i in range(len(centers)):
    #     plt.plot(centers[i])
    #     plt.pause(0.1)
    # plt.show()

    sys.exit()

    # 子类展示
    for i in range(len(cluster_index)):
        num = len(cluster_index[i])
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.title("ppg")
        plt.ylabel('P/mmHg')
        for j in range(num):
            # bbp_data[cluster_index[i][j]] = signal.resample(bbp_data[cluster_index[i][j]], 125).tolist()
            plt.plot(ppg_data[cluster_index[i][j]])
        plt.plot(centers[i], linestyle='--')
        plt.subplot(2, 1, 2)
        plt.title("abp")
        plt.ylabel("p/mmHg")
        for j in range(num):
            # abp_data[cluster_index[i][j]] = signal.resample(abp_data[cluster_index[i][j]], 125).tolist()
            plt.plot(abp_data[cluster_index[i][j]])

        plt.tight_layout()
        # plt.legend(loc='upper right', fontsize=6)
        plt.show()
