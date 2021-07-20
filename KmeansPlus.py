# K-means++
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import math
from MIMICData import MIMICHelper
from scipy import signal
from SphygmoCorData import SphygmoCorHelper


# data = []
# labels = []
# # 数据读取
# with codecs.open("data.txt", "r") as f:
#     for line in f.readlines():
#         x, y, label = line.strip().split('\t')
#         data.append([float(x), float(y)])
#         labels.append(float(label))
# datas = array(data)


# 计算欧氏距离
def distance(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


# 对一个样本找到与该样本距离最近的聚类中心
def nearest(point, cluster_centers):
    min_dist = inf
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i,])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


# 选择尽可能相距较远的类中心
def get_centroids(dataset, k):
    m, n = np.shape(dataset)
    cluster_centers = np.zeros((k, n))
    index = np.random.randint(0, m)
    cluster_centers[0,] = dataset[index,]
    # 2、初始化一个距离的序列
    d = [0.0 for _ in range(m)]
    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(dataset[j,], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random.rand()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all = sum_all - di
            if sum_all > 0:
                continue
            cluster_centers[i,] = dataset[j,]
            break
        print("找到第" + str(i) + "个聚类中心")
    return cluster_centers


# 主程序
def KmeansPlus(dataset, k):
    percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    row_m = shape(dataset)[0]
    cluster_assign = zeros((row_m, 2))
    center = get_centroids(dataset, k)
    change = True
    while change:
        change = False
        percentage_index = 0
        for i in range(row_m):
            mindist = inf
            min_index = -1
            for j in range(k):
                distance1 = distance(center[j, :], dataset[i, :])
                if distance1 < mindist:
                    mindist = distance1
                    min_index = j
            if cluster_assign[i, 0] != min_index:
                change = True
                print("重新计算聚类中心")
            cluster_assign[i, :] = min_index, mindist ** 2
            if i == int(row_m * percentage[percentage_index]):
                print("计算了" + str(percentage[percentage_index] * 100) + "%数据")
                percentage_index += 1
                if percentage_index == 10:
                    percentage_index = 0
        for cen in range(k):
            cluster_data = dataset[nonzero(cluster_assign[:, 0] == cen)]
            center[cen, :] = mean(cluster_data, 0)
    return center, cluster_assign


if __name__ == "__main__":
    # 读数据
    mimicHelper = MIMICHelper()
    # one_ppg_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_ppg.blood")
    # one_abp_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_abp.blood")
    sphygmoCorHelper = SphygmoCorHelper()
    bbp_data, abp_data = sphygmoCorHelper.readSphygmoCorData()

    # resample到125个点
    # one_ppg_data = [signal.resample(data, mimicHelper.SAMPLE_RATE).tolist() for data in one_ppg_data]
    # one_abp_data = [signal.resample(data, mimicHelper.SAMPLE_RATE).tolist() for data in one_abp_data]
    # one_ppg_data = array(one_ppg_data)
    # one_abp_data = array(one_abp_data)
    bbp_data = [signal.resample(data, sphygmoCorHelper.SAMPLE_RATE).tolist() for data in bbp_data]
    abp_data = [signal.resample(data, sphygmoCorHelper.SAMPLE_RATE).tolist() for data in abp_data]
    bbp_data = array(bbp_data)
    abp_data = array(abp_data)

    # 开始聚类，k个类
    k = 500
    path = sphygmoCorHelper.SPHYGMOCOR_500_PATH
    cluster_center, cluster_assign = KmeansPlus(bbp_data, k)
    cluster_center = cluster_center.tolist()
    mimicHelper.writeToFile(cluster_center, path + "center.cluster")
    for i in range(k):
        index_list = nonzero(cluster_assign[:, 0] == i)[0].tolist()
        mimicHelper.writeToFile2(index_list, path + str(i) + ".cluster")

    k = 200
    path = sphygmoCorHelper.SPHYGMOCOR_200_PATH
    cluster_center, cluster_assign = KmeansPlus(bbp_data, k)
    cluster_center = cluster_center.tolist()
    mimicHelper.writeToFile(cluster_center, path + "center.cluster")
    for i in range(k):
        index_list = nonzero(cluster_assign[:, 0] == i)[0].tolist()
        mimicHelper.writeToFile2(index_list, path + str(i) + ".cluster")

    k = 100
    path = sphygmoCorHelper.SPHYGMOCOR_100_PATH
    cluster_center, cluster_assign = KmeansPlus(bbp_data, k)
    cluster_center = cluster_center.tolist()
    mimicHelper.writeToFile(cluster_center, path + "center.cluster")
    for i in range(k):
        index_list = nonzero(cluster_assign[:, 0] == i)[0].tolist()
        mimicHelper.writeToFile2(index_list, path + str(i) + ".cluster")