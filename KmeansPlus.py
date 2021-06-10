# K-means++
from pylab import *
from numpy import *
import codecs
import matplotlib.pyplot as plt
import math


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


datas = [[1, 1, 1, 1], [5, 5, 5, 5], [10, 10, 10, 10], [1, 0, 2, 1], [3, 4, 5, 6], [11, 12, 13, 14], [11, 11, 11, 11],
         [100, 100, 100, 100], [150, 150, 160, 140], [200, 300, 200, 100], [211, 250, 230, 180], [50, 60, 70, 80]]
datas = array(datas)
cluster_center, cluster_assign = KmeansPlus(datas, 3)
print("cluster_center：\r\n" + str(cluster_center))
print("----------------------------------")
print("cluster_assign: \r\n" + str(cluster_assign))
print("----------------------------------")
print(datas[nonzero(cluster_assign[:, 0] == 0)])
print("----------------------------------")
print(datas[nonzero(cluster_assign[:, 0] == 1)])
print("----------------------------------")
print(datas[nonzero(cluster_assign[:, 0] == 2)])


#
# # 设置x,y轴的范围
# xlim(0, 10)
# ylim(0, 10)
# # 做散点图
# f1 = plt.figure(1)
# plt.scatter(datas[nonzero(cluster_assign[:, 0] == 0), 0], datas[nonzero(cluster_assign[:, 0] == 0), 1], marker='o',
#             color='r', label='0', s=30)
# plt.scatter(datas[nonzero(cluster_assign[:, 0] == 1), 0], datas[nonzero(cluster_assign[:, 0] == 1), 1], marker='+',
#             color='b', label='1', s=30)
# plt.scatter(datas[nonzero(cluster_assign[:, 0] == 2), 0], datas[nonzero(cluster_assign[:, 0] == 2), 1], marker='*',
#             color='g', label='2', s=30)
# plt.scatter(cluster_center[:, 1], cluster_center[:, 0], marker='x', color='m', s=50)
# plt.show()
