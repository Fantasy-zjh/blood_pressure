import threading

import numpy as np
import cv2
import time
from multiprocessing import Process, Pipe


def perf_time(func):
    def wrap(*args):
        start = time.time()
        result = func(*args)
        cost = time.time() - start
        print("{} 调用耗时 {} 毫秒".format(func.__name__, cost * 1000))
        return result

    return wrap


# 数据导入
@perf_time
def load_data(file_path):
    '''导入数据
    input:  file_path(string):文件的存储位置
    output: data(mat):数据
    '''
    data = []
    # 读取图片，转码后为矩阵，大小一般为m×n×3
    img = cv2.imread(file_path)
    # 获得图片大小
    m, n, _ = img.shape

    # 把图片展开，铺平
    for i in range(m):
        for j in range(n):
            tmp = []
            data.append(img[i, j,])
    return np.mat(data), m, n


# 定义相似性的度量
def distance(vecA, vecB):
    '''计算vecA与vecB之间的欧式距离的平方
    input:  vecA(mat)A点坐标
            vecB(mat)B点坐标
    output: dist[0, 0](float)A点与B点距离的平方
    '''
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]


@perf_time
def randCenter(data, k):
    '''随机初始化聚类中心
    input:  data(mat):训练数据
            k(int):类别个数
    output: centroids(mat):聚类中心
    '''
    # 属性的个数，也就是load_data里的n
    _, n = data.shape

    # 初始化k个聚类中心，设为0
    centroids = np.mat(np.zeros((k, n)))
    # 初始化聚类中心每一维的坐标
    for j in range(n):
        # 求出每种特征值最小值，在图像里就是RGB取值最小值
        minJ = np.min(data[:, j])
        # 特征值取值范围
        rangeJ = np.max(data[:, j]) - minJ
        # 在最大值和最小值之间随机初始化，公式为：c=min+rand(0,1)×(max-min)
        centroids[:, j] = minJ * np.mat(np.ones((k, 1))) \
                          + np.random.rand(k, 1) * rangeJ
    return centroids


@perf_time
def kmeans(data, k, centroids, conns, processCount, result):
    '''根据KMeans算法求解聚类中心
    input:  data(mat):训练数据
            k(int):类别个数
            centroids(mat):随机初始化的聚类中心
    output: centroids(mat):训练完成的聚类中心
            subCenter(mat):每一个样本所属的类别
    '''
    # m：样本的个数，n：特征的维度
    m, n = np.shape(data)
    # 初始化每一个样本所属的类别，subCenter用来记录类别与相似度
    subCenter = np.mat(np.zeros((m, 2)))
    # 判断是否需要重新计算聚类中心
    change = True

    while change == True:
        change = False  # 重置

        # 拆分数据发送给不同进程计算
        for i in range(processCount):
            print("send msg to process {}".format(i))
            conns[i][0].send((change, centroids, subCenter[i::processCount]))
        start = time.time()
        subCenter_i_s = []
        # 从进程中回收处理后的 subCenter数据
        for i in range(processCount):
            recv_change, subCenter_i = conns[i][0].recv()
            if recv_change == True:
                change = True
            subCenter_i_s.append(subCenter_i)
        cost = time.time() - start
        print("多线程对数据进行标记 耗时 {} 秒, 处理数据长度 {}".format(cost, m))
        # 组合 subCenter
        if processCount == 1:
            subCenter = subCenter_i_s[0]
        else:
            subCenter = np.concatenate((subCenter_i_s[0], subCenter_i_s[1]), 1)
            for i in range(2, processCount):
                tmp = np.concatenate((subCenter, subCenter_i_s[i]), 1)
                subCenter = tmp
        subCenter = subCenter.reshape(m, 2)
        print("组合后 subCenter.shape={}".format(subCenter.shape))
        # 对样本按照聚类中心进行分类
        # 重新计算聚类中心
        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0  # 每个类别中的样本的个数
            for i in range(m):
                if subCenter[i, 0] == j:  # 计算第j个类别
                    sum_all += data[i,]
                    r += 1
            try:
                # 所以centroids是一个2维矩阵，每行为对应类别的中心
                centroids[j,] = sum_all[0,] / r
            except:
                print(" r is zero")
    for i in range(processCount):
        conns[i][0].send((None, None, None))
    result.append(subCenter)
    result.append(centroids)
    # result = [subCenter, centroids]


def annotationProcess(t_id, k, data_len, data, conns, processCount):
    print("annotationProcess in !!!")
    while True:
        change, centroids, subCenter = conns[t_id][1].recv()
        if change == None:
            print("process {} recv None, end now !!!".format(t_id))
            break
        print("process {} recv data, change = {}".format(t_id, change))
        start = time.time()
        sub_id = 0
        count = 0
        for i in range(t_id, data_len, processCount):
            # 设置样本与聚类中心之间的最小的距离，初始值为正无穷
            minDist = np.inf
            minIndex = 0  # 所属的类别
            for j in range(k):
                # 计算i和每个聚类中心之间的距离
                dist = distance(data[i,], centroids[j,])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            # 判断是否需要改变
            if subCenter[sub_id, 0] != minIndex:  # 需要改变
                change = True
                subCenter[sub_id,] = np.mat([minIndex, minDist])
            sub_id += 1
            count += 1
        conns[t_id][1].send((change, subCenter))
        cost = time.time() - start
        print("process {} 本次处理耗时 {} 秒, change = {}, 共处理数据长度 {} ".format(t_id, cost, change, count))


if __name__ == "__main__":
    processCount = 4
    k = 7
    file_path = 'city.jpg'
    # 导入数据
    data, m, n = load_data(file_path)

    # 随机初始化聚类中心
    centroids = randCenter(data, k, )

    conns = []
    for i in range(processCount):
        conn1, conn2 = Pipe()
        conns.append((conn1, conn2))

    result = []
    # 启动 1个kmeans 线程
    main_thread = threading.Thread(target=kmeans, args=(data, k, centroids, conns, processCount, result))
    main_thread.start()
    process_s = []
    # 启动 processCount 个进程进行计算
    for i in range(processCount):
        p = Process(target=annotationProcess, args=(i, k, m * n, data, conns, processCount))
        process_s.append(p)
    for p in process_s:
        p.start()
    for p in process_s:
        p.join()

    main_thread.join()
    # 聚类结果
    subCenter, centroids = result

    # 保存分割后的图片
    new_pic = np.zeros((m * n, 3))
    print(new_pic.shape)

    for i in range(m * n):
        for j in range(k):
            if subCenter[i, 0] == j:
                new_pic[i, :] = centroids[j, :]
    new = np.reshape(new_pic, (m, n, 3))
    cv2.imwrite('new_pic.jpg', new)