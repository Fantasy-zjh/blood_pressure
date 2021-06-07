import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import signal
import h5py
import random
import time
import os
import math


class DataHelper:
    ABPdata = None
    PPGdata = None
    common_abs = None
    common_angle = None
    # T = 80  # 周期设为80
    common = None

    def __init__(self):
        # self.ABPdata, self.PPGdata = self.readMIT()
        # self.ABPdata, self.PPGdata = self.readData5()
        self.common = [1.00024076 + 0.00000000e+00j, -4.80190831 + 1.79689198e+02j,
                       0.77305181 + 3.09325903e+02j, 0.39267281 + 5.47949930e+01j,
                       0.42260549 - 2.70448455e+00j, 0.52301446 - 2.32834763e+01j,
                       0.6048204 - 1.99531896e+01j, 0.64327844 - 1.21194565e+01j,
                       0.63884282 - 7.54515355e+00j, 0.59754004 - 4.75961368e+00j,
                       0.45732813 - 3.07094787e+00j, 0.29783366 - 2.13490714e+00j,
                       0.31489687 - 1.53446401e+00j, 0.41855697 - 1.13171396e+00j,
                       0.50878699 - 8.46853345e-01j, 0.57079715 - 6.33088894e-01j,
                       0.5950648 - 4.76014463e-01j, 0.59714063 - 3.64079462e-01j,
                       0.59672549 - 2.82033962e-01j, 0.59057652 - 2.20308673e-01j,
                       0.58216461 - 1.74369421e-01j, 0.5726364 - 1.40005585e-01j,
                       0.56565424 - 1.13910987e-01j, 0.55722371 - 9.39958554e-02j,
                       0.55060918 - 7.81860638e-02j, 0.5468373 - 6.53446030e-02j,
                       0.54184664 - 5.50853188e-02j, 0.53941755 - 4.78339808e-02j,
                       0.53701489 - 4.09221184e-02j, 0.5340014 - 3.51164212e-02j,
                       0.53133392 - 3.03704919e-02j, 0.52615732 - 2.69176486e-02j,
                       0.51917951 - 2.29593648e-02j, 0.51045038 - 2.03652109e-02j,
                       0.50098074 - 1.74320548e-02j, 0.49235517 - 1.52858658e-02j,
                       0.48345308 - 1.26898758e-02j, 0.4735889 - 1.11110953e-02j,
                       0.46532433 - 9.76531878e-03j, 0.45319527 - 7.81983739e-03j,
                       0.4449566 + 0.00000000e+00j, 0.45319527 + 7.81983739e-03j,
                       0.46532433 + 9.76531878e-03j, 0.4735889 + 1.11110953e-02j,
                       0.48345308 + 1.26898758e-02j, 0.49235517 + 1.52858658e-02j,
                       0.50098074 + 1.74320548e-02j, 0.51045038 + 2.03652109e-02j,
                       0.51917951 + 2.29593648e-02j, 0.52615732 + 2.69176486e-02j,
                       0.53133392 + 3.03704919e-02j, 0.5340014 + 3.51164212e-02j,
                       0.53701489 + 4.09221184e-02j, 0.53941755 + 4.78339808e-02j,
                       0.54184664 + 5.50853188e-02j, 0.5468373 + 6.53446030e-02j,
                       0.55060918 + 7.81860638e-02j, 0.55722371 + 9.39958554e-02j,
                       0.56565424 + 1.13910987e-01j, 0.5726364 + 1.40005585e-01j,
                       0.58216461 + 1.74369421e-01j, 0.59057652 + 2.20308673e-01j,
                       0.59672549 + 2.82033962e-01j, 0.59714063 + 3.64079462e-01j,
                       0.5950648 + 4.76014463e-01j, 0.57079715 + 6.33088894e-01j,
                       0.50878699 + 8.46853345e-01j, 0.41855697 + 1.13171396e+00j,
                       0.31489687 + 1.53446401e+00j, 0.29783366 + 2.13490714e+00j,
                       0.45732813 + 3.07094787e+00j, 0.59754004 + 4.75961368e+00j,
                       0.63884282 + 7.54515355e+00j, 0.64327844 + 1.21194565e+01j,
                       0.6048204 + 1.99531896e+01j, 0.52301446 + 2.32834763e+01j,
                       0.42260549 + 2.70448455e+00j, 0.39267281 - 5.47949930e+01j,
                       0.77305181 - 3.09325903e+02j, -4.80190831 - 1.79689198e+02j]

        # plt.figure()
        # plt.subplot(211)
        # plt.plot(self.ABPdata[0])
        # plt.subplot(212)
        # plt.plot(self.PPGdata[0])
        # plt.show()
        # print("ABPdata 行 is {}, 列 is {},{},{}".format(len(self.ABPdata), len(self.ABPdata[0]), len(self.ABPdata[1]),
        #                                               len(self.ABPdata[2])))
        # print("PPGdata 行 is {}, 列 is {},{},{}".format(len(self.PPGdata), len(self.PPGdata[0]), len(self.PPGdata[1]),
        #                                               len(self.PPGdata[2])))

    # Cuff-Less Blood Pressure Estimation Data Set
    def readMIT(self):
        basepath = "D:\\code\\exercise\\data\\Part_"
        # 中心动脉压
        abp = list()
        # PPG
        ppg = list()
        for part in range(1, 2):
            path = basepath + str(part) + ".mat"
            with h5py.File(path, 'r') as f:
                refs = f['#refs#']
                count = -1
                for key in refs.keys():
                    count += 1
                    # 1728有问题，跳过1700——1750
                    if 1700 < count < 1750:
                        continue
                    if count == 50:
                        break
                    # 转置,3行
                    Tdata = np.array(refs[key]).T
                    # 获取中心动脉压数据，原始太大只取前600ms的数据
                    abpdata = Tdata[1][:600]
                    # ppg原始数据
                    ppgdata = Tdata[0][:600]
                    if not (abpdata.any() and ppgdata.any()):
                        continue

                    # 去除基线漂移
                    abpdata = self.smooth(abpdata)
                    ppgdata = self.smooth(ppgdata)
                    # 周期定为65ms，设置两个波峰之间的距离不小于60
                    abppeaks = signal.find_peaks(abpdata, distance=60)
                    ppgpeaks = signal.find_peaks(ppgdata, distance=60)
                    if len(abppeaks[0]) == 0 or len(ppgpeaks[0]) == 0:
                        continue
                    select = random.randint(1, len(abppeaks[0]) - 2)
                    peak = abppeaks[0][select]
                    abp.append(abpdata[peak - 20: peak + 45])
                    select = random.randint(1, len(ppgpeaks[0]) - 2)
                    peak = ppgpeaks[0][select]
                    ppg.append(ppgdata[peak - 20: peak + 45])
        return np.array(abp), np.array(ppg)

    # 新的数据集，有脉搏波和预测的中心动脉压两组数据
    def readData5(self):
        data = []
        p_data = []
        for file in os.listdir("D:\\code\\exercise\\data5"):
            path = "D:\\code\\exercise\\data5\\" + file
            with open(path, 'r') as f:
                nums1 = []
                nums2 = []
                for line in f.readlines():
                    line = line.strip()
                    num1, num2 = line.split(' ')
                    nums1.append(float(num1))
                    nums2.append(float(num2))
                new_nums1 = signal.resample(nums1[:], self.T)
                new_nums2 = signal.resample(nums2[:], self.T)
                data.append(new_nums1[:])
                p_data.append(new_nums2[:])
        return np.array(data), np.array(p_data)

    # 平滑先验法平滑数据
    def smooth(self, nums):
        n = len(nums)
        I = np.eye(n)
        lamta = 5000
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            for j in range(n):
                if j == i:
                    D[i][j] = 1
                elif j == i + 1:
                    D[i][j] = -2
                elif j == i + 2:
                    D[i][j] = 1
        K = I - np.matrix((I + (lamta ** 2) * np.dot(D.T, D))).I
        f = K * nums.reshape(-1, 1)
        return np.array(f).reshape(1, -1)[0]

    # 训练通用传递函数, MIT版本
    def trainTransfer(self, v1):
        # 将ABP和PPG数据进行傅里叶变换
        # 行row = 很多
        # 列col = 65
        row = len(self.ABPdata)
        col = len(self.ABPdata[0])
        fft_ABP = list()
        fft_PPG = list()
        for i in range(row):
            fft_ABP.append(fft(self.ABPdata[i]))
            fft_PPG.append(fft(self.PPGdata[i]))
        # 以1HZ为单位，计算全部模和幅角的均值
        abs_abp = [0] * col
        angle_abp = [0] * col
        abs_ppg = [0] * col
        angle_ppg = [0] * col
        for i in range(row):
            abs__abp = np.abs(fft_ABP[i])
            angle__abp = np.angle(fft_ABP[i])
            abs__ppg = np.abs(fft_PPG[i])
            angle__ppg = np.angle(fft_PPG[i])
            for j in range(col):
                abs_abp[j] += abs__abp[j]
                abs_ppg[j] += abs__ppg[j]
                angle_abp[j] += angle__abp[j]
                angle_ppg[j] += angle__ppg[j]
        abs_abp = [num / row for num in abs_abp]
        abs_ppg = [num / row for num in abs_ppg]
        angle_abp = [num / row for num in angle_abp]
        angle_ppg = [num / row for num in angle_ppg]
        # 计算通用传递函数 ppg/abp
        abs_common = [0.0] * col
        angle_common = [0.0] * col
        for i in range(col):
            abs_common[i] = abs_ppg[i] / abs_abp[i]
            angle_common[i] = angle_ppg[i] - angle_abp[i]
        self.common_abs = abs_common
        self.common_angle = angle_common
        common = list()
        for i in range(col):
            common.append(complex(abs_common[i], angle_common[i]))
        common = np.array(common)
        print(common)

    # 训练通用传递函数，新数据集data5版本
    def trainTransfer(self, v1, v2):
        # 将ABP和PPG数据进行傅里叶变换
        # 行row = 很多
        # 列col = 80
        row = len(self.ABPdata)
        col = len(self.ABPdata[0])
        fft_ABP = list()
        fft_PPG = list()
        for i in range(row):
            fft_ABP.append(fft(self.ABPdata[i]))
            fft_PPG.append(fft(self.PPGdata[i]))
        # 以1HZ为单位，计算全部模和幅角的均值
        abs_abp = [0] * col
        angle_abp = [0] * col
        abs_ppg = [0] * col
        angle_ppg = [0] * col
        for i in range(row):
            abs__abp = np.real(fft_ABP[i])
            angle__abp = np.imag(fft_ABP[i])
            abs__ppg = np.real(fft_PPG[i])
            angle__ppg = np.imag(fft_PPG[i])
            for j in range(col):
                abs_abp[j] += abs__abp[j]
                abs_ppg[j] += abs__ppg[j]
                angle_abp[j] += angle__abp[j]
                angle_ppg[j] += angle__ppg[j]
        abs_abp_mean = [num / row for num in abs_abp]
        abs_ppg_mean = [num / row for num in abs_ppg]
        angle_abp_mean = [num / row for num in angle_abp]
        angle_ppg_mean = [num / row for num in angle_ppg]
        # 计算通用传递函数 ppg/abp
        abs_common = [0.0] * col
        angle_common = [0.0] * col
        for i in range(col):
            abs_common[i] = abs_ppg_mean[i] / abs_abp_mean[i]
            angle_common[i] = angle_ppg_mean[i] - angle_abp_mean[i]
        self.common_abs = abs_common
        self.common_angle = angle_common
        common = list()
        for i in range(col):
            common.append(complex(abs_common[i], angle_common[i]))
        common = np.array(common)
        print(common)
        print(len(common))
        print(self.common_abs)
        print(self.common_angle)

    # 推测中心动脉压
    def produce(self):
        # data =      self.PPGdata[135]
        # real_data = self.ABPdata[135]
        # fft_data = fft(data)
        # abs_data = np.real(fft_data)
        # angle_data = np.imag(fft_data)
        #
        # self.common_abs = np.real(self.common)
        # self.common_angle = np.imag(self.common)
        # fft_center = list()
        # for i in range(len(self.common_angle)):
        #     real = abs_data[i] / self.common_abs[i]
        #     imag = angle_data[i] - self.common_angle[i]
        #     fft_center.append(complex(real, imag))
        # fft_center = np.array(fft_center)
        # center = ifft(fft_center)
        # real_center = np.real(center)
        # print(abs(max(real_data) - max(real_center)))


        # plt.subplot(211)
        # plt.plot(data)
        # plt.subplot(212)
        # plt.plot(real_data, 'black')
        # plt.plot(real_center, 'red')
        #
        # plt.show()
        count_below05 = 0
        count_below10 = 0
        count_above10 = 0
        count_above15 = 0
        count_above20 = 0

        for i in range(len(self.PPGdata)):
            data = self.PPGdata[i]
            real_data = self.ABPdata[i]
            fft_data = fft(data)
            abs_data = np.real(fft_data)
            angle_data = np.imag(fft_data)

            self.common_abs = np.real(self.common)
            self.common_angle = np.imag(self.common)
            fft_center = list()
            for i in range(len(self.common_angle)):
                real = abs_data[i] / self.common_abs[i]
                imag = angle_data[i] - self.common_angle[i]
                fft_center.append(complex(real, imag))
            fft_center = np.array(fft_center)
            center = ifft(fft_center)
            real_center = np.real(center)
            # print()
            a = abs(max(real_data) - max(real_center))
            print(a)
            if a <= 10:
                count_below10 += 1
                if a <= 5:
                    count_below05 += 1
            else:
                count_above10 += 1
                if a > 15:
                    count_above15 += 1
                if a > 20:
                    count_above20 += 1
        print("低于10 " + str(count_below10) + " 占比 " + str(count_below10 / len(self.PPGdata)))
        print("低于05 " + str(count_below05) + " 占比 " + str(count_below05 / len(self.PPGdata)))
        print("超过10 " + str(count_above10) + " 占比 " + str(count_above10 / len(self.PPGdata)))
        print("超过15 " + str(count_above15) + " 占比 " + str(count_above15 / len(self.PPGdata)))
        print("超过20 " + str(count_above20) + " 占比 " + str(count_above20 / len(self.PPGdata)))

    # 欧陆克专利，不知道有啥用，老师让做的
    def unknown(self):
        data = self.PPGdata[15]
        peaks = signal.find_peaks(data)
        # 最大值点
        SMAX = 0
        for index in peaks[0]:
            # print("最大值点:", index)
            if data[index] > data[SMAX]:
                SMAX = index
        # print(SMAX)
        # 导数
        # first_derivative = np.gradient(data)
        # second_derivative = np.gradient(first_derivative)
        # third_derivative = np.gradient(second_derivative)
        first_derivative = self.cal_deriv(data)
        second_derivative = self.cal_deriv(first_derivative)
        third_derivative = self.cal_deriv(second_derivative)
        # 一阶导数最大值点和起始点
        peaks = signal.find_peaks(first_derivative)
        MAXDPDT = 0
        for index in peaks[0]:
            # print("一阶导数最大值点:", index)
            if first_derivative[index] > first_derivative[MAXDPDT]:
                MAXDPDT = index
        # print(MAXDPDT)
        SO = 0
        # 三阶导数最大值点和第一个负正交界点
        peaks = signal.find_peaks(third_derivative)
        MAX3RD = 0
        for index in peaks[0]:
            # print("三阶导数最大值点:", index)
            if third_derivative[index] > third_derivative[MAX3RD]:
                MAX3RD = index
        # print(MAX3RD)
        ZC3RD = 0
        for i in range(len(data) - 1):
            if third_derivative[i] < 0 and third_derivative[i + 1] > 0:
                ZC3RD = i + 1
                break
        # 找shoulder
        first_shoulder, second_shoulder = 0, 0
        if SMAX >= MAX3RD + 20:
            # type A
            first_shoulder = MAX3RD
            second_shoulder = SMAX
        else:
            # type B or C
            first_shoulder = SMAX
            if SMAX - 10 < ZC3RD < SMAX + 10 or ZC3RD > SO + 45:
                second_shoulder = 0
            else:
                second_shoulder = ZC3RD

        print("RSI = ", data[second_shoulder] / data[first_shoulder])

        plt.figure()
        plt.subplot(311)
        plt.plot(data)
        plt.plot(SMAX, data[SMAX], '*')
        plt.plot(MAXDPDT, data[MAXDPDT], '*')
        plt.plot(MAX3RD, data[MAX3RD], '^')
        plt.subplot(312)
        plt.plot(first_derivative)
        plt.plot(MAXDPDT, first_derivative[MAXDPDT], '*')
        plt.subplot(313)
        plt.plot(third_derivative)
        plt.plot(MAX3RD, third_derivative[MAX3RD], '*')
        plt.plot(ZC3RD, third_derivative[ZC3RD], '^')
        plt.show()

    # 求导函数
    def cal_deriv(self, y):
        # print("原数组大小：" + str(len(y)))
        slopes = []  # 用来存储斜率
        for i in range(len(y) - 1):
            slopes.append(y[i + 1] - y[i])
        # print("斜率数组大小：" + str(len(slopes)))
        deriv = []  # 用来存储一阶导数
        for i in range(len(slopes) - 1):
            deriv.append(0.5 * (slopes[i] + slopes[i + 1]))  # 根据离散点导数的定义，计算并存储结果
        deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
        deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率
        # print("导数数组大小：" + str(len(deriv)))
        return deriv  # 返回存储一阶导数结果的列表

    def showpic(self, file, H, W):
        bbp = []
        abp = []
        # "D:\\code\\exercise\\data5\\0_28一月2016-105603_pwa.txt"
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                num1, num2 = line.split(' ')
                bbp.append(float(num1))
                abp.append(float(num2))
            n = len(bbp)
            N = 128  # 采样频率，常数
            T = n / N * 1000  # 周期
            t = np.linspace(0, T, n)
            # print(t)
            # bbp_ = [-x for x in bbp]
            deriv1 = self.cal_deriv(bbp)  # 一阶导
            deriv2 = self.cal_deriv(deriv1)  # 二阶导

            # peaks = signal.find_peaks(bbp)
            # peaks_ = signal.find_peaks(bbp_)
            dis = math.ceil(170 / (t[1] - t[0]))  # 波峰、波谷的间距
            max_peaks = [0] * n
            min_peaks = [0] * n
            max_len, min_len = self.findPeaks(bbp, n, dis, max_peaks, min_peaks)
            if max_len == 1:
                # print((max_peaks[1] - max_peaks[0]) * (t[1] - t[0]))
                # 怎么样都得再找一个波峰，距第1波峰170ms - 320ms以内斜率最接近0的点
                index = -1
                mind = 100
                for i in range(max_peaks[0] + math.floor(170 / (t[1] - t[0])),
                               max_peaks[0] + math.floor(320 / (t[1] - t[0]))):
                    if abs(deriv1[i]) < mind:
                        mind = abs(deriv1[i])
                        index = i
                max_peaks[max_len] = index
                max_len += 1
            # 筛选波谷
            k = min_len
            for i in range(k):
                if min_peaks[i] > max_peaks[1] or min_peaks[i] < max_peaks[0]:
                    min_peaks[i] = 0
                    min_len -= 1
            if min_len < 1:
                # 怎么样都得找一个波谷，距第2波峰100ms以内斜率最接近0的点
                index = -1
                mind = 100
                for i in range(max_peaks[1] - math.floor(100 / (t[1] - t[0])), max_peaks[1]):
                    if abs(deriv1[i]) < mind:
                        mind = abs(deriv1[i])
                        index = i
                min_peaks[min_len] = index
                min_len += 1


            # 计算各个点
            index_tB = max_peaks[0]  # B点横坐标，第一个波峰
            index_tF = max_peaks[1]  # F点横坐标，第二个波峰
            index_tE = min_peaks[min_len - 1]  # E点横坐标，波谷
            CASBP = self.heyu(abp)  # 中心动脉收缩压（mmHg）
            index_tD = index_tB  # D点横坐标，与中心动脉压相对应的位置
            tmp = bbp[index_tD]
            for i in range(index_tB + 1, index_tE):
                if (abs(CASBP - bbp[i]) < tmp):
                    tmp = abs(CASBP - bbp[i])
                    index_tD = i

            index_tG = math.floor(0.4 * n)  # 4：6的位置是G点

            tmp = abs(deriv2[index_tB])
            index_tC = index_tB
            for i in range(index_tB, index_tD):
                if abs(deriv2[i]) < tmp:
                    tmp = abs(deriv2[i])
                    index_tC = i
            # index_tG = self.findG(bbp, index_tD, index_tD + math.ceil(50 / (t[1] - t[0])))

            # 点计算完毕，开始计算各种参数
            SBP = bbp[index_tB]  # 肱动脉收缩压（mmHg）
            DBP = bbp[0]  # 肱动脉舒张压，起点位置（mmHg）
            MBP = (SBP + 2 * DBP) / 3  # 平均动脉压（mmHg）
            PP = SBP - DBP  # 脉压差（mmHg）
            Hb = bbp[index_tB]
            Hc = bbp[index_tC]
            Hd = bbp[index_tD]
            He = bbp[index_tE]
            Hf = bbp[index_tF]
            AIr = (Hd - DBP) / (Hb - DBP)  # 桡动脉反射波增强指数（%）
            APr = Hd - Hb  # 桡动脉反射波增强压（mmHg）
            HR = 60 * N / n  # 心率（bpm）=60×1000 / T

            t1 = t[index_tG]
            t2 = T - t1
            tAB = t[index_tB]
            tAD = t[index_tD]
            tBD = t[index_tD] - t[index_tB]
            tDF = t[index_tF] - t[index_tD]
            tBF = t[index_tF] - t[index_tB]
            S1 = self.integration(bbp, t, 0, index_tB)  # 主波上升面积（mmHg * ms）
            Pm = self.integration(bbp, t, 0, len(bbp) - 1) / T  # 平均血压（mmHg）
            Pm1 = self.integration(bbp, t, 0, index_tG) / t1  # 收缩期平均压（mmHg）
            Pm2 = self.integration(bbp, t, index_tG, len(bbp) - 1) / t2  # 舒张期平均压（mmHg）
            H1 = Hb - Hd  # 主波、潮波幅值差（mmHg）=Hb - Hd
            H2 = Hb - Hf  # 主波、重搏波幅值差（mmHg）=Hb - Hf
            X1 = t1 / t2  # 心脏收缩期和心脏舒张期比值
            X2 = tAB / t1  # 主波上升时间和心脏收缩期比值
            CTR = tAB / T  # 主峰时间比 = tab / T
            height = H  # Height身高
            weight = W  # Weight体重
            NCT = height / tAB  # NCT：标准化主峰时间 = HEIGHT / tab；
            ASI = height / tBF  # ASI：血管硬化指数 = HEIGHT / tbf；
            RI = Hf / Hb  # 血管反射指数 = Hf / Hb；
            k = (Hb - DBP) / tAB  # 主波斜率 =（Hb - DBP） / tab；
            K = (Pm - DBP) / (SBP - DBP)  # 波形系数 =（Pm - DBP） / （SBP - DBP）
            K1 = (Pm1 - DBP) / (SBP - DBP)  # K1：收缩期面积特征量 =（Pm1 - DBP） / （SBP - DBP）
            K2 = (Pm2 - DBP) / (SBP - DBP)  # K2：舒张期面积特征量 =（Pm2 - DBP） / （SBP - DBP）
            BSA = 0.0061 * height + 0.0128 * weight - 0.1592  # BSA：体表面积（m2）=0.0061×HEIGHT + 0.0128×WEIGHT - 0.1592；
            SV = 0.283 * 60 * (SBP - DBP) / (HR * K2)  # SV：每搏输出量（ml / B）=0.283×60×（SBP - DBP） / （HR×K2）；
            CO = SV * HR / 1000  # CO：心输出量（L / min）=SV×HR / 1000；
            TPR = (60 * Pm) / (SV * HR)  # TPR：外周阻力（PRU）=（60×Pm） / （SV×HR）；
            CI = CO / BSA  # CI：心脏指数（L /（min·m3））=CO / BSA；
            SI = SV / BSA  # SI：心搏指数（ml /（B·m2））=SV / BSA；
            V = 11.43 * K  # 血液粘度（cp）=11.43×K；
            BV = 2.65 * BSA  # BV：血总容量（L）=2.65×BSA；
            ALK = 0.0126 * CO / BSA  # ALK：血流半更新率（1 / s）=0.0126×CO / BSA；
            ALT = 0.693 / ALK  # ALT：血流半更新时间（s）=0.693 / ALK；
            TM = 1 / ALK  # TM：平均滞留时间（s）=1 / ALK；
            print("")
            print("SBP：{:.0f} mmHg".format(SBP))
            print("DBP：{:.0f} mmHg".format(DBP))
            print("PP：{:.0f} mmHg".format(PP))
            print("MAP1：{:.0f} mmHg".format(MBP))
            print("MAP2：{:.0f} mmHg".format(Pm))
            print("HR：{:.0f} bpm".format(HR))
            print("CASP：{:.0f} mmHg".format(CASBP))
            print("rAI：{:.0%}".format(AIr))
            print("rAP：{:.0f} mmHg".format(APr))
            print("K：{:.0f} mmHg".format(K))
            print("CI：{:.0f} mmHg".format(CI))
            print("SI：{:.0f} mmHg".format(SI))
            print("V：{:.0f} mmHg".format(V))
            #新增的项
            P_MAX_DPDT = max(deriv1)  # 1----------------
            abp_deriv1 = self.cal_deriv(abp)  # abp的一阶导
            abp_deriv2 = self.cal_deriv(abp_deriv1)  # abp的二阶导
            abp_max_peaks = [0] * n
            abp_min_peaks = [0] * n
            abp_max_len, abp_min_len = self.findPeaks(abp, n, dis, abp_max_peaks, abp_min_peaks)
            index_T2 = abp_max_peaks[0]  # abp的波峰
            tmp = abs(abp_deriv2[index_T2])
            index_TED = index_T2  # 2-------------ED的位置，二阶导为零的点
            for i in range(index_T2, index_T2 + math.ceil(200 / (t[1] - t[0]))):
                if abs(abp_deriv2[i]) < tmp:
                    tmp = abs(deriv2[i])
                    index_TED = i
            P_AI = bbp[index_tB] - bbp[index_tF]  # 3------------脉搏波两个波峰的差
            P_ESP = bbp[79]  # 4--------------脉搏波最后的压力值





            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False
            # plt.subplot(311)
            plt.title('脉搏波')
            plt.xlabel('t/ms')
            plt.ylabel('P/mmHg')
            plt.plot(t, bbp, label = "脉搏波")  # t是横坐标，bbp是纵坐标
            plt.plot(t, abp, label = "中心动脉压")
            plt.plot(t[index_tB], bbp[index_tB], '*')
            plt.plot(t[index_tF], bbp[index_tF], '*')
            plt.plot(t[index_tE], bbp[index_tE], '*')
            plt.plot(t[index_tD], bbp[index_tD], '*')
            plt.plot(t[index_tC], bbp[index_tC], 'o')
            plt.plot(t[index_tG], bbp[index_tG], '*')
            plt.text(t[index_tB] + 0.25, bbp[index_tB] + 1, 'B', ha='center', va='bottom', fontsize=10.5)
            plt.text(t[index_tF] + 0.25, bbp[index_tF] + 1, 'F', ha='center', va='bottom', fontsize=10.5)
            plt.text(t[index_tE] + 0.25, bbp[index_tE] + 1, 'E', ha='center', va='bottom', fontsize=10.5)
            plt.text(t[index_tD] + 0.25, bbp[index_tD] + 1, 'D', ha='center', va='bottom', fontsize=10.5)
            plt.text(t[index_tC] + 0.25, bbp[index_tC] + 1, 'C', ha='center', va='bottom', fontsize=10.5)
            plt.text(t[index_tG] + 0.25, bbp[index_tG] + 3, 'G', ha='center', va='bottom', fontsize=10.5)
            # plt.subplot(312)
            # plt.title('一阶导')
            # plt.xlabel('t/ms')
            # plt.plot(t, deriv1)
            # plt.subplot(313)
            # plt.title('二阶导')
            # plt.xlabel('t/ms')
            # plt.plot(t, deriv2)

            plt.tight_layout()
            plt.legend()
            plt.show()

    def heyu(self, abp):
        return max(abp)

    def integration(self, nums, t, begin, end):
        s = 0
        for i in range(begin, end):
            dt = t[i + 1] - t[i]
            ds = (nums[i + 1] + nums[i]) * dt / 2
            s += ds
        return s

    # 寻找G点，波形由缓慢下降到急剧下降的转折点
    def findG(self, nums, index_D, index_E):
        deta_min = nums[index_D]
        deta_max = -1
        for i in range(index_D, index_E):
            deta_max = max(deta_max, nums[i] - nums[i + 1])
            deta_min = min(deta_min, nums[i] - nums[i + 1])
        # print("我开始了")
        # print("deta_max = " + str(deta_max))
        # print("deta_min = " + str(deta_min))
        index_G = index_D
        for i in range(index_D + 1, index_E):
            deta_k1 = nums[i - 1] - nums[i]
            deta_k2 = nums[i] - nums[i + 1]
            if deta_k1 <= deta_min and deta_k2 >= 0.8 * deta_max:
                index_G = i
        if index_G == index_D:
            # 重新计算
            for i in range(index_D + 1, index_E):
                deta_k1 = nums[i - 1] - nums[i]
                deta_k2 = nums[i] - nums[i + 1]
                if deta_k1 <= 1.5 * deta_min and deta_k2 >= 0.5 * deta_max:
                    index_G = i
        # print("我结束了")
        return index_G

    # 寻找波峰、波谷函数
    def findPeaks(self, src, src_lenth, distance, indMax, indMin):
        sign = [0] * src_lenth
        max_index = 0
        min_index = 0

        for i in range(1, src_lenth):
            diff = src[i] - src[i - 1]
            if diff > 0:
                sign[i - 1] = 1
            elif diff < 0:
                sign[i - 1] = -1
            else:
                sign[i - 1] = 0

        for j in range(1, src_lenth - 1):
            diff = sign[j] - sign[j - 1]
            if diff < 0:
                indMax[max_index] = j
                max_index += 1
            elif diff > 0:
                indMin[min_index] = j
                min_index += 1

        flag_max_index = [0] * max(max_index, min_index)
        idelete = [0] * max(max_index, min_index)
        temp_max_index = [0] * max(max_index, min_index)
        bigger = 0
        tempvalue = 0

        # 波峰
        for i in range(max_index):
            flag_max_index[i] = 0
            idelete[i] = 0

        for i in range(0, max_index):
            tempvalue = -1
            for j in range(0, max_index):
                if not flag_max_index[j]:
                    if src[indMax[j]] > tempvalue:
                        bigger = j
                        tempvalue = src[indMax[j]]
            flag_max_index[bigger] = 1
            if not idelete[bigger]:
                for k in range(max_index):
                    idelete[k] |= (indMax[k] - distance <= indMax[bigger] & indMax[bigger] <= indMax[k] + distance)
                idelete[bigger] = 0

        j = 0
        for i in range(max_index):
            if not idelete[i]:
                temp_max_index[j] = indMax[i]
                j += 1

        for i in range(max_index):
            if (i < j):
                indMax[i] = temp_max_index[i]
            else:
                indMax[i] = 0
        max_index = j

        # 波谷
        for i in range(min_index):
            flag_max_index[i] = 0
            idelete[i] = 0
        for i in range(min_index):
            tempvalue = 1
            for j in range(min_index):
                if not flag_max_index[j]:
                    if src[indMin[j]] < tempvalue:
                        bigger = j
                        tempvalue = src[indMin[j]]
            flag_max_index[bigger] = 1
            if not idelete[bigger]:
                for k in range(min_index):
                    idelete[k] |= (indMin[k] - distance <= indMin[bigger] & indMin[bigger] <= indMin[k] + distance)
                idelete[bigger] = 0

        j = 0
        for i in range(min_index):
            if not idelete[i]:
                temp_max_index[j] = indMin[i]
                j += 1

        for i in range(min_index):
            if i < j:
                indMin[i] = temp_max_index[i]
            else:
                indMin[i] = 0
        min_index = j

        return max_index, min_index

if __name__ == "__main__":
    starttime = time.time()
    dataHelper = DataHelper()
    # dataHelper.trainTransfer(1, 2)
    dirs = os.listdir("D:\\code\\exercise\\blood_data\\data5")
    # for file in dirs:
    dataHelper.showpic("D:\\code\\exercise\\blood_data\\data5\\" + "白敬华_21十二月2017-092047_pwa.txt", 170, 75)

    endtime = time.time()
    print("total time: {}".format(endtime - starttime))
