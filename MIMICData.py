import os
import time

import h5py
import numpy as np
import math
from Plt import Plt
from scipy import signal
from detecta import detect_peaks
import matplotlib.pyplot as plt
from FileHelper import FileHelper
import random


class MIMICHelper:
    SAMPLE_RATE = 125
    MIMIC_FILE_PATH = "E:\\毕业论文\\blood_data\\MIMIC\\"
    MIMIC_DATA_PATH = MIMIC_FILE_PATH + "extract\\origin\\"
    ANOMALY_DATA_PATH = MIMIC_FILE_PATH + "extract\\anomaly\\"
    MIMIC_ONE_DATA_PATH = MIMIC_FILE_PATH + "extract\\originOne\\"
    MIMIC_ONE_1000_PATH = MIMIC_ONE_DATA_PATH + "1000\\"
    MIMIC_JAVA_1000_PATH = MIMIC_ONE_DATA_PATH + "java_1000\\"
    MIMIC_TRAIN_DATA_PATH = MIMIC_ONE_DATA_PATH + "train\\"
    MIMIC_TEST_DATA_PATH = MIMIC_ONE_DATA_PATH + "test\\"

    # 新路径
    NEW_ONE_HOME = MIMIC_FILE_PATH + "extract\\newone\\"

    # 读取最原始的MIMIC数据
    @staticmethod
    def readMIMICData():
        # 中心动脉压
        abp = list()
        # PPG
        ppg = list()
        # ECG
        ecg = list()
        for file in os.listdir(MIMICHelper.MIMIC_FILE_PATH):
            path = MIMICHelper.MIMIC_FILE_PATH + file
            if os.path.isdir(path):
                continue
            with h5py.File(path, 'r') as f:
                refs = f['#refs#']
                count = -1
                for key in refs.keys():
                    count += 1
                    # 1728有问题，跳过1700——1750
                    if 1700 < count < 1750:
                        continue
                    # if count == 50:
                    #     break

                    # 转置,3行
                    Tdata = np.array(refs[key]).T
                    # 获取中心动脉压数据
                    abpdata = Tdata[1][:]
                    # ppg原始数据
                    ppgdata = Tdata[0][:]
                    # ECG原始数据
                    ecgdata = Tdata[2][:]
                    if not (abpdata.any() and ppgdata.any() and ecgdata.any()):
                        continue
                    abp.append(abpdata)
                    ppg.append(ppgdata)
                    ecg.append(ecgdata)

        return abp, ppg, ecg

    # 125Hz采样频率计算心率
    @staticmethod
    def heartRateTransfer(distance):
        hz = 125
        heartRate = 60 * hz / distance
        return heartRate

    @staticmethod
    def distanceTransfer(heartRate):
        hz = 125
        distance = 60 * hz / heartRate
        return distance

    # 简单的移动平均滤波
    @staticmethod
    def moveAverage(a, n, mode="same"):
        return np.convolve(a=a, v=np.ones((n,)) / n, mode=mode)

    # 4阶带通滤波器，保留0.5-8Hz
    @staticmethod
    def bindPassFilter(data):
        b, a = signal.butter(4, [0.008, 0.128], 'bandpass')
        return signal.filtfilt(b, a, data)

    # Savitzky–Golay滤波器
    @staticmethod
    def sgFilter(data):
        return signal.savgol_filter(data, 7, 3)

    # 预处理数据
    @staticmethod
    def process():
        abpData, ppgData, ecgData = MIMICHelper.readMIMICData()
        abpResult = []
        ppgResult = []
        ecgResult = []
        YResult = []
        count = 0
        for i in range(len(ppgData)):
            collect = False
            ppgdata = ppgData[i]
            abpdata = abpData[i]
            ecgdata = ecgData[i]
            dataLength = len(ppgdata)
            x = np.arange(dataLength)
            smoothPPG = MIMICHelper.bindPassFilter(ppgdata)  # 对PPG的处理是带通滤波
            smoothABP = MIMICHelper.sgFilter(abpdata)  # 对ABP的处理是Savitzky–Golay滤波器

            # 展示平滑后的ppg
            # Plt.prepare()
            # Plt.figure(1)
            # Plt.subPlot(211)
            # Plt.plotLiner(x[:1000], abpdata[:1000])
            # Plt.subPlot(212)
            # Plt.plotLiner(x[:1000], smoothABP[:1000])
            # Plt.show()

            # 峰值检测，以10s为片段切片
            ppgdataSplit = np.array_split(smoothPPG, math.ceil(dataLength / 1250))
            abpdataSplit = np.array_split(smoothABP, math.ceil(dataLength / 1250))
            ecgdataSplit = np.array_split(ecgdata, math.ceil(dataLength / 1250))
            for j in range(len(ppgdataSplit)):
                # print("开始峰值检测")
                if MIMICHelper.peaksDetect(ppgdataSplit[j]) or MIMICHelper.peaksDetect(abpdataSplit[j]):
                    continue
                # print("开始排除不符合要求的数据")
                if MIMICHelper.excludeAbnormal(abpdataSplit[j]):
                    continue
                abpResult.append(abpdataSplit[j].tolist())
                ppgResult.append(ppgdataSplit[j].tolist())
                ecgResult.append(ecgdataSplit[j].tolist())
                dbp = abpdataSplit[j].min()
                sbp = abpdataSplit[j].max()
                YResult.append([dbp, sbp])
                collect = True
            if collect:
                count += 1
        print("共有{}组数据".format(len(abpResult)))
        print("共有{}个人的数据".format(count))
        FileHelper.writeToFile(abpResult, MIMICHelper.NEW_ONE_HOME + "abp.blood")
        FileHelper.writeToFile(ppgResult, MIMICHelper.NEW_ONE_HOME + "ppg.blood")
        FileHelper.writeToFile(ecgResult, MIMICHelper.NEW_ONE_HOME + "ecg.blood")
        FileHelper.writeToFile(YResult, MIMICHelper.NEW_ONE_HOME + "Y.blood")
        # fig = 1
        # count = 1
        # for i in range(len(result)):
        #     plt.figure(fig, figsize=(12, 8))
        #     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        #     plt.rcParams['axes.unicode_minus'] = False
        #
        #     plt.subplot(8, 4, count)
        #     plt.title("ppg_" + str(i))
        #     plt.plot(result[i])
        #
        #     count += 1
        #     if count % 33 == 0:
        #         count = 1
        #         fig += 1
        #     if fig % 2 == 0:
        #         plt.tight_layout()
        #         fig = 1
        #         plt.show()

    # 峰值检测，返回true说明不合要求
    @staticmethod
    def peaksDetect(data):
        ind = detect_peaks(data, show=False, mpd=50, mph=0)
        peaksValue = np.array([data[i] for i in ind])
        mean = peaksValue.mean()
        std = peaksValue.std()
        func = np.frompyfunc(lambda x: x < mean - 2 * std or x > mean + 2 * std, 1, 1)
        arr = func(peaksValue)
        return arr.__contains__(True)

    # 排除不符合要求的数据，返回true说明不合要求
    @staticmethod
    def excludeAbnormal(data):
        ind_p = detect_peaks(data, show=False, mpd=50, mph=0)
        ind_v = detect_peaks(data, valley=True, show=False, mpd=50)
        peaksValue = np.array([data[i] for i in ind_p])
        func = np.frompyfunc(lambda x: x >= 180 or x < 90, 1, 1)
        arr = func(peaksValue)
        if arr.__contains__(True):
            return True
        valleysValue = np.array([data[i] for i in ind_v])
        func = np.frompyfunc(lambda x: x >= 120 or x < 60, 1, 1)
        arr = func(valleysValue)
        if arr.__contains__(True):
            return True
        if len(ind_v) - 1 <= 5:
            return True
        timeInterval = np.array([ind_v[i + 1] - ind_v[i] for i in range(len(ind_v) - 1)])
        if timeInterval.std() >= 5:
            return True
        if peaksValue.std() >= 5:
            return True
        for i in range(len(peaksValue) - 1):
            if peaksValue[i + 1] - peaksValue[i] > 10:
                return True
        return False

    @staticmethod
    def splitDataset():
        abpdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "abp.blood")
        ppgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ppg.blood")
        ecgdata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "ecg.blood")
        Ydata = FileHelper.readFromFileFloat(MIMICHelper.NEW_ONE_HOME + "Y.blood")
        abp_train = []
        abp_valid = []
        abp_test = []
        ppg_train = []
        ppg_valid = []
        ppg_test = []
        ecg_train = []
        ecg_valid = []
        ecg_test = []
        Y_train = []
        Y_valid = []
        Y_test = []

        randomNumber = [i for i in range(len(Ydata))]
        random.shuffle(randomNumber)
        for i in randomNumber:
            abp = abpdata[i]
            ppg = ppgdata[i]
            ecg = ecgdata[i]
            Y = Ydata[i]
            length = min(len(abp), len(ppg), len(ecg))
            if length < 512:
                continue
            randStart = random.randint(0, length - 512)
            abp = abp[randStart:randStart + 512]
            ppg = ppg[randStart:randStart + 512]
            ecg = ecg[randStart:randStart + 512]
            rand = random.randint(1, 10)
            if rand <= 7:
                abp_train.append(abp)
                ppg_train.append(ppg)
                ecg_train.append(ecg)
                Y_train.append(Y)
            elif rand <= 8:
                abp_valid.append(abp)
                ppg_valid.append(ppg)
                ecg_valid.append(ecg)
                Y_valid.append(Y)
            else:
                abp_test.append(abp)
                ppg_test.append(ppg)
                ecg_test.append(ecg)
                Y_test.append(Y)
        FileHelper.writeToFile(abp_train, MIMICHelper.NEW_ONE_HOME + "abp_train.blood")
        FileHelper.writeToFile(ppg_train, MIMICHelper.NEW_ONE_HOME + "ppg_train.blood")
        FileHelper.writeToFile(ecg_train, MIMICHelper.NEW_ONE_HOME + "ecg_train.blood")
        FileHelper.writeToFile(Y_train, MIMICHelper.NEW_ONE_HOME + "Y_train.blood")
        FileHelper.writeToFile(abp_valid, MIMICHelper.NEW_ONE_HOME + "abp_valid.blood")
        FileHelper.writeToFile(ppg_valid, MIMICHelper.NEW_ONE_HOME + "ppg_valid.blood")
        FileHelper.writeToFile(ecg_valid, MIMICHelper.NEW_ONE_HOME + "ecg_valid.blood")
        FileHelper.writeToFile(Y_valid, MIMICHelper.NEW_ONE_HOME + "Y_valid.blood")
        FileHelper.writeToFile(abp_test, MIMICHelper.NEW_ONE_HOME + "abp_test.blood")
        FileHelper.writeToFile(ppg_test, MIMICHelper.NEW_ONE_HOME + "ppg_test.blood")
        FileHelper.writeToFile(ecg_test, MIMICHelper.NEW_ONE_HOME + "ecg_test.blood")
        FileHelper.writeToFile(Y_test, MIMICHelper.NEW_ONE_HOME + "Y_test.blood")


# 心率50-150的话，两个波峰最小距离为50（采样频率125hz）
if __name__ == "__main__":
    s_t = time.time()
    # MIMICHelper.process()
    MIMICHelper.splitDataset()
    e_t = time.time()
    print("耗时{}".format(e_t - s_t))
