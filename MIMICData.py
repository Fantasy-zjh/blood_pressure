import os
import h5py
import numpy as np
from scipy import signal
import time


class MIMICHelper:
    SAMPLE_RATE = 125
    MIMIC_FILE_PATH = "E:\\毕业论文\\blood_data\\MIMIC\\"
    MIMIC_DATA_PATH = MIMIC_FILE_PATH + "extract\\origin\\"
    ANOMALY_DATA_PATH = MIMIC_FILE_PATH + "extract\\anomaly\\"
    MIMIC_ONE_DATA_PATH = MIMIC_FILE_PATH + "extract\\originOne\\"

    def readMIMICData(self):
        # 中心动脉压
        abp = list()
        # PPG
        ppg = list()
        for file in os.listdir(self.MIMIC_FILE_PATH):
            path = self.MIMIC_FILE_PATH + file
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
                    # 获取中心动脉压数据，原始太大只取前600ms的数据
                    abpdata = Tdata[1][:]
                    # abpdata = Tdata[1]
                    # ppg原始数据
                    ppgdata = Tdata[0][:]
                    # ppgdata = Tdata[0]
                    if not (abpdata.any() and ppgdata.any()):
                        continue
                    abp.append(abpdata)
                    ppg.append(ppgdata)

                    # 去除基线漂移
                    # abpdata = self.smooth(abpdata)
                    # ppgdata = self.smooth(ppgdata)

                    # 周期定为65ms，设置两个波峰之间的距离不小于60
                    # abppeaks = signal.find_peaks(abpdata, distance=60)
                    # ppgpeaks = signal.find_peaks(ppgdata, distance=60)
                    # if len(abppeaks[0]) == 0 or len(ppgpeaks[0]) == 0:
                    #     continue
                    # select = random.randint(1, len(abppeaks[0]) - 2)
                    # peak = abppeaks[0][select]
                    # abp.append(abpdata[peak - 20: peak + 45])
                    # select = random.randint(1, len(ppgpeaks[0]) - 2)
                    # peak = ppgpeaks[0][select]
                    # ppg.append(ppgdata[peak - 20: peak + 45])
            # break
        return abp, ppg

    # 合成的方法放弃
    def mixToOne(self, data):
        ret = list()
        for listOfData in data:
            mixedData = [0] * self.SAMPLE_RATE
            valleys = signal.find_peaks([-x for x in listOfData], distance=60)[0]
            length = len(valleys)
            for i in range(0, length - 1):
                oldData = listOfData[valleys[i]:valleys[i + 1]]
                newData = signal.resample(oldData, self.SAMPLE_RATE)
                for j in range(self.SAMPLE_RATE):
                    mixedData[j] += newData[j]
            mixedData = [x / (length - 1) for x in mixedData]
            ret.append(mixedData)
        return ret

    def getOne(self, abp_data, ppg_data):
        abp_ret = list()
        ppg_ret = list()
        for index in range(len(abp_data)):
            listOfData = abp_data[index]
            peaks = signal.find_peaks(listOfData, distance=60)[0]

    def writeToFile(self, data, filename):
        with open(filename, 'w') as f:
            total = len(data)
            print("写到(" + filename + ")的原始数据一共:" + str(total) + "行")
            cur = 1
            for data_list in data:
                size = len(data_list)
                for i in range(size):
                    if i > 0:
                        f.write(" ")
                    f.write(str(data_list[i]))
                f.write("\n")
                cur += 1
                if cur % 200 == 0:
                    print("写了：" + str((cur / total) * 100) + "%数据，时间：" + time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                       time.localtime(time.time())))

    def readFromFileFloat(self, filename):
        ret = list()
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')
                data = [float(d) for d in line]
                ret.append(data)
        return ret

    def readFromFileInteger(self, filename):
        ret = list()
        with open(filename, 'r') as f:
            for line in f.readlines():
                line.strip()
                ret.append(int(line))
        return ret
