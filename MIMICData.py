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
    MIMIC_ONE_1000_PATH = MIMIC_ONE_DATA_PATH + "1000\\"

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

        return abp, ppg

    @staticmethod
    def writeToFile(data, filename):
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

    @staticmethod
    def writeToFile2(data, filename):
        with open(filename, 'w') as f:
            total = len(data)
            print("写到(" + filename + ")的原始数据一共:" + str(total) + "行")
            cur = 1
            for i in range(total):
                if i > 0:
                    f.write("\n")
                f.write(str(data[i]))
                cur += 1
                if cur % 200 == 0:
                    print("写了：" + str((cur / total) * 100) + "%数据，时间：" + time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                       time.localtime(time.time())))

    @staticmethod
    def readFromFileFloat(filename):
        ret = list()
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')
                data = [float(d) for d in line]
                ret.append(data)
        return ret

    @staticmethod
    def readFromFileInteger(filename):
        ret = list()
        with open(filename, 'r') as f:
            for line in f.readlines():
                line.strip()
                ret.append(int(line))
        return ret
