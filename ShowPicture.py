from MIMICData import MIMICHelper
from SphygmoCorData import SphygmoCorHelper
import matplotlib.pyplot as plt
import time
from AnomalyDetector import AnomalyDetector
from scipy import signal
from WaveletDenoising import wavelet_noising

if __name__ == "__main__":
    # mimicHelper = MIMICHelper()
    start_time = time.time()
    # abp_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_DATA_PATH + "abp.blood")
    # ppg_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_DATA_PATH + "ppg.blood")
    # invalid_index = mimicHelper.readFromFileInteger(mimicHelper.ANOMALY_DATA_PATH + "invalid_index.blood")
    bbp_data, abp_data = SphygmoCorHelper.readSphygmoCorData()
    end_time = time.time()
    # print("行：" + str(len(abp_data)))  #11808
    # print("列：" + str(len(abp_data[0])))  #1000
    print("读取数据耗时：" + str(end_time - start_time))  # 45.54142904281616

    # anomalyDetector = AnomalyDetector()
    fig = 1
    count = 1
    zero = 1
    for i in range(len(abp_data)):
        # 跳过异常值
        # if i in invalid_index:
        #     continue
        # 识别异常值
        # anomalyDetector.setData(ppg_data[i])
        # detect_ret = anomalyDetector.SHESDdetect()
        # if len(detect_ret) > 0:
        #     invalid_index.append(i)
        # if i % 100 == 0:
        #     print("处理了" + str(i/11808) + "%数据")

        # 找ppg波谷
        # peaks_index1 = signal.find_peaks(ppg_data[i], distance=45)[0]
        # if len(peaks_index1) == 0:
        #     continue
        # distance1 = (peaks_index1[-1] - peaks_index1[0]) / (len(peaks_index1) - 1)
        # distance1 = int(distance1) - 5
        # valleys_index1 = signal.find_peaks([-x for x in ppg_data[i]], distance=distance1)[0]
        # if len(valleys_index1) < 5:
        #     continue

        # 找abp波谷
        # peaks_index2 = signal.find_peaks(abp_data[i], distance=45)[0]
        # if len(peaks_index2) == 0:
        #     continue
        # distance2 = (peaks_index2[-1] - peaks_index2[0]) / (len(peaks_index2) - 1)
        # distance2 = int(distance2) - 5
        # valleys_index2 = signal.find_peaks([-x for x in abp_data[i]], distance=distance2)[0]
        # if len(valleys_index2) < 5:
        #     continue

        plt.figure(fig, figsize=(12, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(8, 4, count)
        plt.title("ppg " + str(i))
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(bbp_data[i])
        # for j in range(len(valleys_index1)):
        # plt.plot(valleys_index1[2], ppg_data[i][valleys_index1[2]], 'o', color='red')
        # plt.plot(valleys_index1[3], ppg_data[i][valleys_index1[3]], 'o', color='red')

        plt.subplot(8, 4, count + 4)
        plt.title("abp " + str(i))
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(abp_data[i])
        # for j in range(len(valleys_index2)):
        # plt.plot(valleys_index2[2], abp_data[i][valleys_index2[2]], 'o', color='red')
        # plt.plot(valleys_index2[3], abp_data[i][valleys_index2[3]], 'o', color='red')

        count += 1
        zero += 1
        if zero % 5 == 0:
            count += 4
            zero = 1
        if count % 33 == 0:
            count = 1
            fig += 1
        if fig % 21 == 0:
            # plt.tight_layout()
            plt.legend()
            fig = 1
            plt.show()
        # plt.text(t[index_tB] + 0.25, bbp[index_tB] + 1, 'B', ha='center', va='bottom', fontsize=10.5)

    # 写异常值的索引进文件里
    # filename = mimicHelper.ANOMALY_DATA_PATH + "invalid_index.blood"
    # with open(filename, 'w') as f:
    #     print("写到(" + filename + ")的原始数据一共:" + str(len(invalid_index)) + "行")
    #     for i in range(len(invalid_index)):
    #         if i > 0:
    #             f.write("\n")
    #         f.write(str(invalid_index[i]))
