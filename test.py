from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
from AnomalyDetector import AnomalyDetector
from scipy import signal


if __name__ == "__main__":
    mimicHelper = MIMICHelper()
    start_time = time.time()
    # abp_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_DATA_PATH + "abp.blood")
    # ppg_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_DATA_PATH + "ppg.blood")
    # invalid_index = mimicHelper.readFromFileInteger(mimicHelper.ANOMALY_DATA_PATH + "invalid_index.blood")
    ppg_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_ppg.blood")
    abp_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_abp.blood")

    end_time = time.time()
    print("读取数据耗时：" + str(end_time - start_time))

    fig = 1
    count = 1
    zero = 1
    for i in range(len(ppg_data)):
        plt.figure(fig, figsize=(12, 8))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

        plt.subplot(8, 4, count)
        plt.title("ppg " + str(i))
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(ppg_data[i])

        plt.subplot(8, 4, count + 4)
        plt.title("abp " + str(i))
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(abp_data[i])

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

    # write_data_abp = list()
    # write_data_ppg = list()
    # for i in range(len(ppg_data)):
    #     # 排除异常值
    #     if i in invalid_index:
    #         continue
    #
    #     # ppg
    #     peaks_index1 = signal.find_peaks(ppg_data[i], distance=45)[0]
    #     if len(peaks_index1) == 0:
    #         continue
    #     distance1 = (peaks_index1[-1] - peaks_index1[0]) / (len(peaks_index1) - 1)
    #     distance1 = int(distance1) - 5
    #     valleys_index1 = signal.find_peaks([-x for x in ppg_data[i]], distance=distance1)[0]
    #     if len(valleys_index1) < 5:
    #         continue
    #
    #     # abp
    #     peaks_index2 = signal.find_peaks(abp_data[i], distance=45)[0]
    #     if len(peaks_index2) == 0:
    #         continue
    #     distance2 = (peaks_index2[-1] - peaks_index2[0]) / (len(peaks_index2) - 1)
    #     distance2 = int(distance2) - 5
    #     valleys_index2 = signal.find_peaks([-x for x in abp_data[i]], distance=distance2)[0]
    #     if len(valleys_index2) < 5:
    #         continue
    #
    #     one_ppg = ppg_data[i][valleys_index1[2]:valleys_index1[3]]
    #     one_abp = abp_data[i][valleys_index2[2]:valleys_index2[3]]
    #     write_data_ppg.append(one_ppg)
    #     write_data_abp.append(one_abp)
    # mimicHelper.writeToFile(write_data_ppg, mimicHelper.MIMIC_ONE_DATA_PATH + "one_ppg.blood")
    # mimicHelper.writeToFile(write_data_abp, mimicHelper.MIMIC_ONE_DATA_PATH + "one_abp.blood")