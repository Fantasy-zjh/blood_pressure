from MIMICData import MIMICHelper
from SphygmoCorData import SphygmoCorHelper
import matplotlib.pyplot as plt
import time
from AnomalyDetector import AnomalyDetector
from scipy import signal
from WaveletDenoising import wavelet_noising
from FileHelper import FileHelper
from detecta import detect_peaks

if __name__ == "__main__":
    start_time = time.time()
    abp_data = FileHelper.readFromFileFloat(MIMICHelper.NEW_CLUSTER + "abp_train.blood")
    ppg_data = FileHelper.readFromFileFloat(MIMICHelper.NEW_CLUSTER + "ppg_train.blood")
    # invalid_index = mimicHelper.readFromFileInteger(mimicHelper.ANOMALY_DATA_PATH + "invalid_index.blood")
    # bbp_data, abp_data = SphygmoCorHelper.readSphygmoCorData()
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
        # if i not in invalid_index:
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

        # # 找波峰还算准确
        # ppg_ind_p = detect_peaks(ppg_data[i], valley=False, show=False, mpd=50)
        # abp_ind_p = detect_peaks(abp_data[i], valley=False, show=False, mpd=50)
        # # 波峰前是波谷
        # ppg_ind_v = []
        # abp_ind_v = []
        # jump = False
        # for index in ppg_ind_p:
        #     v_index = index
        #     for j in range(index-1, -1, -1):
        #         if ppg_data[i][j] < ppg_data[i][j+1]:
        #             v_index = j
        #         else:
        #             break
        #     if v_index != index and abs(v_index - index) > 10:
        #         ppg_ind_v.append(v_index)
        #     else:
        #         jump = True
        # for index in abp_ind_p:
        #     v_index = index
        #     for j in range(index-1, -1, -1):
        #         if abp_data[i][j] < abp_data[i][j+1]:
        #             v_index = j
        #         else:
        #             break
        #     if v_index != index and abs(v_index - index) > 10:
        #         abp_ind_v.append(v_index)
        #     else:
        #         jump = True
        # if jump:
        #     continue

        plt.subplot(8, 4, count)
        plt.title("ppg_" + str(i))
        # plt.ylabel('P/mmHg')
        plt.plot(ppg_data[i])
        # for j in range(len(ppg_ind_v)):
        #     plt.plot(ppg_ind_v[j], ppg_data[i][ppg_ind_v[j]], 'o', color='red')

        plt.subplot(8, 4, count + 4)
        plt.title("abp_" + str(i))
        # plt.ylabel('P/mmHg')
        plt.plot(abp_data[i])
        # for j in range(len(abp_ind_v)):
        #     plt.plot(abp_ind_v[j], abp_data[i][abp_ind_v[j]], 'o', color='red')

        count += 1
        zero += 1
        if zero % 5 == 0:
            count += 4
            zero = 1
        if count % 33 == 0:
            count = 1
            fig += 1
        if fig % 2 == 0:
            plt.tight_layout()
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
