from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
import KmeansPlus
from scipy import signal
import numpy as np
from scipy.fftpack import fft, ifft
from SphygmoCorData import SphygmoCorHelper

if __name__ == "__main__":
    mimicHelper = MIMICHelper()
    sphygmoCorHelper = SphygmoCorHelper()

    # 读取原始的未处理的abp和ppg波形
    # abp_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_abp.blood")
    # ppg_data = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_DATA_PATH + "one_ppg.blood")
    # bbp_data, abp_data = sphygmoCorHelper.readSphygmoCorData()
    bbp_data = mimicHelper.readFromFileFloat(sphygmoCorHelper.SPHYGMOCOR_TRAIN_PATH + "bbp_train_73.blood")
    abp_data = mimicHelper.readFromFileFloat(sphygmoCorHelper.SPHYGMOCOR_TRAIN_PATH + "abp_train_73.blood")
    test_bbp_data = mimicHelper.readFromFileFloat(sphygmoCorHelper.SPHYGMOCOR_TEST_PATH + "bbp_test_73.blood")
    test_abp_data = mimicHelper.readFromFileFloat(sphygmoCorHelper.SPHYGMOCOR_TEST_PATH + "abp_test_73.blood")

    # 读取ppg聚类中心波形
    # centers = mimicHelper.readFromFileFloat(mimicHelper.MIMIC_ONE_1000_PATH + "center.cluster")
    centers = mimicHelper.readFromFileFloat(sphygmoCorHelper.JAVA_1000_PATH + "center.cluster")

    # 读取子类索引
    # cluster_index = list()
    # for i in range(1000):
    #     index = mimicHelper.readFromFileInteger(mimicHelper.MIMIC_ONE_1000_PATH + str(i) + ".cluster")
    #     cluster_index.append(index)
    cluster_index = list()
    for i in range(1000):
        index = mimicHelper.readFromFileInteger(sphygmoCorHelper.JAVA_1000_PATH + str(i) + ".cluster")
        cluster_index.append(index)

    # resample至125个点
    N = 125
    abp_data_125 = list()
    bbp_data_125 = list()
    test_abp_data_125 = list()
    test_bbp_data_125 = list()
    for i in range(len(abp_data)):
        abp_125 = signal.resample(abp_data[i], N).tolist()
        bbp_125 = signal.resample(bbp_data[i], N).tolist()
        test_abp_125 = signal.resample(abp_data[i], N).tolist()
        test_bbp_125 = signal.resample(bbp_data[i], N).tolist()
        abp_data_125.append(abp_125)
        bbp_data_125.append(bbp_125)
        test_abp_data_125.append(test_abp_125)
        test_bbp_data_125.append(test_bbp_125)
    t = np.linspace(0.0, 2 * np.pi, N)

    # 计算聚类中心对应的中心动脉压的平均波形
    centers_abp = list()
    for i in range(len(cluster_index)):
        num = len(cluster_index[i])
        abp_center = np.zeros(N)
        for j in range(num):
            abp_center = abp_center + np.array(abp_data_125[cluster_index[i][j]])
        abp_center = abp_center / num
        centers_abp.append(abp_center.tolist())
    # 计算全部中心动脉压的平均波形
    all_centers_abp = np.array(centers_abp).mean(axis=0).tolist()

    # 计算传递函数 f1 = ppg / abp, 预测的时候 abp_y = ppg / f
    f1 = list()
    for i in range(len(centers_abp)):
        _f = [x / y for x, y in zip(centers[i], centers_abp[i])]
        f1.append(_f)

    # 计算传递函数f2 利用傅里叶变换转换为频率 每个聚类计算一个传递函数
    f2_abs_common = list()
    f2_angle_common = list()
    for i in range(len(cluster_index)):
        row = len(cluster_index[i])
        fft_ABP = list()
        fft_PPG = list()
        for j in range(row):
            fft_ABP.append(fft(abp_data_125[cluster_index[i][j]]))
            fft_PPG.append(fft(bbp_data_125[cluster_index[i][j]]))
        # 以1HZ为单位，计算全部模和幅角的均值
        abs_abp = np.zeros(N)
        angle_abp = np.zeros(N)
        abs_ppg = np.zeros(N)
        angle_ppg = np.zeros(N)
        for j in range(row):
            tmp_abs = np.abs(fft_ABP[j])
            tmp_abs = tmp_abs / N * 2
            tmp_abs[0] /= 2
            abs_abp += tmp_abs
            tmp_abs = np.abs(fft_PPG[j])
            tmp_abs = tmp_abs / N * 2
            tmp_abs[0] /= 2
            abs_ppg += tmp_abs
            angle_abp += np.angle(fft_ABP[j])
            angle_ppg += np.angle(fft_PPG[j])
        abs_abp_mean = abs_abp / row
        abs_ppg_mean = abs_ppg / row
        angle_abp_mean = angle_abp / row
        angle_ppg_mean = angle_ppg / row
        # 计算通用传递函数 ppg/abp 模相除，相位相减
        abs_common = np.divide(abs_ppg_mean, abs_abp_mean,
                               # out=np.array([9999999] * N, dtype='float64'),
                               # where=abs_abp_mean != 0
                               )
        angle_common = angle_ppg_mean - angle_abp_mean
        f2_abs_common.append(abs_common)
        f2_angle_common.append(angle_common)

    # 计算传递函数f3 对照组 论文里的方法 计算通用的传递函数
    f3_ppg_fft_abs = np.zeros(N)
    f3_ppg_fft_angel = np.zeros(N)
    f3_abp_fft_abs = np.zeros(N)
    f3_abp_fft_angel = np.zeros(N)
    for i in range(len(bbp_data_125)):
        f3_ppg_fft = fft(bbp_data_125[i])
        f3_abp_fft = fft(abp_data_125[i])
        # abs要除以N/2
        tmp_abs = np.abs(f3_ppg_fft)
        tmp_abs = tmp_abs / N * 2
        tmp_abs[0] /= 2
        f3_ppg_fft_abs += tmp_abs
        tmp_abs = np.abs(f3_abp_fft)
        tmp_abs = tmp_abs / N * 2
        tmp_abs[0] /= 2
        f3_abp_fft_abs += tmp_abs
        f3_ppg_fft_angel += np.angle(f3_ppg_fft)
        f3_abp_fft_angel += np.angle(f3_abp_fft)
    f3_ppg_fft_abs_mean = f3_ppg_fft_abs / len(bbp_data_125)
    f3_ppg_fft_angel_mean = f3_ppg_fft_angel / len(bbp_data_125)
    f3_abp_fft_abs_mean = f3_abp_fft_abs / len(bbp_data_125)
    f3_abp_fft_angel_mean = f3_abp_fft_angel / len(bbp_data_125)
    f3_abs_common = np.divide(f3_ppg_fft_abs_mean, f3_abp_fft_abs_mean,
                              # out=np.array([9999999] * N, dtype='float64'),
                              # where=f3_abp_fft_abs_mean != 0
                              )
    f3_angel_common = f3_ppg_fft_angel_mean - f3_abp_fft_angel_mean

    # 预测中心动脉压，拿测试集去预测
    # dis = predict - origin 差异，计算预测模型的准确度，进行比较
    # DBP 中心动脉舒张压，起点位置
    # SBP 中心动脉收缩压，最高点位置
    f1_DBP_dis_array = []
    f2_DBP_dis_array = []
    f3_DBP_dis_array = []
    f1_SBP_dis_array = []
    f2_SBP_dis_array = []
    f3_SBP_dis_array = []
    start_time = time.time()
    percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    percentage_i = 0
    for i in range(len(abp_data)):
        if i == percentage[percentage_i] * len(abp_data) - 1:
            end_time = time.time()
            print("计算了" + str(percentage[percentage_i] * 100) + "%数据，耗时：" + str(end_time - start_time))
            percentage_i += 1
        origin_ppg = test_bbp_data_125[i]  # 测试集原始ppg
        origin_abp = test_abp_data_125[i]  # 测试集原始abp
        # 计算该ppg属于哪一类聚类，索引值是index
        min_dis = 99999
        index = 0
        for j in range(len(centers)):
            dis = KmeansPlus.distance(origin_ppg, centers[j])
            if dis < min_dis:
                min_dis = dis
                index = j
        # f1预测
        predict_abp_f1 = np.divide(np.array(origin_ppg), np.array(f1[index]),
                                   out=np.array(origin_abp, dtype='float64'),
                                   where=np.array(f1[index]) != 0)

        origin_ppg_fft = fft(origin_ppg)
        origin_ppg_abs = np.abs(origin_ppg_fft)
        origin_ppg_angel = np.angle(origin_ppg_fft)
        tmp_abs = origin_ppg_abs / N * 2
        tmp_abs[0] /= 2
        # f2预测
        f2_common_abs = np.array(f2_abs_common[index])
        f2_common_angel = np.array(f2_angle_common[index])
        f2_predict_abs = np.divide(tmp_abs, f2_common_abs,
                                   # out=np.array(centers_abp[index], dtype='float64'),
                                   # where=f2_common_abs != 0
                                   )  # 用聚类中心的平均中心动脉压作为预测的补充值
        f2_predict_angel = origin_ppg_angel - f2_common_angel
        predict_abp_f2 = f2_predict_abs[0]
        for k in range(1, 11):
            predict_abp_f2 += f2_predict_abs[k] * np.cos(k * t + f2_predict_angel[k])
        # f3预测
        f3_predict_abs = np.divide(tmp_abs, f3_abs_common,
                                   # out=np.array(all_centers_abp, dtype='float64'),
                                   # where=f3_abs_common != 0
                                   )
        f3_predict_angel = origin_ppg_angel - f3_angel_common
        predict_abp_f3 = f3_predict_abs[0]
        for k in range(1, len(f3_predict_abs)):
            predict_abp_f3 += f3_predict_abs[k] * np.cos(k * t + f3_predict_angel[k])

        # 展示一下各个预测值
        # plt.figure(figsize=(8, 4), dpi=200)
        # plt.suptitle(str(i), fontsize=16)
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # plt.subplot(2, 1, 1)
        # plt.plot(origin_ppg, label="初始ppg")
        # plt.plot(centers[index], label="最近的聚类中心")
        # plt.legend(loc='upper right', fontsize=6)
        # plt.subplot(2, 1, 2)
        # plt.plot(origin_abp, label="初始中心动脉压", color="black")
        # plt.plot(predict_abp_f1, label="f1预测的中心动脉压")
        # plt.plot(predict_abp_f2, label="f2预测的中心动脉压")
        # plt.plot(predict_abp_f3, label="f3预测的中心动脉压")
        # plt.legend(loc='upper right', fontsize=6)
        # plt.show()

        # 计算指标差异性，互相比较
        origin_DBP_value = origin_abp[0]  # 原数据DBP
        # peaks, properties = signal.find_peaks(origin_abp)
        # origin_SBP_value = max([origin_abp[index] for index in peaks])  # 原数据SBP
        origin_SBP_value = max(origin_abp)

        f1_DBP_value = predict_abp_f1[0]  # f1预测的DBP
        # peaks, properties = signal.find_peaks(predict_abp_f1)
        # f1_SBP_value = max([predict_abp_f1[index] for index in peaks])  # f1预测的SBP
        f1_SBP_value = max(predict_abp_f1)
        f1_DBP_dis_array.append(abs(f1_DBP_value - origin_DBP_value))
        f1_SBP_dis_array.append(abs(f1_SBP_value - origin_SBP_value))

        f2_DBP_value = predict_abp_f2[0]  # f2预测的DBP
        # peaks, properties = signal.find_peaks(predict_abp_f2)
        # f2_SBP_value = max([predict_abp_f2[index] for index in peaks])  # f2预测的SBP
        f2_SBP_value = max(predict_abp_f2)
        f2_DBP_dis_array.append(abs(f2_DBP_value - origin_DBP_value))
        f2_SBP_dis_array.append(abs(f2_SBP_value - origin_SBP_value))

        f3_DBP_value = predict_abp_f3[0]  # f3预测的DBP
        # peaks, properties = signal.find_peaks(predict_abp_f3)
        # f3_SBP_value = max([predict_abp_f3[index] for index in peaks])  # f3预测的SBP
        f3_SBP_value = max(predict_abp_f3)
        f3_DBP_dis_array.append(abs(f3_DBP_value - origin_DBP_value))
        f3_SBP_dis_array.append(abs(f3_SBP_value - origin_SBP_value))
    end_time = time.time()
    print("预测时间：" + str(end_time - start_time))

    # 计算平均值、方差等
    f1_DBP_dis_mean = np.mean(f1_DBP_dis_array)
    f1_DBP_dis_std = np.std(f1_DBP_dis_array)
    f1_SBP_dis_mean = np.mean(f1_SBP_dis_array)
    f1_SBP_dis_std = np.std(f1_SBP_dis_array)

    f2_DBP_dis_mean = np.mean(f2_DBP_dis_array)
    f2_DBP_dis_std = np.std(f2_DBP_dis_array)
    f2_SBP_dis_mean = np.mean(f2_SBP_dis_array)
    f2_SBP_dis_std = np.std(f2_SBP_dis_array)

    f3_DBP_dis_mean = np.mean(f3_DBP_dis_array)
    f3_DBP_dis_std = np.std(f3_DBP_dis_array)
    f3_SBP_dis_mean = np.mean(f3_SBP_dis_array)
    f3_SBP_dis_std = np.std(f3_SBP_dis_array)
    print("预测结果以 均值±标准差 的形式表示")
    print("f1预测的结果：1.DBP：" + str(f1_DBP_dis_mean) + "±" + str(f1_DBP_dis_std) + " 2.SBP：" + str(
        f1_SBP_dis_mean) + "±" + str(
        f1_SBP_dis_std))
    print("f2预测的结果：1.DBP：" + str(f2_DBP_dis_mean) + "±" + str(f2_DBP_dis_std) + " 2.SBP：" + str(
        f2_SBP_dis_mean) + "±" + str(
        f2_SBP_dis_std))
    print("f3预测的结果：1.DBP：" + str(f3_DBP_dis_mean) + "±" + str(f3_DBP_dis_std) + " 2.SBP：" + str(
        f3_SBP_dis_mean) + "±" + str(
        f3_SBP_dis_std))
