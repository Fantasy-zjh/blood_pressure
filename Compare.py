from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
import KmeansPlus
from scipy import signal
import numpy as np
from scipy.fftpack import fft, ifft
from SphygmoCorData import SphygmoCorHelper
from Plt import Plt
from scipy.stats import pearsonr, ttest_rel
from FileHelper import FileHelper

if __name__ == "__main__":
    mimicHelper = MIMICHelper()
    sphygmoCorHelper = SphygmoCorHelper()

    # 读取原始的未处理的abp和ppg波形
    readPath = mimicHelper.NEW_CLUSTER_ORIGINAL
    ppg_data = FileHelper.readFromFileFloat(readPath + "ppg_train.blood")
    abp_data = FileHelper.readFromFileFloat(readPath + "abp_train.blood")
    test_ppg_data = FileHelper.readFromFileFloat(readPath + "ppg_test.blood")
    test_abp_data = FileHelper.readFromFileFloat(readPath + "abp_test.blood")
    # print(len(ppg_data[0]))
    # print(len(test_ppg_data[0]))

    cluster_num = 5000
    # 读取ppg聚类中心波形
    centers = FileHelper.readFromFileFloat(readPath + "java_" + str(cluster_num) + "\\center.cluster")

    # 读取子类索引
    cluster_index = list()
    for i in range(cluster_num):
        index = FileHelper.readFromFileInteger(
            readPath + "java_" + str(cluster_num) + "\\" + str(i) + ".cluster")
        cluster_index.append(index)

    # resample至125个点
    N = 125
    # abp_data_125 = list()
    # ppg_data_125 = list()
    # test_abp_data_125 = list()
    # test_ppg_data_125 = list()
    # for i in range(len(abp_data)):
    #     abp_125 = signal.resample(abp_data[i], N).tolist()
    #     ppg_125 = signal.resample(ppg_data[i], N).tolist()
    #     test_abp_125 = signal.resample(abp_data[i], N).tolist()
    #     test_ppg_125 = signal.resample(ppg_data[i], N).tolist()
    #     abp_data_125.append(abp_125)
    #     ppg_data_125.append(ppg_125)
    #     test_abp_data_125.append(test_abp_125)
    #     test_ppg_data_125.append(test_ppg_125)
    t = np.linspace(0.0, 2 * np.pi, N)

    # 计算聚类中心对应的中心动脉压的平均波形
    centers_abp = list()
    for i in range(len(cluster_index)):
        num = len(cluster_index[i])
        abp_center = np.zeros(N)
        for j in range(num):
            abp_center = abp_center + np.array(abp_data[cluster_index[i][j]])
        abp_center = abp_center / num
        centers_abp.append(abp_center.tolist())
    # 计算全部中心动脉压的平均波形
    all_centers_abp = np.array(centers_abp).mean(axis=0).tolist()

    # f1 移动平均法

    # 计算传递函数f2 自己的方法 利用傅里叶变换转换为频率 每个聚类计算一个传递函数
    f2_abs_common = list()
    f2_angle_common = list()
    for i in range(len(cluster_index)):
        row = len(cluster_index[i])
        fft_ABP = list()
        fft_PPG = list()
        for j in range(row):
            fft_ABP.append(fft(abp_data[cluster_index[i][j]]))
            fft_PPG.append(fft(ppg_data[cluster_index[i][j]]))
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
                               out=np.array([9999999] * N, dtype='float64'),
                               where=abs_abp_mean != 0
                               )
        angle_common = angle_ppg_mean - angle_abp_mean
        f2_abs_common.append(abs_common)
        f2_angle_common.append(angle_common)

    # 计算传递函数f3 范琳琳论文里的方法 计算通用的传递函数
    f3_ppg_fft_abs = np.zeros(N)
    f3_ppg_fft_angel = np.zeros(N)
    f3_abp_fft_abs = np.zeros(N)
    f3_abp_fft_angel = np.zeros(N)
    for i in range(len(abp_data)):
        f3_ppg_fft = fft(ppg_data[i])
        f3_abp_fft = fft(abp_data[i])
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
    f3_ppg_fft_abs_mean = f3_ppg_fft_abs / len(abp_data)
    f3_ppg_fft_angel_mean = f3_ppg_fft_angel / len(abp_data)
    f3_abp_fft_abs_mean = f3_abp_fft_abs / len(abp_data)
    f3_abp_fft_angel_mean = f3_abp_fft_angel / len(abp_data)
    f3_abs_common = np.divide(f3_ppg_fft_abs_mean, f3_abp_fft_abs_mean,
                              out=np.array([9999999] * N, dtype='float64'),
                              where=f3_abp_fft_abs_mean != 0
                              )
    f3_angel_common = f3_ppg_fft_angel_mean - f3_abp_fft_angel_mean

    # 计算传递函数f4 吴樟洋论文方法 GTF法
    # 时域传递函数
    f4_transfer_function = list()
    for i in range(len(cluster_index)):
        transfer = np.divide(centers[i], centers_abp[i])
        f4_transfer_function.append(transfer)
    # 预测中心动脉压，拿测试集去预测
    # AE = |est - true| 绝对误差
    # RE = |est - true| / true 相对误差
    # DBP 中心动脉舒张压，起点位置
    # SBP 中心动脉收缩压，最高点位置
    # PP 脉压差，SBP - DBP
    f1_DBP_AE_array = []
    f1_DBP_RE_array = []
    f1_SBP_AE_array = []
    f1_SBP_RE_array = []
    f1_PP_AE_array = []
    f1_PP_RE_array = []

    f2_DBP_AE_array = []
    f2_DBP_RE_array = []
    f2_SBP_AE_array = []
    f2_SBP_RE_array = []
    f2_PP_AE_array = []
    f2_PP_RE_array = []

    f3_DBP_AE_array = []
    f3_DBP_RE_array = []
    f3_SBP_AE_array = []
    f3_SBP_RE_array = []
    f3_PP_AE_array = []
    f3_PP_RE_array = []

    f4_DBP_AE_array = []
    f4_DBP_RE_array = []
    f4_SBP_AE_array = []
    f4_SBP_RE_array = []
    f4_PP_AE_array = []
    f4_PP_RE_array = []

    o_DBP_array = []
    o_SBP_array = []
    o_PP_array = []
    f1_DBP_array = []
    f1_SBP_array = []
    f1_PP_array = []
    f2_DBP_array = []
    f2_SBP_array = []
    f2_PP_array = []
    f3_DBP_array = []
    f3_SBP_array = []
    f3_PP_array = []
    f4_DBP_array = []
    f4_SBP_array = []
    f4_PP_array = []

    start_time = time.time()
    percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    percentage_i = 0
    for i in range(len(test_abp_data)):
        if i == int(percentage[percentage_i] * len(test_abp_data)) - 1:
            end_time = time.time()
            print("计算了" + str(percentage[percentage_i] * 100) + "%数据，耗时：" + str(end_time - start_time))
            percentage_i += 1
        origin_ppg = test_ppg_data[i]  # 测试集原始ppg
        origin_abp = test_abp_data[i]  # 测试集原始abp
        # 计算该ppg属于哪一类聚类，索引值是index
        min_dis = 99999
        index = 0
        for j in range(len(centers)):
            dis = KmeansPlus.distance(origin_ppg, centers[j])
            if dis < min_dis:
                min_dis = dis
                index = j
        # f1预测
        y_ppg = max(origin_ppg)
        x_ppg = min(origin_ppg)
        y_abp = max(origin_abp)
        x_abp = min(origin_abp)
        predict_abp_f1 = list()
        block = N // 6  # N=5 block=2 k=2,3,4,5
        for k in range(block, N + 1):
            total = 0
            for l in range(k - block, k):
                total += origin_ppg[l]
            NPMA_ppg = total / block
            NPMA_abp = x_abp + (NPMA_ppg - x_ppg) * ((y_abp - x_abp) / (y_ppg - x_ppg))
            predict_abp_f1.append(NPMA_abp)

        # 准备测试数据的幅值和相位
        origin_ppg_fft = fft(origin_ppg)
        origin_ppg_abs = np.abs(origin_ppg_fft)
        origin_ppg_angel = np.angle(origin_ppg_fft)
        tmp_abs = origin_ppg_abs / N * 2
        tmp_abs[0] /= 2
        # f2预测
        f2_common_abs = np.array(f2_abs_common[index])
        f2_common_angel = np.array(f2_angle_common[index])
        f2_predict_abs = np.divide(tmp_abs, f2_common_abs,
                                   out=np.array(centers_abp[index], dtype='float64'),
                                   where=f2_common_abs != 0
                                   )  # 用聚类中心的平均中心动脉压作为预测的补充值
        f2_predict_angel = origin_ppg_angel - f2_common_angel
        predict_abp_f2 = f2_predict_abs[0]
        for k in range(1, 11):
            predict_abp_f2 += f2_predict_abs[k] * np.cos(k * t + f2_predict_angel[k])
        # f3预测
        f3_predict_abs = np.divide(tmp_abs, f3_abs_common,
                                   out=np.array(all_centers_abp, dtype='float64'),
                                   where=f3_abs_common != 0
                                   )
        f3_predict_angel = origin_ppg_angel - f3_angel_common
        predict_abp_f3 = f3_predict_abs[0]
        for k in range(1, len(f3_predict_abs)):
            predict_abp_f3 += f3_predict_abs[k] * np.cos(k * t + f3_predict_angel[k])
        # f4预测，用f3的就可以，最后只取前10次谐波
        # f4_predict_abs = np.divide(tmp_abs, f3_abs_common,
        #                            # out=np.array(all_centers_abp, dtype='float64'),
        #                            # where=f3_abs_common != 0
        #                            )
        # f4_predict_angel = origin_ppg_angel - f3_angel_common
        # predict_abp_f4 = f4_predict_abs[0]
        # for k in range(1, 11):
        #     predict_abp_f4 += f4_predict_abs[k] * np.cos(k * t + f4_predict_angel[k])
        predict_abp_f4 = np.divide(origin_ppg, f4_transfer_function[index])
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
        origin_PP_value = origin_SBP_value - origin_DBP_value
        if origin_PP_value == 0:
            continue
        o_DBP_array.append(origin_DBP_value)
        o_SBP_array.append(origin_SBP_value)
        o_PP_array.append(origin_PP_value)

        f1_DBP_value = min(predict_abp_f1)  # f1预测的DBP
        f1_SBP_value = max(predict_abp_f1)
        PP = f1_SBP_value - f1_DBP_value
        DBP_AE = abs(f1_DBP_value - origin_DBP_value)
        DBP_RE = abs(f1_DBP_value - origin_DBP_value) / origin_DBP_value
        SBP_AE = abs(f1_SBP_value - origin_SBP_value)
        SBP_RE = abs(f1_SBP_value - origin_SBP_value) / origin_SBP_value
        PP_AE = abs(PP - origin_PP_value)
        PP_RE = abs(PP - origin_PP_value) / origin_PP_value
        f1_DBP_AE_array.append(DBP_AE)
        f1_DBP_RE_array.append(DBP_RE)
        f1_SBP_AE_array.append(SBP_AE)
        f1_SBP_RE_array.append(SBP_RE)
        f1_PP_AE_array.append(PP_AE)
        f1_PP_RE_array.append(PP_RE)
        f1_DBP_array.append(f1_DBP_value)
        f1_SBP_array.append(f1_SBP_value)
        f1_PP_array.append(PP)

        f2_DBP_value = predict_abp_f2[0]  # f2预测的DBP
        # f2_DBP_value = min(predict_abp_f2)
        # peaks, properties = signal.find_peaks(predict_abp_f2)
        # f2_SBP_value = max([predict_abp_f2[index] for index in peaks])  # f2预测的SBP
        f2_SBP_value = max(predict_abp_f2)
        PP = f2_SBP_value - f2_DBP_value
        DBP_AE = abs(f2_DBP_value - origin_DBP_value)
        DBP_RE = abs(f2_DBP_value - origin_DBP_value) / origin_DBP_value
        SBP_AE = abs(f2_SBP_value - origin_SBP_value)
        SBP_RE = abs(f2_SBP_value - origin_SBP_value) / origin_SBP_value
        PP_AE = abs(PP - origin_PP_value)
        PP_RE = abs(PP - origin_PP_value) / origin_PP_value
        f2_DBP_AE_array.append(DBP_AE)
        f2_DBP_RE_array.append(DBP_RE)
        f2_SBP_AE_array.append(SBP_AE)
        f2_SBP_RE_array.append(SBP_RE)
        f2_PP_AE_array.append(PP_AE)
        f2_PP_RE_array.append(PP_RE)
        f2_DBP_array.append(f2_DBP_value)
        f2_SBP_array.append(f2_SBP_value)
        f2_PP_array.append(PP)

        f3_DBP_value = predict_abp_f3[0]  # f3预测的DBP
        # f3_DBP_value = min(predict_abp_f3)
        # peaks, properties = signal.find_peaks(predict_abp_f3)
        # f3_SBP_value = max([predict_abp_f3[index] for index in peaks])  # f3预测的SBP
        f3_SBP_value = max(predict_abp_f3)
        PP = f3_SBP_value - f3_DBP_value
        DBP_AE = abs(f3_DBP_value - origin_DBP_value)
        DBP_RE = abs(f3_DBP_value - origin_DBP_value) / origin_DBP_value
        SBP_AE = abs(f3_SBP_value - origin_SBP_value)
        SBP_RE = abs(f3_SBP_value - origin_SBP_value) / origin_SBP_value
        PP_AE = abs(PP - origin_PP_value)
        PP_RE = abs(PP - origin_PP_value) / origin_PP_value
        f3_DBP_AE_array.append(DBP_AE)
        f3_DBP_RE_array.append(DBP_RE)
        f3_SBP_AE_array.append(SBP_AE)
        f3_SBP_RE_array.append(SBP_RE)
        f3_PP_AE_array.append(PP_AE)
        f3_PP_RE_array.append(PP_RE)
        f3_DBP_array.append(f3_DBP_value)
        f3_SBP_array.append(f3_SBP_value)
        f3_PP_array.append(PP)

        f4_DBP_value = predict_abp_f4[0]  # f4预测的DBP
        # f4_DBP_value = min(predict_abp_f4)
        f4_SBP_value = max(predict_abp_f4)  # f4预测的SBP
        PP = f4_SBP_value - f4_DBP_value
        DBP_AE = abs(f4_DBP_value - origin_DBP_value)
        DBP_RE = abs(f4_DBP_value - origin_DBP_value) / origin_DBP_value
        SBP_AE = abs(f4_SBP_value - origin_SBP_value)
        SBP_RE = abs(f4_SBP_value - origin_SBP_value) / origin_SBP_value
        PP_AE = abs(PP - origin_PP_value)
        PP_RE = abs(PP - origin_PP_value) / origin_PP_value
        f4_DBP_AE_array.append(DBP_AE)
        f4_DBP_RE_array.append(DBP_RE)
        f4_SBP_AE_array.append(SBP_AE)
        f4_SBP_RE_array.append(SBP_RE)
        f4_PP_AE_array.append(PP_AE)
        f4_PP_RE_array.append(PP_RE)
        f4_DBP_array.append(f4_DBP_value)
        f4_SBP_array.append(f4_SBP_value)
        f4_PP_array.append(PP)
    end_time = time.time()
    print("预测时间：" + str(end_time - start_time))

    # 计算平均值、方差等
    f1_DBP_AE_mean = np.mean(f1_DBP_AE_array)
    f1_DBP_AE_std = np.std(f1_DBP_AE_array)
    f1_DBP_RE_mean = np.mean(f1_DBP_RE_array)
    f1_DBP_RE_std = np.std(f1_DBP_RE_array)
    f1_SBP_AE_mean = np.mean(f1_SBP_AE_array)
    f1_SBP_AE_std = np.std(f1_SBP_AE_array)
    f1_SBP_RE_mean = np.mean(f1_SBP_RE_array)
    f1_SBP_RE_std = np.std(f1_SBP_RE_array)
    f1_PP_AE_mean = np.mean(f1_PP_AE_array)
    f1_PP_AE_std = np.std(f1_PP_AE_array)
    f1_PP_RE_mean = np.mean(f1_PP_RE_array)
    f1_PP_RE_std = np.std(f1_PP_RE_array)

    f2_DBP_AE_mean = np.mean(f2_DBP_AE_array)
    f2_DBP_AE_std = np.std(f2_DBP_AE_array)
    f2_DBP_RE_mean = np.mean(f2_DBP_RE_array)
    f2_DBP_RE_std = np.std(f2_DBP_RE_array)
    f2_SBP_AE_mean = np.mean(f2_SBP_AE_array)
    f2_SBP_AE_std = np.std(f2_SBP_AE_array)
    f2_SBP_RE_mean = np.mean(f2_SBP_RE_array)
    f2_SBP_RE_std = np.std(f2_SBP_RE_array)
    f2_PP_AE_mean = np.mean(f2_PP_AE_array)
    f2_PP_AE_std = np.std(f2_PP_AE_array)
    f2_PP_RE_mean = np.mean(f2_PP_RE_array)
    f2_PP_RE_std = np.std(f2_PP_RE_array)

    f3_DBP_AE_mean = np.mean(f3_DBP_AE_array)
    f3_DBP_AE_std = np.std(f3_DBP_AE_array)
    f3_DBP_RE_mean = np.mean(f3_DBP_RE_array)
    f3_DBP_RE_std = np.std(f3_DBP_RE_array)
    f3_SBP_AE_mean = np.mean(f3_SBP_AE_array)
    f3_SBP_AE_std = np.std(f3_SBP_AE_array)
    f3_SBP_RE_mean = np.mean(f3_SBP_RE_array)
    f3_SBP_RE_std = np.std(f3_SBP_RE_array)
    f3_PP_AE_mean = np.mean(f3_PP_AE_array)
    f3_PP_AE_std = np.std(f3_PP_AE_array)
    f3_PP_RE_mean = np.mean(f3_PP_RE_array)
    f3_PP_RE_std = np.std(f3_PP_RE_array)

    f4_DBP_AE_mean = np.mean(f4_DBP_AE_array)
    f4_DBP_AE_std = np.std(f4_DBP_AE_array)
    f4_DBP_RE_mean = np.mean(f4_DBP_RE_array)
    f4_DBP_RE_std = np.std(f4_DBP_RE_array)
    f4_SBP_AE_mean = np.mean(f4_SBP_AE_array)
    f4_SBP_AE_std = np.std(f4_SBP_AE_array)
    f4_SBP_RE_mean = np.mean(f4_SBP_RE_array)
    f4_SBP_RE_std = np.std(f4_SBP_RE_array)
    f4_PP_AE_mean = np.mean(f4_PP_AE_array)
    f4_PP_AE_std = np.std(f4_PP_AE_array)
    f4_PP_RE_mean = np.mean(f4_PP_RE_array)
    f4_PP_RE_std = np.std(f4_PP_RE_array)
    print("           N点平移      本文方法      范琳琳        吴樟洋")
    print("DBP AE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (
        f1_DBP_AE_mean, f1_DBP_AE_std, f2_DBP_AE_mean, f2_DBP_AE_std, f3_DBP_AE_mean, f3_DBP_AE_std, f4_DBP_AE_mean,
        f4_DBP_AE_std))
    print("DBP RE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f\n" % (
        f1_DBP_RE_mean, f1_DBP_RE_std, f2_DBP_RE_mean, f2_DBP_RE_std, f3_DBP_RE_mean, f3_DBP_RE_std, f4_DBP_RE_mean,
        f4_DBP_RE_std))
    print("SBP AE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (
        f1_SBP_AE_mean, f1_SBP_AE_std, f2_SBP_AE_mean, f2_SBP_AE_std, f3_SBP_AE_mean, f3_SBP_AE_std, f4_SBP_AE_mean,
        f4_SBP_AE_std))
    print("SBP RE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f\n" % (
        f1_SBP_RE_mean, f1_SBP_RE_std, f2_SBP_RE_mean, f2_SBP_RE_std, f3_SBP_RE_mean, f3_SBP_RE_std, f4_SBP_RE_mean,
        f4_SBP_RE_std))
    print("PP AE:    %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (
        f1_PP_AE_mean, f1_PP_AE_std, f2_PP_AE_mean, f2_PP_AE_std, f3_PP_AE_mean, f3_PP_AE_std, f4_PP_AE_mean,
        f4_PP_AE_std))
    print("PP RE:    %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (
        f1_PP_RE_mean, f1_PP_RE_std, f2_PP_RE_mean, f2_PP_RE_std, f3_PP_RE_mean, f3_PP_RE_std, f4_PP_RE_mean,
        f4_PP_RE_std))

    pearson_result = pearsonr(o_DBP_array, f2_DBP_array)
    print("DBP pearson:" + str(pearson_result))
    pearson_result = pearsonr(o_SBP_array, f2_SBP_array)
    print("SBP pearson:" + str(pearson_result))
    pearson_result = pearsonr(o_PP_array, f2_PP_array)
    print("PP pearson:" + str(pearson_result))

    t_result = ttest_rel(o_DBP_array, f2_DBP_array)
    print("DBP t:" + str(t_result))
    t_result = ttest_rel(o_SBP_array, f2_SBP_array)
    print("SBP t:" + str(t_result))
    t_result = ttest_rel(o_PP_array, f2_PP_array)
    print("PP t:" + str(t_result))

    np_o_DBP_array = np.array(o_DBP_array)
    np_o_SBP_array = np.array(o_SBP_array)
    np_o_PP_array = np.array(o_PP_array)
    np_f2_DBP_array = np.array(f2_DBP_array)
    np_f2_SBP_array = np.array(f2_SBP_array)
    np_f2_PP_array = np.array(f2_PP_array)
    np_f2_DBP_diff_array = np_f2_DBP_array - np_o_DBP_array
    np_f2_SBP_diff_array = np_f2_SBP_array - np_o_SBP_array
    np_f2_PP_diff_array = np_f2_PP_array - np_o_PP_array
    print("DBP est:" + str(np_f2_DBP_array.mean()) + "+-" + str(np_f2_DBP_array.std()) + "   mea:" + str(
        np_o_DBP_array.mean()) + "+-" + str(np_o_DBP_array.std()) + "    " + str(
        np_f2_DBP_diff_array.mean()) + "+-" + str(np_f2_DBP_diff_array.std()))
    print("SBP est:" + str(np_f2_SBP_array.mean()) + "+-" + str(np_f2_SBP_array.std()) + "   mea:" + str(
        np_o_SBP_array.mean()) + "+-" + str(np_o_SBP_array.std()) + "    " + str(
        np_f2_SBP_diff_array.mean()) + "+-" + str(np_f2_SBP_diff_array.std()))
    print("PP est:" + str(np_f2_PP_array.mean()) + "+-" + str(np_f2_PP_array.std()) + "   mea:" + str(
        np_o_PP_array.mean()) + "+-" + str(np_o_PP_array.std()) + "    " + str(np_f2_PP_diff_array.mean()) + "+-" + str(
        np_f2_PP_diff_array.std()))

    # Plt.prepare()
    # Plt.figure(1)
    # Plt.plotScatter(o_DBP_array, f2_DBP_array, color='black', xstr="DBP measured value(mmHg)",
    #                 ystr="DBP estimated value(mmHg)", text="r=0.963,P<0.001")
    # Plt.plotBox([o_DBP_array, f1_DBP_array, f3_DBP_array, f4_DBP_array, f2_DBP_array], showmeans=True,
    #             labels=["Origin", "Shih YT", "Fan", "Wu", "This paper"],
    #             title="DBP推测值箱型图", ystr="DBP value(mmHg)",
    #             showfliers=False)
    # Plt.bland_altman_plot(o_DBP_array, f2_DBP_array, xstr="Mean DBP(mmHg)", ystr="Difference DBP(mmHg)")
    # Plt.figure(2)
    # Plt.plotScatter(o_SBP_array, f2_SBP_array, color='black', xstr="SBP measured value(mmHg)",
    #                 ystr="SBP estimated value(mmHg)", text="r=0.992,P<0.001")
    # Plt.plotBox([o_SBP_array, f1_SBP_array, f3_SBP_array, f4_SBP_array, f2_SBP_array], showmeans=True,
    #             labels=["Origin", "Shih YT", "Fan", "Wu", "This paper"],
    #             title="SBP推测值箱型图", ystr="SBP value(mmHg)",
    #             showfliers=False)
    # Plt.bland_altman_plot(o_SBP_array, f2_SBP_array, xstr="Mean SBP(mmHg)", ystr="Difference SBP(mmHg)")
    # Plt.figure(3)
    # Plt.plotScatter(o_PP_array, f2_PP_array, color='black', xstr="PP measured value(mmHg)",
    #                 ystr="PP estimated value(mmHg)", text="r=0.958,P<0.001")
    # Plt.plotBox([o_PP_array, f1_PP_array, f3_PP_array, f4_PP_array, f2_PP_array], showmeans=True,
    #             labels=["Origin", "Shih YT", "Fan", "Wu", "This paper"],
    #             title="PP推测值箱型图", ystr="PP value(mmHg)",
    #             showfliers=False)
    # Plt.bland_altman_plot(o_PP_array, f2_PP_array, xstr="Mean PP(mmHg)", ystr="Difference PP(mmHg)")
    # Plt.show()
