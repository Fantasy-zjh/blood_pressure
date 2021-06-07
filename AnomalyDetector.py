import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
from pyculiarity import detect_vec
import pandas as pd
from scipy import signal


class AnomalyDetector:
    NONE_ANOMALY = 0
    INVALID_ANOMALY = 1
    SPIKE_ANOMALY = 2

    def __init__(self, *args):
        if len(args) == 1:
            self.data = args
        self.times = pd.date_range(start='2021-05-25', periods=1000, freq='S')

    def setData(self, d):
        self.data = d

    # 基于Chauvenet准则的异常信号检测
    def detect(self):
        m = len(self.data)
        n = len(self.data[0])
        mean = list()
        std = list()
        for i in range(m):
            mean_data = np.mean(self.data[i])
            std_data = np.std(self.data[i])
            mean.append(mean_data)
            std.append(std_data)
        meanstd = np.mean(std)
        stdstd = np.std(std)
        Tpc = (1 + 0.4 * math.log(m, math.e)) * stdstd
        for i in range(m):
            # 接触不良
            if abs(std[i] - meanstd) > Tpc:
                return self.INVALID_ANOMALY
            Tpn = (1 + 0.4 * math.log(n, math.e)) * std[i]
            for j in range(n):
                if abs(self.data[i][j] - mean[i]) > Tpn:
                    return self.SPIKE_ANOMALY
        return self.NONE_ANOMALY

    # 基于ADF的异常检测
    def ADFdetect(self):
        return adfuller(self.data)

    # 基于S-H-ESD的异常检测
    def SHESDdetect(self):
        s = pd.Series(self.data)
        peaks = signal.find_peaks(self.data, distance=50)[0]
        p = peaks[1] - peaks[0]
        results = detect_vec(s, max_anoms=0.001, direction='both', only_last=False, period=p, alpha=0.05,
                             threshold='med_max')
        return results['anoms']
