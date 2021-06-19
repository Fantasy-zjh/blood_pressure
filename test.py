from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
from AnomalyDetector import AnomalyDetector
from scipy import signal
from WaveletDenoising import wavelet_noising
import math
import numpy as np
from pylab import *
from numpy import *
import codecs
import matplotlib.pyplot as plt
import math
from scipy import signal
from MIMICData import MIMICHelper


# 计算两点之间的距离
def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


if __name__ == "__main__":
    # data = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
    # data = [signal.resample(d, 5).tolist() for d in data]
    # print(data)
    # print(type(data))
    d = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
    d = array(d)
    out = nonzero(d[:, 0] > 2)[0].tolist()
    print(out)
    print(type(out))
    data = [1, 2, 3, 4]
    mic = MIMICHelper()
    mic.writeToFile2(data, mic.MIMIC_ONE_1000_PATH + "center.cluster")
