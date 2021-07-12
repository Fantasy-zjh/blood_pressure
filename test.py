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
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt


# 计算两点之间的距离
def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


if __name__ == "__main__":
    a = [[1, 10], [3, 20]]
    a_r = np.array(a)
    print(a_r.mean(axis=0))