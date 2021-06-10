from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
from AnomalyDetector import AnomalyDetector
from scipy import signal
from WaveletDenoising import wavelet_noising
import math
import numpy as np


# 计算两点之间的距离
def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


if __name__ == "__main__":
    X = [[1, 2, 3, 4],[1,2,3,4]]
    Y = [0, 1, 2, 3]
    cluster_centers = np.zeros((2, 3))
    print(cluster_centers)
    print(cluster_centers[0,])