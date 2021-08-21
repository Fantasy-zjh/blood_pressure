import matplotlib.pyplot as plt
from MIMICData import MIMICHelper
import numpy as np
from WaveletDenoising import wavelet_noising
from scipy import signal
import random

if __name__ == "__main__":
    mean = 1.23
    std = 2.34
    print("           N点平移      本文方法      范琳琳        吴樟洋")
    print("DBP AE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (mean, std, mean, std, mean, std, mean, std))
    print("DBP RE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f\n" % (mean, std, mean, std, mean, std, mean, std))
    print("SBP AE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (mean, std, mean, std, mean, std, mean, std))
    print("SBP RE:   %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f\n" % (mean, std, mean, std, mean, std, mean, std))
    print("PP RE:    %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (mean, std, mean, std, mean, std, mean, std))
    print("PP RE:    %.2f±%.2f   %.2f±%.2f    %.2f±%.2f    %.2f±%.2f" % (mean, std, mean, std, mean, std, mean, std))
