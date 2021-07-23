from MIMICData import MIMICHelper
import matplotlib.pyplot as plt
import time
from AnomalyDetector import AnomalyDetector
from scipy import signal
from WaveletDenoising import wavelet_noising
import math
import numpy as np
from pylab import *
import codecs
import matplotlib.pyplot as plt
import math
from scipy import signal
from MIMICData import MIMICHelper
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import os
import json
from SphygmoCorData import SphygmoCorHelper

if __name__ == "__main__":
    a = np.array([100, 2, 2],)
    a = a / 2
    a[0] /= 5
    print(a)
