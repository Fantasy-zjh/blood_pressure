import os
import matplotlib.pyplot as plt
from scipy import signal


class SphygmoCorHelper:
    SAMPLE_RATE = 125
    SPHYGMOCOR_FILE_PATH = "E:\\毕业论文\\blood_data\\hospital\\"
    SPHYGMOCOR_500_PATH = SPHYGMOCOR_FILE_PATH + "extract\\500\\"
    SPHYGMOCOR_200_PATH = SPHYGMOCOR_FILE_PATH + "extract\\200\\"
    SPHYGMOCOR_100_PATH = SPHYGMOCOR_FILE_PATH + "extract\\100\\"
    JAVA_100_PATH = SPHYGMOCOR_FILE_PATH + "extract\\java_100\\"
    JAVA_200_PATH = SPHYGMOCOR_FILE_PATH + "extract\\java_200\\"
    JAVA_300_PATH = SPHYGMOCOR_FILE_PATH + "extract\\java_300\\"
    JAVA_500_PATH = SPHYGMOCOR_FILE_PATH + "extract\\java_500\\"
    JAVA_1000_PATH = SPHYGMOCOR_FILE_PATH + "extract\\java_1000\\"
    SPHYGMOCOR_TRAIN_PATH = SPHYGMOCOR_FILE_PATH + "extract\\train\\"
    SPHYGMOCOR_TEST_PATH = SPHYGMOCOR_FILE_PATH + "extract\\test\\"

    def readSphygmoCorData(self):
        bbp = []
        abp = []
        for file in os.listdir(self.SPHYGMOCOR_FILE_PATH):
            path = self.SPHYGMOCOR_FILE_PATH + file
            if os.path.isdir(path):
                continue
            with open(path, 'r') as f:
                nums1 = []
                nums2 = []
                for line in f.readlines():
                    line = line.strip()
                    num1, num2 = line.split(' ')
                    nums1.append(float(num1))
                    nums2.append(float(num2))
                bbp.append(nums1)
                abp.append(nums2)

        return bbp, abp


if __name__ == "__main__":
    # start_time = time.time()
    sphygmoCorHelper = SphygmoCorHelper()
    bbp_data, abp_data = sphygmoCorHelper.readSphygmoCorData()
    # end_time = time.time()
    # print("行：" + str(len(abp_data)))  11808
    # print("列：" + str(len(abp_data[0])))  1000
    # print("读取数据耗时：" + str(end_time - start_time))  45.54142904281616

    for i in range(len(bbp_data)):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(211)
        plt.title('bbp')
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(bbp_data[i], label='bbp')  # t是横坐标，bbp是纵坐标
        plt.legend(loc='upper right', fontsize=6)

        plt.subplot(212)
        plt.title('abp')
        plt.xlabel('t/ms')
        plt.ylabel('P/mmHg')
        plt.plot(abp_data[i], label='abp')
        plt.legend(loc='upper right', fontsize=6)

        plt.show()
