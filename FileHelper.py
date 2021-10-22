import time

class FileHelper:
    @staticmethod
    def writeToFile(data, filename):
        with open(filename, 'w') as f:
            total = len(data)
            print("写到(" + filename + ")的原始数据一共:" + str(total) + "行")
            cur = 1
            for data_list in data:
                size = len(data_list)
                for i in range(size):
                    if i > 0:
                        f.write(" ")
                    f.write(str(data_list[i]))
                f.write("\n")
                cur += 1
                if cur % 200 == 0:
                    print("写了：" + str((cur / total) * 100) + "%数据，时间：" + time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                       time.localtime(time.time())))

    @staticmethod
    def writeToFile2(data, filename):
        with open(filename, 'w') as f:
            total = len(data)
            print("写到(" + filename + ")的原始数据一共:" + str(total) + "行")
            cur = 1
            for i in range(total):
                if i > 0:
                    f.write("\n")
                f.write(str(data[i]))
                cur += 1
                if cur % 200 == 0:
                    print("写了：" + str((cur / total) * 100) + "%数据，时间：" + time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                       time.localtime(time.time())))

    @staticmethod
    def readFromFileFloat(filename):
        ret = list()
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')
                data = [float(d) for d in line]
                ret.append(data)
        return ret

    @staticmethod
    def readFromFileInteger(filename):
        ret = list()
        with open(filename, 'r') as f:
            for line in f.readlines():
                line.strip()
                ret.append(int(line))
        return ret