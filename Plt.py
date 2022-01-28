import matplotlib.pyplot as plt
import numpy as np


class Plt:

    @staticmethod
    def figure(num=None):
        n = 1
        if num is not None:
            n = num
        plt.figure(n, figsize=(9, 9))

    @staticmethod
    def subPlot(integer):
        plt.subplot(integer)

    @staticmethod
    def prepare():
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def doSomethingelse(args, kwargs):
        xstr = kwargs.get("xstr")
        ystr = kwargs.get("ystr")
        title = kwargs.get("title")
        label = kwargs.get("label")
        handles = kwargs.get("handles")
        xyFont = {
            'family': 'Times New Roman',
            'size': 30
        }
        if xstr:
            plt.xlabel(xstr, xyFont)
        if ystr:
            plt.ylabel(ystr, xyFont)
        plt.xticks(size=30)
        plt.yticks(size=30)
        if title:
            plt.title(title, fontsize=30)
        if label and handles:
            plt.legend(handles=handles, labels=label, loc="upper right", fontsize=30)
        elif label:
            plt.legend(labels=label, loc="upper right", fontsize=30)
        plt.tight_layout()

    # altman图
    @staticmethod
    def bland_altman_plot(data1=None, data2=None, *args, **kwargs):
        if data1 is None or data2 is None:
            return
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96 * sd, color='red', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='red', linestyle='--')
        total = len(data1)
        num = 0
        bottom = md - 1.96 * sd
        top = md + 1.96 * sd
        for dif in diff:
            if dif < bottom or dif > top:
                num += 1
        perc = num / total
        print("离异点百分比：{:.5f}  差值均值：{}  上下限：{}---{}".format(perc, md, bottom, top))
        Plt.doSomethingelse(args, kwargs)

    # 散点图
    @staticmethod
    def plotScatter(x_data=None, y_data=None, *args, **kwargs):
        color = kwargs.get("color")
        plt.scatter(x_data, y_data, c=color, s=250, alpha=0.3)
        plt.plot(x_data, x_data, c=color)
        text = kwargs.get("text")
        if text:
            plt.text(x=min(min(x_data), min(y_data)), y=max(max(x_data), max(y_data)), s=text, fontsize=30,
                     fontdict={'family': 'Times New Roman', 'size': 30})
        Plt.doSomethingelse(args, kwargs)

    # 箱型图
    @staticmethod
    def plotBox(data=None, *args, **kwargs):
        showmeans = kwargs.get("showmeans")
        labels = kwargs.get("labels")
        showfliers = kwargs.get("showfliers")
        plt.boxplot(data, showmeans=showmeans, labels=labels, showfliers=showfliers)
        Plt.doSomethingelse(args, kwargs)

    @staticmethod
    def plotLiner(x_data=None, y_data=None, *args, **kwargs):
        plt.plot(x_data, y_data)
        Plt.doSomethingelse(args, kwargs)
