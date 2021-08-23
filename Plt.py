import matplotlib.pyplot as plt


class Plt:

    @staticmethod
    def figure(num):
        plt.figure(num, figsize=(8, 8))

    # 散点图
    @staticmethod
    def plotScatter(x_data, y_data, **kwargs):
        color = kwargs.get("color")
        xstr = kwargs.get("xstr")
        ystr = kwargs.get("ystr")
        title = kwargs.get("title")
        label = kwargs.get("label")
        handles = kwargs.get("handles")
        plt.scatter(x_data, y_data, c=color, s=300, alpha=0.3)
        plt.plot(x_data, x_data, c=color)
        if xstr:
            plt.xlabel(xstr, fontsize=30)
        if ystr:
            plt.ylabel(ystr, fontsize=30)
        plt.xticks(size=30)
        plt.yticks(size=30)
        if title:
            plt.title(title, fontsize=30)
        if label and handles:
            plt.legend(handles=handles, labels=label, loc="upper right", fontsize=30)
        elif label:
            plt.legend(labels=label, loc="upper right", fontsize=30)
        plt.tight_layout()

    # 箱型图
    @staticmethod
    def plotBox(data, **kwargs):
        xstr = kwargs.get("xstr")
        ystr = kwargs.get("ystr")
        title = kwargs.get("title")
        showmeans = kwargs.get("showmeans")
        labels = kwargs.get("labels")
        label = kwargs.get("label")
        handles = kwargs.get("handles")
        showfliers = kwargs.get("showfliers")
        plt.boxplot(data, showmeans=showmeans, labels=labels, showfliers=showfliers)
        if xstr:
            plt.xlabel(xstr, fontsize=30)
        if ystr:
            plt.ylabel(ystr, fontsize=30)
        plt.xticks(size=30)
        plt.yticks(size=30)
        if title:
            plt.title(title, fontsize=30)
        if label and handles:
            plt.legend(handles=handles, labels=label, loc="upper right", fontsize=30)
        elif label:
            plt.legend(labels=label, loc="upper right", fontsize=30)
        plt.tight_layout()

    @staticmethod
    def prepare():
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

    @staticmethod
    def show():
        plt.show()
