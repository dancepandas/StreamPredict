import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import config

mpl.rcParams['font.sans-serif'] = ['SimSun']# 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def plot_data(pre_data,target_data):
    x=range(len(pre_data))
    target_data=target_data[-len(pre_data):]
    plt.plot(x,np.array(pre_data)[:,-1],label='pre')
    plt.plot(x,np.array(target_data),label='obs')
    plt.xlabel('序列索引')
    plt.ylabel('流量')
    plt.title('预测结果')
    plt.legend(loc='best')
    plt.savefig(config.fig_save_file+'/plot_data.png')
    plt.show()

