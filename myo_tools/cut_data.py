import numpy as np  # 导入numpy模块
import scipy.io as sio  # 导入mat数据
import os


class CutData():
    def __init__(self, file_name_list, feature_list, win_size, step_size):
        self.file_name_list = file_name_list
        self.feature_list = feature_list
        self.win_size = win_size
        self.step_size = step_size
        self.threshold = 100
        self.feature_func = {}

    def multZC(self, data):
        mults = []
        for i in range(1, len(data)):
            mult = np.multiply(data[i], data[i - 1]) < 0 & (abs(np.diff(data, axis=0)[i - 1]) > self.threshold)
            mults.append(mult)
        mults = np.array(mults)
        return mults

    def multSCC(self, data):
        mults = []
        for i in range(1, len(data)):
            mult = (np.multiply(data[i], data[i - 1]) < 0) & (abs(data[i]) > self.threshold) & (
                    abs(data[i - 1]) > self.threshold)
            mults.append(mult)
        mults = np.array(mults)
        return mults

    def mav(self, data):
        return np.mean(abs(data), axis=0)

    def rms(self, data):
        return np.sqrt(np.mean(data ** 2, axis=0))

    def iemg(self, data):
        return np.sum(abs(data), axis=0)

    def zc(self, data):
        return np.sum(self.multZC(data), axis=0)

    def var(self, data):
        return np.var(data, axis=0)

    def scc(self, data):
        return np.sum(self.multSCC(np.diff(data, axis=0)), axis=0)

    def wl(self, data):
        return np.sum(abs(np.diff(data, axis=0)), axis=0) / len(data)

    def std(self, data):
        return np.std(data, axis=0)

    def wa(self, data):
        return np.sum(abs(np.diff(data, axis=0)) > self.threshold, axis=0)

    def logd(self, data):
        return np.exp(np.mean(np.log(abs(data) + 0.1), axis=0))

    def min(self, data):
        return np.min(data, axis=0)

    def max(self, data):
        return np.max(data, axis=0)

    def ft(self, data):
        ft = []
        for i in range(8):
            ft.append(abs(np.fft.fft(data[:, i])[1:int(self.win_size / 2) + 1]))
        return np.array(ft)

    def ft_mean(self, data):
        return np.mean(self.ft(data), axis=1)

    def ft_var(self, data):
        return np.var(self.ft(data), axis=1)

    def ft_std(self, data):
        return np.std(self.ft(data), axis=1)

    def windows(self, data):
        data_size = len(data)  # 输入数据的长度
        win_data = []
        count = 0
        while data_size - count - self.win_size >= 0:  # 当数据还没有滑完，即>=0,就一直循环
            window = data[count:count + self.win_size]
            x = []
            if self.feature_list == ['all']:
                for value in self.feature_func.values():
                    x.append(value(window))
            elif self.feature_list == ['original']:
                x.append(window)
            else:
                for feature in self.feature_list:
                    x.append(self.feature_func[feature](window))
            win_data.append(np.array(x).reshape(-1))
            count += self.step_size
        return np.array(win_data)

    def run(self):
        file_path = '+'.join(str(i) for i in self.feature_list)
        if not os.path.exists('data/cut_data/' + file_path):
            if not os.path.exists('data/cut_data'):
                os.mkdir(os.getcwd() + '/data/cut_data')
            os.mkdir(os.getcwd() + '/data/cut_data/' + file_path)
        self.feature_func = {'mav': self.mav, 'rms': self.rms, 'iemg': self.iemg, 'zc': self.zc, 'var': self.var,
                             'scc': self.scc, 'wl': self.wl, 'std': self.std, 'wa': self.wa, 'logd': self.logd,
                             'min': self.min, 'max': self.max, 'ft_mean': self.ft_mean, 'ft_var': self.ft_var,
                             'ft_std': self.ft_std}
        for name in self.file_name_list:
            print('正在处理%s.mat' % name)
            save_data = sio.loadmat('./data/save_data/%s.mat' % name)['save_data']
            repeat_num = save_data.shape[0]
            gesture_num = save_data.shape[1]
            cut_data = []
            for r_num in range(repeat_num):
                data = []
                for g_num in range(gesture_num):
                    win_data = self.windows(save_data[r_num, g_num])
                    print(win_data.shape)
                    data.append(win_data)
                cut_data.append(data)
            cut_data = np.array(cut_data)
            sio.savemat('./data/cut_data/' + file_path + '/%s.mat' % name, {'cut_data': cut_data})
            print(cut_data.shape)
            print('%s.mat处理完成' % name)
