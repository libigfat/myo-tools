import time
import scipy.io as sio
import myo
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import load_model


class Listener(myo.DeviceListener):
    def __init__(self, classifier, gesture_list, feature_list=[], file_name_list=[]):
        self.emg = []
        self.threshold = 100
        self.classifier = classifier
        self.gesture_list = gesture_list
        self.feature_list = feature_list
        self.file_name_list = file_name_list
        self.feature_func = {'mav': self.mav, 'rms': self.rms, 'iemg': self.iemg, 'zc': self.zc, 'var': self.var,
                             'scc': self.scc, 'wl': self.wl, 'std': self.std, 'wa': self.wa, 'logd': self.logd,
                             'min': self.min, 'max': self.max, 'ft_mean': self.ft_mean, 'ft_var': self.ft_var,
                             'ft_std': self.ft_std}
        self.switch_classifier()

    def switch_classifier(self):
        if self.classifier in ['lda', 'knn', 'svm']:
            self.classifier_func = {'lda': LinearDiscriminantAnalysis, 'knn': KNeighborsClassifier, 'svm': SVC}
            lo_data, lo_label = self.load_ml_data()
            self.cls = self.classifier_func[self.classifier]()
            self.cls.fit(lo_data, lo_label)
        else:
            if self.classifier == 'ann':
                save_path = "model/ann/best"
            elif self.classifier == 'cnn':
                save_path = "model/cnn/best"
            else:
                save_path = "model/tcn/best"
            self.model = load_model(save_path)
            print(save_path)

    def load_ml_data(self):
        file_path = '+'.join(str(i) for i in self.feature_list)
        lo_data = []
        lo_label = []
        for name in self.file_name_list:
            cut_data = sio.loadmat('./data/cut_data/' + file_path + '/%s.mat' % name)['cut_data']
            repeat_num = cut_data.shape[0]
            gesture_num = cut_data.shape[1]
            win_num = cut_data.shape[2]
            for r_num in range(repeat_num):
                x = []
                y = []
                for g_num in range(gesture_num):
                    for w_num in range(win_num):
                        x.append(cut_data[r_num, g_num, w_num])
                        y.append(g_num)
                x = np.array(x)
                y = np.array(y)
                lo_data.append(x)
                lo_label.append(y)
        lo_data = np.array(lo_data)
        lo_label = np.array(lo_label)
        data_num = lo_data.shape[0] * lo_data.shape[1]
        lo_data = lo_data.reshape(data_num, -1)
        lo_label = lo_label.reshape(data_num)
        print(lo_data.shape)
        print(lo_label.shape)
        return lo_data, lo_label

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
            ft.append(abs(np.fft.fft(data[:, i])[1:int(data.shape[0] / 2) + 1]))
        return np.array(ft)

    def ft_mean(self, data):
        return np.mean(self.ft(data), axis=1)

    def ft_var(self, data):
        return np.var(self.ft(data), axis=1)

    def ft_std(self, data):
        return np.std(self.ft(data), axis=1)

    def run_ml_predict(self, input):
        inputs = []
        for feature in self.feature_list:
            inputs.append(self.feature_func[feature](input))
        inputs = np.array(inputs).reshape(1, -1)
        output = self.cls.predict(inputs)
        return output

    def run_ann_predict(self, input):
        inputs = []
        for feature in self.feature_list:
            inputs.append(self.feature_func[feature](input))
        inputs = np.array(inputs).reshape(1, -1)
        output = self.model.predict(inputs)
        return output

    def run_dl_predict(self, input):
        input = input.reshape(1, 40, 40, 1)
        output = self.model.predict(input)
        return output

    def run_tcn_predict(self, inputss):
        output = self.model.predict(inputss)
        return output

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        self.emg.append(event.emg)
        if self.classifier!='tcn':
            if len(self.emg) == 200:
                input = np.array(self.emg)
                if self.classifier in ['lda', 'knn', 'svm']:
                    output = self.run_ml_predict(input)
                    print(output)
                elif self.classifier == 'ann':
                    output = self.run_ann_predict(input)
                    print(np.argmax(output))
                else:
                    output = self.run_dl_predict(input)
                    print(np.argmax(output))
                self.emg = self.emg[100:]
        else:
            if len(self.emg) == 600:
                inputss=[]
                input = np.array(self.emg)
                for i in range(21):
                    inputs=[]
                    for feature in self.feature_list:
                        inputs.append(self.feature_func[feature](input[i*20:i*20+200]))
                    inputss.append(np.array(inputs).reshape(1, -1))
                inputss=np.array(inputss).reshape((1,21,32))

                output = self.run_tcn_predict(inputss)
                print(np.argmax(output))
                self.emg = self.emg[20:] #滑动大小
