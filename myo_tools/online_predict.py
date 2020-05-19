import pygame
import sys
import time
import scipy.io as sio
import myo
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow as tf


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
        self.pygame_init()
        self.switch_classifier()

    def pygame_init(self):
        pygame.init()
        info = pygame.display.Info()
        self.width = info.current_w
        self.height = info.current_h
        self.title_font_size = int(info.current_h / 5 * 1 / 4 * 3)
        self.text_font_size = int(info.current_w / 3 * 1 / 10 / 4 * 3)
        for j in range(len(self.gesture_list)):
            setattr(self, 'picture%s' % j, pygame.image.load('source/%s.png' % j))
            setattr(self, 'img%s' % j,
                    pygame.transform.scale(getattr(self, 'picture%s' % j),
                                           (int(info.current_w / 3 * 2), int(info.current_h / 5 * 4))))
        self.screen = pygame.display.set_mode(flags=pygame.FULLSCREEN)

    def pygame_refresh(self, output, delay_time):
        for event in pygame.event.get():  # 遍历所有事件
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
        colour = (255, 255, 255)
        title_font = pygame.font.Font('source/msyh.ttc', self.title_font_size)
        text_font = pygame.font.Font('source/msyh.ttc', self.text_font_size)
        title_txt = title_font.render('手部动作识别', True, colour)
        title_w, title_h = title_txt.get_size()
        title_position_x = int((self.width - title_w) / 2)
        title_position_y = int((self.height / 5 * 1 - title_h) / 2)
        gesture_txt = text_font.render('当前识别动作:' + self.gesture_list[output], True, colour)
        delay_time = format(delay_time * 1000, '.3f')
        delay_time_txt = text_font.render('当前识别耗时:' + str(delay_time) + 'ms', True, colour)
        text_w, text_h = delay_time_txt.get_size()
        text_position_x = int(self.width / 3 * 2 + (self.width / 3 * 1 - text_w) / 2)
        text_position_y1 = int(self.height / 5 * 4 + (self.height / 5 * 1 - text_h * 2) / 3)
        text_position_y2 = int(self.height / 5 * 4 + (self.height / 5 * 1 - text_h * 2) / 3 * 2 + text_h)
        self.screen.fill((0, 0, 0))
        self.screen.blit(getattr(self, 'img%s' % output), (0, int(self.height / 5 * 1)))
        self.screen.blit(title_txt, (title_position_x, title_position_y))
        self.screen.blit(gesture_txt, (text_position_x, text_position_y1))
        self.screen.blit(delay_time_txt, (text_position_x, text_position_y2))
        pygame.display.update()

    def switch_classifier(self):
        if self.classifier in ['lda', 'knn', 'svm']:
            self.classifier_func = {'lda': LinearDiscriminantAnalysis, 'knn': KNeighborsClassifier, 'svm': SVC}
            lo_data, lo_label = self.load_ml_data()
            self.cls = self.classifier_func[self.classifier]()
            self.cls.fit(lo_data, lo_label)
        else:
            if self.classifier == 'ann':
                save_path = "./model/ann"
            else:
                save_path = "./model/cnn"
            self.sess = tf.Session()
            ckpt = tf.train.latest_checkpoint(save_path)
            print(ckpt)
            saver = tf.train.import_meta_graph(ckpt + '.meta')
            saver.restore(self.sess, ckpt)
            graph = tf.get_default_graph()
            self.x = graph.get_tensor_by_name('x:0')
            self.y = graph.get_tensor_by_name('predict_result:0')

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
        output = self.sess.run(self.y, feed_dict={self.x: inputs})
        return output

    def run_dl_predict(self, input):
        input = input.reshape(1, 40, 40, 1)
        output = self.sess.run(self.y, feed_dict={self.x: input})
        return output

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        self.emg.append(event.emg)
        if len(self.emg) == 200:
            start_time = time.time()
            input = np.array(self.emg)
            if self.classifier in ['lda', 'knn', 'svm']:
                output = self.run_ml_predict(input)
            elif self.classifier == 'ann':
                output = self.run_ann_predict(input)
            else:
                output = self.run_dl_predict(input)
            end_time = time.time()
            delay_time = end_time - start_time
            self.pygame_refresh(output[0], delay_time)
            self.emg = self.emg[100:]
