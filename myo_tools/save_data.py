from threading import Timer
import scipy.io as sio
import myo
import numpy as np
import os


class SaveData(myo.DeviceListener):
    def __init__(self, gesture_list, data_size):
        myo.init()
        self.hub = myo.Hub()
        self.gesture_list = gesture_list
        self.data_size = data_size
        self.n = len(gesture_list)
        self.switch = False
        self.x = []
        self.data = []
        self.gesture_name = ''
        self.mark = 0
        self.pose = 0

    def change_switch(self):
        self.switch = not self.switch

    def change_mark(self):
        if self.mark == 4:
            if self.pose == self.n:
                self.mark = 5
            else:
                self.mark = 0
        else:
            self.mark += 1

    def trainer(self, count):
        while self.hub.run(self.on_event, 5):

            if self.mark == 0:
                self.change_mark()
                self.gesture_name = self.gesture_list[self.pose]
                t = Timer(1, self.change_switch)
                t.start()

            elif self.mark == 1:
                print('\r' + '三秒后训练手部动作(%s)' % self.gesture_name, end='')

            elif self.mark == 2:
                print('\r' + '手部动作(%s)训练中' % self.gesture_name, end='')

            elif self.mark == 3:
                self.change_mark()
                t = Timer(1, self.change_mark)
                t.start()

            elif self.mark == 4:
                print('\r' + '手部动作(%s)训练结束,' % self.gesture_name + '放松五秒', end='')

            elif self.mark == 5:
                self.mark = 0
                self.pose = 0
                print('\r' + '第%s组采集完成,' % count + '继续采集 or 保存数据:', end='')
                break

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        if self.switch:
            self.x.append(event.emg)
            if len(self.x) == 1:
                self.change_mark()
            if len(self.x) == self.data_size:
                self.change_switch()
                self.pose += 1
                self.data.append(self.x)
                self.x = []
                self.change_mark()

    def run(self):
        if not os.path.exists('data/save_data'):
            if not os.path.exists('data'):
                os.mkdir(os.getcwd() + '/data')
            os.mkdir(os.getcwd() + '/data/save_data')
        count = 1
        save_data = []
        file_name = input(('请输入要保存数据的文件名:'))
        ready = input(('准备好,输入yes,开始训练:'))
        if ready == 'yes':
            self.trainer(count)
            save_data.append(self.data)
            self.data = []
            yn = int(input(''))
            if yn == 0:
                while True:
                    count = count + 1
                    self.trainer(count)
                    save_data.append(self.data)
                    self.data = []
                    yn = int(input(''))
                    if yn == 0:
                        pass
                    else:
                        save_data = np.array(save_data)
                        print(save_data.shape)
                        sio.savemat('./data/save_data/%s.mat' % file_name, {'save_data': save_data})
                        break
            else:
                save_data = np.array(save_data)
                print(save_data.shape)
                sio.savemat('./data/save_data/%s.mat' % file_name, {'save_data': save_data})
