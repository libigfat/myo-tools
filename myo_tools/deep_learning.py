import scipy.io as sio
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Add, SpatialDropout1D, \
    BatchNormalization, Conv1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


class DeepLearning():
    def __init__(self, file_name_list, feature_list, tvt, batch_size, num_epochs):
        self.file_name_list = file_name_list
        self.feature_list = feature_list
        self.tvt = tvt
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.picture_height = 40
        self.picture_width = 40

    def model(self, input_size, output_size):
        inputs = Input(shape=(input_size))
        x = Conv2D(10, [3, 3], activation='relu')(inputs)
        x = MaxPooling2D([2, 2], [2, 2])(x)
        x = Conv2D(20, [3, 3], activation='relu')(x)
        x = MaxPooling2D([2, 2], [2, 2])(x)
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        outputs = Dense(output_size,activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def convert_to_one_hot(self, Y):
        C = np.max(Y) + 1  # Y:原始手势标签，C：手势个数，eye：单位对角矩阵
        Y = np.eye(C)[Y]  # 将手势标签转换成热码形式，按下标索引取手势标签对应的热码
        return Y

    def load_data(self):
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
        lo_data = lo_data.reshape(data_num, self.picture_height, self.picture_width, 1)
        lo_label = self.convert_to_one_hot(lo_label.reshape(data_num))
        print(lo_data.shape)
        print(lo_label.shape)
        return lo_data, lo_label

    def cnn_data(self, data, label):
        checkpoint_filepath = 'model/cnn/best'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True)
        input_size = data.shape[1:]
        output_size = label.shape[1]
        model = self.model(input_size, output_size)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(data, label, batch_size=self.batch_size, epochs=self.num_epochs, validation_split=1 - self.tvt,
                  callbacks=[model_checkpoint])

    def run(self):
        lo_data, lo_label = self.load_data()
        self.cnn_data(lo_data, lo_label)
