import scipy.io as sio
import numpy as np
import tensorflow as tf
import os


class ArtificialNeuralNetwork():
    def __init__(self, file_name_list, feature_list, tvt, batch_size, num_epochs):
        self.file_name_list = file_name_list
        self.feature_list = feature_list
        self.tvt = tvt
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def model(self, input_size, output_size):
        x = tf.placeholder(tf.float32, shape=[None, input_size], name='x')
        y = tf.placeholder(tf.float32, shape=[None, output_size])
        shadow = tf.layers.dense(x, 50, activation=tf.nn.sigmoid)
        output = tf.layers.dense(shadow, output_size)
        return x, y, output

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
        lo_data = lo_data.reshape(data_num, -1)
        lo_label = self.convert_to_one_hot(lo_label.reshape(data_num))
        print(lo_data.shape)
        print(lo_label.shape)
        return lo_data, lo_label

    def random_data(self, data, label):
        permutation = np.random.permutation(len(data))
        random_data = data[permutation]
        random_label = label[permutation]
        tvt_num = int(self.tvt * len(data))
        train_data = random_data[:tvt_num]
        train_label = random_label[:tvt_num]
        test_data = random_data[tvt_num:]
        test_label = random_label[tvt_num:]
        return train_data, train_label, test_data, test_label

    def random_mini_batches(self, X, Y, mini_batch_size):
        m = X.shape[0]
        mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation]
        shuffled_Y = Y[permutation]
        num_complete_minibatches = int(m / mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size: k * (mini_batch_size + 1)]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * (mini_batch_size + 1)]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def ann_data(self, data, label):
        train_data, train_label, test_data, test_label = self.random_data(data, label)
        save_path = './model/ann/'
        input_size = train_data.shape[1]
        output_size = train_label.shape[1]
        x, y, output = self.model(input_size, output_size)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(output, axis=1), tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        predict_result = tf.argmax(output, axis=1, name='predict_result')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            try:
                print('尝试恢复历史训练参数')
                ckpt = tf.train.latest_checkpoint(save_path)
                saver.restore(sess, ckpt)
                print('恢复参数成功')
            except:
                print('未检测到历史训练参数，重新开始训练')
                if not os.path.exists('model'):
                    os.mkdir(os.getcwd() + '/model')
                sess.run(tf.global_variables_initializer())

            for i in range(self.num_epochs):
                mini_batches = self.random_mini_batches(train_data, train_label, self.batch_size)

                for mini_batch in mini_batches:
                    batch_xs, batch_ys = mini_batch
                    _ = sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

                train_loss, train_acc = sess.run([loss, accuracy], feed_dict={x: train_data, y: train_label})
                print('迭代次数:%d---训练准确率:%.3f---训练损失:%.3f' % (i + 1, train_acc * 100, train_loss))
                if (i + 1) % 10 == 0:
                    test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: test_data, y: test_label})
                    print('迭代次数:%d---测试准确率:%.3f---测试损失:%.3f' % (i + 1, test_acc * 100, test_loss))
                    saver.save(sess, save_path + 'ann', global_step=i + 1)
                    print("保存参数")

    def run(self):
        lo_data, lo_label = self.load_data()
        self.ann_data(lo_data, lo_label)
