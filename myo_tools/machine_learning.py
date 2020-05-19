import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class MachineLearning():
    def __init__(self, file_name_list, feature_list, classifier_list, tvt, ml_num):
        self.file_name_list = file_name_list
        self.feature_list = feature_list
        self.classifier_list = classifier_list
        self.tvt = tvt
        self.ml_num = ml_num

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
        lo_label = lo_label.reshape(data_num)
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

    def ml_data(self, data, label):
        classifier_func = {'lda': LinearDiscriminantAnalysis, 'knn': KNeighborsClassifier, 'svm': SVC}
        for classifier in self.classifier_list:
            score_data = []
            for m_num in range(self.ml_num):
                train_data, train_label, test_data, test_label = self.random_data(data, label)
                cls = classifier_func[classifier]()
                cls.fit(train_data, train_label)
                cls.score(test_data, test_label)
                score = cls.score(test_data, test_label)
                score_data.append(score)
                print('%s分类器第%s次准确率:%s' % (classifier, (m_num + 1), str(score)))
            mean_score = np.mean(np.array(score_data))
            std_score = np.std(np.array(score_data))
            print('%s分类器准确率平均值:%s,标准差:%s' % (classifier, str(mean_score), str(std_score)))

    def run(self):
        lo_data, lo_label = self.load_data()
        self.ml_data(lo_data, lo_label)
