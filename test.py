from myo_tools.save_data import SaveData
from myo_tools.cut_data import CutData
from myo_tools.machine_learning import MachineLearning
from myo_tools.artificial_neural_network import ArtificialNeuralNetwork
from myo_tools.deep_learning import DeepLearning
from myo_tools.TCN import TCN
from myo_tools.online_predict import Listener
import scipy.io as sio
import numpy as np
import myo

# gesture_list = ['握拳', '张手', '一', '二', '三', '四', '向上', '向下', '向左', '向右']
# file_name_list = ['1']
# feature_list=['original']
# feature_list = ['mav', 'rms','std','wl']
# classifier_list = ['lda', 'knn']
# sd = SaveData(gesture_list, 600)
# sd.run()
# cd=CutData(file_name_list,feature_list,200,20)
# cd.run()
# ml=MachineLearning(file_name_list,feature_list,classifier_list,0.9,10)
# ml.run()
# ann=ArtificialNeuralNetwork(file_name_list,feature_list,0.9,32,100)
# ann.run()
# dl=DeepLearning(file_name_list,feature_list,0.9,64,100)
# dl.run()
# tcn=TCN(file_name_list,feature_list,0.9,5,20)
# tcn.run()
# myo.init()
# hub = myo.Hub()
# L = Listener('tcn',gesture_list,feature_list,file_name_list)
# while hub.run(L.on_event, 5):
#     pass
