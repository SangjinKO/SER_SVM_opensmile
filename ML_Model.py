import pickle
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from Common_Model import Common_Model


class MLModel(Common_Model):

    def __init__(self, **params):
        super(MLModel, self).__init__(**params)


    def save_model(self, model_name):
        save_path = 'Models/' + model_name + '.m'
        pickle.dump(self.model, open(save_path, "wb"))


    def train(self, x_train, y_train, x_val = None, y_val = None):
        self.model.fit(x_train, y_train)
        self.trained = True


    def predict(self, samples):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return self.model.predict(samples)

class SVM_Model(MLModel):
    def __init__(self, **params):
        params['name'] = 'SVM'
        super(SVM_Model, self).__init__(**params)
        self.model = SVC(kernel = 'rbf', probability = True, gamma = 'auto')

class MLP_Model(MLModel):
    def __init__(self, **params):
        params['name'] = 'Neural Network'
        super(MLP_Model, self).__init__(**params)
        self.model = MLPClassifier(alpha = 1.9, max_iter = 700)
