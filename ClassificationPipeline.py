import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import naive_bayes
import scikitplot as skplt
import matplotlib.pyplot as plt
import pickle


class ClassificationPipeline:
    def __init__(self, controller=None):
        self.feature_controller = controller
        self.model = naive_bayes.GaussianNB()

    def train(self):
        self.feature_controller.train_test_split(0.8)
        self.feature_controller.tune_gradients(self.model)
        self.feature_controller.tune_diffusion(self.model)

        features = self.feature_controller.extract_features()
        self.model.fit(features, self.feature_controller.y_train)

    def test(self, f_test=None, l_test=None):
        features_test = self.feature_controller.extract_features(False) if self.feature_controller is not None else f_test
        labels_test = self.feature_controller.y_test if self.feature_controller is not None else l_test
        predictions = self.model.predict(features_test)
        report = classification_report(labels_test, predictions)
        result = confusion_matrix(labels_test, predictions)

        return report, result

    def save_model(self):
        with open('model.pkl', 'wb') as of:
            pickle.dump(self.model, of)

    def save_test_features(self):
        t_features = self.feature_controller.extract_features(False);
        with open('features.npy', 'wb') as of:
            np.save(of,t_features)
        with open('labels.npy', 'wb') as of:
            np.save(of,self.feature_controller.y_test)

    def load_model(self):
        with open('model.pkl', 'rb') as infl:
            self.model = pickle.load(infl)

    def load_features(self):
        features = np.load('features.npy')
        labels = np.load('labels.npy')

        return features, labels

    def plot_roc_curve(self, f_test=None, l_test=None):
        features_test = self.feature_controller.extract_features(
            False) if self.feature_controller is not None else f_test
        y_probs = self.model.predict_proba(features_test)

        skplt.metrics.plot_roc(l_test,y_probs)
        plt.show()