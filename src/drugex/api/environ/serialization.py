"""
serialization

Created by: Martin Sicho
On: 25-11-19, 14:45
"""
import os
from abc import ABC, abstractmethod
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


class EnvironSerializer(ABC):

    @abstractmethod
    def saveModel(self, model):
         pass


class EnvironDeserializer(ABC):

    @abstractmethod
    def getModel(self):
         pass


class FileEnvSerializer(EnvironSerializer):

    def __init__(self, out_dir, identifier="EnvironModel", include_perf=True):
        self.identifier = identifier
        self.include_perf = include_perf
        self.out_dir = out_dir

    def saveModel(self, model):
        joblib.dump(model, os.path.join(self.out_dir, "{0}.pkg".format(self.identifier)))

        if self.include_perf:
            cv, test = model.getPerfData()
            cv.to_csv(os.path.join(self.out_dir, "{0}.cv.txt".format(self.identifier)), index=None)
            # Compute ROC curve and ROC area for each class
            fpr, tpr, _ = roc_curve(cv["LABEL"], cv["SCORE"])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.out_dir, "{0}.cv.png".format(self.identifier)))

            test.to_csv(os.path.join(self.out_dir, "{0}.ind.txt".format(self.identifier)), index=None)


class FileEnvDeserializer(EnvironDeserializer):

    def __init__(self, out_dir, identifier):
        self.path = os.path.join(out_dir, identifier + ".pkg")

    def getModel(self):
        return joblib.load(self.path)