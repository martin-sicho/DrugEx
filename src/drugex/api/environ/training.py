"""
training

Created by: Martin Sicho
On: 18-11-19, 16:25
"""

from drugex.api.environ.models import EnvironModel


class EnvironTrainer:

    def __init__(self, model : EnvironModel):
        self.model = model
        self.cv = None
        self.test = None

    def train(self):
        self.cv, self.test = self.model.fit()

    def saveModel(self, out):
        self.model.save(out)

    def savePerfData(self, out_cv, out_test):
        self.cv.to_csv(out_cv, index=None)
        self.test.to_csv(out_test, index=None)

