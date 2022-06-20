import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix

class PrecisionReport:
    def __init__(self, y_test, preds):
        self.precision, self.recall, self.thresholds = precision_recall_curve(y_test, preds)
        self.fscore = (2 * self.precision * self.recall) / (self.precision + self.recall)
        self.roc_auc = roc_auc_score(y_test, preds)
        self.ix = np.argmax(self.fscore)
        self.cnf_matrix = confusion_matrix(y_test, preds > self.thresholds[self.ix])
        self.metrics_df = None


    def report(self, model_name=None):

        return {
                    'model': model_name,
                    'thresh': self.thresholds[self.ix],
                    'F-Score': self.fscore[self.ix],
                    'Precision': self.precision[self.ix],
                    'Recall': self.recall[self.ix],
                    'ROC AUC': self.roc_auc
                    }
    def print_report(self):
        print(f'Best Threshold={self.thresholds}, ',
              f'F-Score={self.fscore:.3f}, ',
              f'Precision={self.precision:.3f},',
              f'Recall={self.recall:.3f}'
              )







