import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


class LgbPredictor:

    def __init__(self, path):
        self.model = lgb.Booster(model_file=path)
    
    def predict(self, test_path):
        # load test
        X_test = pd.read_csv(test_path, header=None, low_memory=False)
        y_pred = self.model.predict(X_test, num_iteration=7)
        return y_pred


def pred_traces(trace_list):
    preds = []
    for trace in trace_list:
        lgbp = LgbPredictor('models/{}_model.txt'.format(trace))
        y_pred = lgbp.predict('data/preprocessed/trace_specific_dataset/{}_test_feature.csv'.format(trace))
        preds.append(y_pred)
    return preds