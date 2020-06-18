import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class LgbTrainer:
    """
    ref: https://github.com/microsoft/LightGBM/blob/master/examples/
    """
    def __init__(self, a):
        self.model = lgb
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            # 'learning_rate': 0.05,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
    def train(self, feature_path, target_path, trace):
        x_all = pd.read_csv(feature_path, header=None, low_memory=False)
        y_all = pd.read_csv(target_path, header=None, low_memory=False)
        print('x shape', x_all.shape[0])
        print('y shape', y_all.shape[0])
        assert x_all.shape[0] == y_all.shape[0]
        X_train = x_all.sample(frac=0.8)
        y_train = y_all[y_all.index.isin(X_train.index)].reset_index(drop=True)
        X_test = x_all[~x_all.index.isin(X_train.index)].reset_index(drop=True)
        y_test = y_all[~y_all.index.isin(X_train.index)].reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test)
        print('Starting training...')
        self.gbm = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=50,
            valid_sets=lgb_eval,
            early_stopping_rounds=20)
        print('Saving model...')
        self.gbm.save_model('models/{}_model.txt'.format(trace))


def train_traces(trace_list):
    failed_list = []
    for trace in trace_list:
        lgbt = LgbTrainer(None)
        try:
            lgbt.train(
                feature_path='data/preprocessed/trace_specific_dataset/{}_train_feature.csv'.format(trace),
                target_path='data/preprocessed/trace_specific_dataset/{}_train_target.csv'.format(trace),
                trace=trace
            )
        except:
            failed_list.append(trace)

    return failed_list