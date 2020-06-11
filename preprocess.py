# TODO:
# 1. load data - 通用（训练、预测），相关数据结构
# 2. 数据集拆分 - train_test_split, shuffle
# 3. 数据清洗 - 简单部分，复杂的放到extract_feature.py

def foo():
    print(233)


import numpy as np
import pandas as pd

import random
import math
import datetime, time
import pickle
import os

import matplotlib.pyplot as plt
import seaborn as sns


# contants (TODO: define enum)
from enum import Enum
from typing import Union, Optional

CHUNK_SIZE = 1024 * 1024

TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.000Z'
RECORD_TIME_FORMAT = '%Y%m%d  %H:%M:%S'

vessel_status = {
    "under way using engine": 0,
    "under way sailing": 1,
    "at anchor": 2,
    "not under command(失控)": 3,
    "moored": 4,
    "contrained by her draft": 5,
}

vessel_datasource = {
    "Coastal AIS": 0,
    "Satellite": 1
}

event_code = {'nan', 'TRANSIT PORT ETD', 'UPDATE SHIPMENT ETA', 'TRANSIT PORT ETA', 'SHIPMENT MIT INBOUND DATE', 'ESTIMATED ARRIVAL TIME TO PORT', 'TRANSIT PORT DECLARATION BEGIN', 'ESTIMATED ARRIVAL TO PORT', 'DISCHARGED', 'ESTIMATED ONBOARD DATE', 'PLANNED PICK UP DATE', 'SHIPMENT ONBOARD DATE', 'RDC ATD', 'CONTAINER LOADED ON BOARD', 'DAILY TRACK AND TRACE', 'TRANSIT PORT ATA', 'ARRIVAL AT CFS OR CY', 'CARGO ARRIVAL AT DESTINATION', 'IMP CUSTOMS CLEARANCE START', 'IMP CUSTOMS CLEARANCE FINISHED', 'TRANSIT PORT CUSTOMS RELEASE', 'ARRIVAL AT PORT', 'TRANSIT PORT ATD', 'PICKED UP'}



# support functions

def parse_trace(trace_string):
    trace = trace_string.split('-')

def editDistance(word1, word2) : #used to find the most similar string
    m, n = len(word1), len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1) :
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(m):
        for j in range(n):
            dp[i][j] = min(dp[i][j]+(0 if word1[i] == word2[j] else 1),
                        dp[i][j + 1]+1,
                        dp[i + 1][j]+1,
                        )
    return dp[m][n]


def getDistance(lo1, la1, lo2, la2) : #calculate distance by coordinates
    lng1, lat1, lng2, lat2 = map(math. radians, [float(lo1), float(la1), float(lo2), float(la2)])
    return 6371 * math.acos(math.sin(lng1) * math.sin(lng2) + math.cos(lng1) * math.cos(lng2) * math.cos(lat1 - lat2))


from functools import wraps

def func_timer(function):
    """
    计时装饰器
    参考：https://www.cnblogs.com/jiayongji/p/7588133.html
    """
    @wraps(function)
    def function_timer(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        end_time = time.time()
        print('function `{name}` took {time:.2f} seconds'.format(function.__name__, end_time - start_time))
        return result
    return function_timer


def encode(encodee: Union[dict, set], strategy='one-hot') -> dict:
    """
    encode various states with selected strategy
    """
    if type(encodee) == dict:
        encodee = list(encodee.keys())
    elif type(encodee) == set:
        encodee == list(encodee)
    if strategy == 'one-hot':
        e = np.eye(len(encodee))
        return {key: e[i] for i, key in enumerate(encodee)}
    elif strategy == 'uniform':
        e = np.random.uniform(size=len(encodee))
        return {key: e[i] for i, key in enumerate(encodee)}
    else:
        raise NotImplementedError
    return {}


def main():
    pass


import tensorflow as tf

class DataProcessor:
    """
    data processor for the project
    """
    def __init__(self, output_dir='data/preprocessed'):
        self.cache_dir = output_dir

    def load_train_data(self, port_path, gps_path, event_path, rewrite=False):
        # load port
        port_df = pd.read_csv(port_path)
        port_loc = {} # portName -> location(coordinates)
        for _, r in port_df.iterrows() :
            port_loc.update({r['TRANS_NODE_NAME'] : [r['LATITUDE'], r['LONGITUDE']]})
        # load gps records
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)
        # get ETAs (about 1 min / chunk)
        order_to_port, order_to_eta, port_standardizing = self._generate_etas(port_loc, 
                                                                            gps_record_reader, 
                                                                            cached=True, 
                                                                            test_mode=True)
        # generate training data
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)  # reset
        count = 0
        for chunk in gps_record_reader:
            feature_df, target_df = self._generate_feature_and_target(
                chunk, 
                port_to_loc=port_loc,
                order_to_port=order_to_port,
                order_to_eta=order_to_eta
            )
            if rewrite: # for debugging
                feature_df.to_csv(os.path.join(self.cache_dir, 'features.csv'), mode='w', header=False)
                target_df.to_csv(os.path.join(self.cache_dir, 'target.csv'), mode='w', header=False)
                rewrite = False
            else:
                feature_df.to_csv(os.path.join(self.cache_dir, 'features.csv'), mode='a', header=False)
                target_df.to_csv(os.path.join(self.cache_dir, 'target.csv'), mode='a', header=False)
            count += 1
            if count > 1:   # for debugging
                break
        
        return pd.read_csv(os.path.join(self.cache_dir, 'features.csv'), header=None, low_memory=False, chunksize=CHUNK_SIZE), \
            pd.read_csv(os.path.join(self.cache_dir, 'target.csv'), header=None, low_memory=False, chunksize=CHUNK_SIZE)

    def _generate_etas(self, port_loc, gps_record_reader, cached=True, test_mode=False):

        if cached:
            port_mapping = pickle.load(open(os.path.join(self.cache_dir, 'port_mapping'), 'rb'))
            eta_mapping = pickle.load(open(os.path.join(self.cache_dir, 'eta_mapping'), 'rb'))
            port_standardizing = pickle.load(open(os.path.join(self.cache_dir, 'port_standardizing'), 'rb'))
            return port_mapping, eta_mapping, port_standardizing

        port_mapping = {} # order -> [current nextPort, distance, timestamp, speed]
        eta_mapping = {} # [order+nextPort] -> timestamp
        port_standardizing = {} # informal port name -> most similar formal port name (NOT PRECISE)
        
        count = 0
        previous_time = time.time()
        for chunk in gps_record_reader:
            for (order, timestamp, longitude, latitude, speed, nextport) in zip(chunk[0], chunk[2], chunk[3], chunk[4], chunk[6], chunk[8]) :
                cur_dis = 0

                if order in port_mapping : #if arriving at some port
                    cur_dis = getDistance(longitude, latitude, port_loc[port_mapping[order][0]][0], port_loc[port_mapping[order][0]][1])
                    if cur_dis < port_mapping[order][1] : #getting closer: not arrived, update stats
                        port_mapping[order][1] = cur_dis
                        port_mapping[order][2] = timestamp
                        port_mapping[order][3] = speed
                    else :
                        str = order + port_mapping[order][0]
                        if str in eta_mapping :
                            if speed < port_mapping[order][3] : #slowing, preparing for stop
                                eta_mapping[order] = timestamp
                        else :
                            eta_mapping.update({str: port_mapping[order][2]})
                            #print(cur_dis)

                if order in port_mapping : #
                    if pd.isna(nextport) == False and nextport not in port_loc : #standardize informal port name
                        if nextport in port_standardizing :
                            nextport = port_standardizing[nextport][0]
                        else :
                            port_standardizing.update({nextport : [list(port_loc.keys())[0], editDistance(nextport, list(port_loc.keys())[0])]})
                            for i in port_loc :
                                eDis = editDistance(nextport, i)
                                if eDis < port_standardizing[nextport][1] :
                                    port_standardizing[nextport] = [i, eDis]
                            nextport = port_standardizing[nextport][0]

                    if pd.isna(nextport) == False and port_mapping[order][0] != nextport : #update nextport
                        port_mapping[order][0] = nextport
                    port_mapping[order][1] = cur_dis
                    port_mapping[order][2] = timestamp
                    port_mapping[order][3] = speed
                else :
                    if pd.isna(nextport) == False :
                        if nextport not in port_loc : #standardize informal port name
                            if nextport in port_standardizing :
                                nextport = port_standardizing[nextport][0]
                            else :
                                port_standardizing.update({nextport: [list(port_loc.keys())[0], editDistance(nextport, list(port_loc.keys())[0])]})
                                for i in port_loc :
                                    eDis = editDistance(nextport, i)
                                    if eDis < port_standardizing[nextport][1] :
                                        port_standardizing[nextport] = [i, eDis]
                                nextport = port_standardizing[nextport][0]

                        cur_dis = getDistance(longitude, latitude, port_loc[nextport][0], port_loc[nextport][1])
                        port_mapping.update({order: [nextport, cur_dis, timestamp, speed]})

                count += 1
            if test_mode and count > CHUNK_SIZE * 2 :
                break
            current_time = time.time()
            print("chunk took {} seconds".format(current_time - previous_time))
            previous_time = current_time

        pickle.dump(port_mapping, open(os.path.join(self.cache_dir, 'port_mapping'), 'wb'))
        pickle.dump(eta_mapping, open(os.path.join(self.cache_dir, 'eta_mapping'), 'wb'))
        pickle.dump(port_standardizing, open(os.path.join(self.cache_dir, 'port_standardizing'), 'wb'))

        return port_mapping, eta_mapping, port_standardizing

    
    def load_test_data(self, test_data_path):
        # load
        # get paths
        # vectorize
        # return
        pass
    
    def form_results(self, test_data, pred_deltas):
        pass

    @func_timer
    def _generate_feature_and_target(
            self,
            chunk: pd.DataFrame, 
            port_to_loc: dict, 
            order_to_port: dict, 
            order_to_eta: dict
        ):
        """
        the function to generate the training data
        :param chunk: gps data chunk
        """
        # x (depend on the feature engineering strategy); currently <one_GPS_instance, end_pos>
        # 1. need encoding of many things like carrier, vesselStatus, dataSource...
        # 2. need start pos (in get_etas)
        # 3. basic preprocessing
        # TODO: better feature engineering as mentioned above
        features = pd.DataFrame(
            [
                chunk[3], chunk[4], chunk[6], chunk[7], chunk[0].apply(lambda x: port_to_loc[order_to_port[x][0]])
            ]
        )

        # y = eta - timestamp
        delta_time = pd.to_datetime(chunk[0].apply(lambda x: order_to_eta[x + order_to_port[x][0]])) - pd.to_datetime(chunk[2])
        return features, delta_time

    def _generate_feature(
            self,
            chunk: pd.DataFrame, 
            port_to_loc: dict, 
            order_to_port: dict
        ):
        """
        the primary function for feature engineering
        used for both training data and testing data
        """
        pass


    def _get_paths(self, df):
        # path by order
        pass

    def _get_query(self, df):
        # extract query from test data
        pass
    

if __name__ == '__main__':
    main()