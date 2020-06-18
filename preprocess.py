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
    "not under command": 3,
    "moored": 4,
    "contrained by her draft": 5,
}

vessel_datasource = {
    "Coastal AIS": 0,
    "Satellite": 1
}

event_code = {'nan', 'TRANSIT PORT ETD', 'UPDATE SHIPMENT ETA', 'TRANSIT PORT ETA', 'SHIPMENT MIT INBOUND DATE', 'ESTIMATED ARRIVAL TIME TO PORT', 'TRANSIT PORT DECLARATION BEGIN', 'ESTIMATED ARRIVAL TO PORT', 'DISCHARGED', 'ESTIMATED ONBOARD DATE', 'PLANNED PICK UP DATE', 'SHIPMENT ONBOARD DATE', 'RDC ATD', 'CONTAINER LOADED ON BOARD', 'DAILY TRACK AND TRACE', 'TRANSIT PORT ATA', 'ARRIVAL AT CFS OR CY', 'CARGO ARRIVAL AT DESTINATION', 'IMP CUSTOMS CLEARANCE START', 'IMP CUSTOMS CLEARANCE FINISHED', 'TRANSIT PORT CUSTOMS RELEASE', 'ARRIVAL AT PORT', 'TRANSIT PORT ATD', 'PICKED UP'}
# 主要是SHIPMENT ONBOARD DATE和ARRIVAL AT PORT


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


def encode(encodee: Union[dict, set, list], strategy='one-hot', add_unknown=True, size=Optional[int]) -> dict:
    """
    encode various states with selected strategy
    """
    if type(encodee) == dict:
        encodee = list(encodee.keys())
    elif type(encodee) == set:
        encodee == list(encodee)
    
    if not add_unknown:
        raise NotImplementedError

    if strategy == 'one-hot':
        e = np.eye(len(encodee))
        encoder = {key: e[i] for i, key in enumerate(encodee)}
        encoder.update({'<unk>': np.zeros(len(encodee))})
        return encoder
    elif strategy == 'uniform' and size is not None:
        e = np.random.uniform(size=(len(encodee)+1, size))
        encoder = {key: e[i].tolist() for i, key in enumerate(encodee)}
        encoder.update({'<unk>': e[-1].tolist()})
        return encoder
    else:
        raise NotImplementedError
    return {}


def main():
    pass


import tensorflow as tf

class DataProcessor:
    """
    data processor for the project
    Scheme: test data oriented training (limited TRACE in test data)
    """
    def __init__(self, output_dir='data/preprocessed'):
        self.cache_dir = output_dir

        # store encodings (temp)
        self.vessel_status_encoding = encode(vessel_status)
        self.vessel_datasource_encoding = encode(vessel_datasource)
        self.vesselMMSI_encoding = None
        self.carrier_name_encoding = None

    def dump_test_oriented_train_data(self, port_path, gps_path, event_path, test_path, rewrite=False, debug=False):
        # prepare data
        test_df = pd.read_csv(test_path, low_memory=False)
        test_trace_dict = test_df['TRANSPORT_TRACE'].value_counts().to_dict()
        test_trace_split_list = [x.split('-') for x in test_trace_dict]
        self.vesselMMSI_encoding = encode(test_df['vesselMMSI'].value_counts().to_dict(), strategy='uniform', size=9)   # might be learned
        self.carrier_name_encoding = encode(test_df['carrierName'].value_counts().to_dict(), strategy='one-hot')  # might be learned
        port_df = pd.read_csv(port_path)
        port_loc = {r['TRANS_NODE_NAME'] : [r['LONGITUDE'], r['LATITUDE']] for _, r in port_df.iterrows()}
        # eta
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)
        order_to_route_eta, start_dicts, _ = self._generate_test_data_oriented_eta(gps_record_reader, port_loc, cached=True)
        # in-trajectory information extraction (not realized)
        # generate training data
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)  # reset
        count = 0
        for chunk in gps_record_reader:
            start_time = time.time()
            feature_narr, target_narr, trace_narr = self._generate_feature_and_target(
                chunk, 
                port_to_loc=port_loc,
                order_to_route_eta=order_to_route_eta,
                start_dicts=start_dicts
            )
            end_time = time.time()
            print('feature and target generation took {} seconds'.format(end_time-start_time))
            if rewrite: # for debugging
                feature_file = open(os.path.join(self.cache_dir, 'train_features.csv'), 'wb')
                target_file = open(os.path.join(self.cache_dir, 'train_target.csv'), 'wb')
                trace_file = open(os.path.join(self.cache_dir, 'train_trace.csv'), 'wb')
                rewrite = False
            else:
                feature_file = open(os.path.join(self.cache_dir, 'train_features.csv'), 'ab')
                target_file = open(os.path.join(self.cache_dir, 'train_target.csv'), 'ab')
                trace_file = open(os.path.join(self.cache_dir, 'train_trace.csv'), 'ab')
            np.savetxt(feature_file, feature_narr, delimiter=',')
            np.savetxt(target_file, target_narr, delimiter=',') # problematic: missing lines (TODO: modify this part)
            np.savetxt(trace_file, trace_narr, delimiter=',', fmt='%s')
            count += 1
            if debug and count > 0:   # for debugging
                break

    def dump_target(self, port_path, gps_path, event_path, test_path, rewrite=False, debug=False):
        """
        补救之前target算的是seconds而不是total_seconds
        """
        # prepare data
        test_df = pd.read_csv(test_path, low_memory=False)
        test_trace_dict = test_df['TRANSPORT_TRACE'].value_counts().to_dict()
        test_trace_split_list = [x.split('-') for x in test_trace_dict]
        self.vesselMMSI_encoding = encode(test_df['vesselMMSI'].value_counts().to_dict(), strategy='uniform', size=9)   # might be learned
        self.carrier_name_encoding = encode(test_df['carrierName'].value_counts().to_dict(), strategy='one-hot')  # might be learned
        port_df = pd.read_csv(port_path)
        port_loc = {r['TRANS_NODE_NAME'] : [r['LONGITUDE'], r['LATITUDE']] for _, r in port_df.iterrows()}
        # eta
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)
        order_to_route_eta, start_dicts, _ = self._generate_test_data_oriented_eta(gps_record_reader, port_loc, cached=True)
        # in-trajectory information extraction (not realized)
        # generate training data
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)  # reset
        count = 0
        for chunk in gps_record_reader:
            start_time = time.time()
            target_narr = self._generate_target(
                chunk, 
                port_to_loc=port_loc,
                order_to_route_eta=order_to_route_eta,
                start_dicts=start_dicts
            )
            end_time = time.time()
            print('feature and target generation took {} seconds'.format(end_time-start_time))
            if rewrite: # for debugging
                target_file = open(os.path.join(self.cache_dir, 'train_target.csv'), 'w')
                rewrite = False
            else:
                target_file = open(os.path.join(self.cache_dir, 'train_target.csv'), 'a')
            # np.savetxt(target_file, target_narr, delimiter=',', newline='\r\n')
            target_file.writelines(['{}\n'.format(x) for x in target_narr])
            target_file.close()
            count += 1
            if debug and count > 0:   # for debugging
                break

    def load_train_data(self, port_path, gps_path, event_path, rewrite=False):
        # load port
        port_df = pd.read_csv(port_path)
        port_loc = {} # portName -> location(coordinates)
        for _, r in port_df.iterrows() :
            port_loc.update({r['TRANS_NODE_NAME'] : [r['LATITUDE'], r['LONGITUDE']]})
        # load gps records
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)
        # missing value filling (mainly eta)
        order_to_port, order_to_eta, port_standardizing = self._generate_etas(port_loc, 
                                                                            gps_record_reader, 
                                                                            cached=True, 
                                                                            test_mode=True)
        # generate training data
        gps_record_reader = pd.read_csv(gps_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)  # reset
        count = 0
        for chunk in gps_record_reader:
            # 1. fill absent data
            #   1.1 NextPort -> [start_port, end_port]
            # chunk = pd.concat(chunk[0].apply(lambda x: order_to_port[x]).apply(pd.Series, index=[])
            # 2. generate feature and target
            feature_df, target_df = self._generate_feature_and_target(
                chunk, 
                port_to_loc=port_loc,
                order_to_port=order_to_port,
                order_to_eta=order_to_eta
            )
            if rewrite: # for debugging
                feature_df.to_csv(os.path.join(self.cache_dir, 'train_features.csv'), mode='w', header=False)
                target_df.to_csv(os.path.join(self.cache_dir, 'train_target.csv'), mode='w', header=False)
                rewrite = False
            else:
                feature_df.to_csv(os.path.join(self.cache_dir, 'train_features.csv'), mode='a', header=False)
                target_df.to_csv(os.path.join(self.cache_dir, 'train_target.csv'), mode='a', header=False)
            count += 1
            if count > 1:   # for debugging
                break
        
        return pd.read_csv(os.path.join(self.cache_dir, 'train_features.csv'), header=None, low_memory=False, chunksize=CHUNK_SIZE), \
            pd.read_csv(os.path.join(self.cache_dir, 'train_target.csv'), header=None, low_memory=False, chunksize=CHUNK_SIZE)

    def _filter_original_data(self, path, cached=False, test_mode=False):
        """
        see `data_selection.ipynb`
        """
        ...

    def _handle_missing_values(self, gps_record_reader, cached=True, test_mode=False):
        """
        filling missing values after `speed`
        """
        ...

    def _locate_between_ports(self, chunk, port_to_loc, trace_set, trace_split_list):
        """
        give start port and end port, remove out of range data

        strategies (support start port is 'A' end port is 'B'):
        1. first decide if is exact trace A-B. if so, no more bothering
        2. if A-C-B, can be seen as exact A-B (real case not found though)
        3. if C-A-E-..-B-D-..., decide if out of range (rather than in range), if not, can be seen as exact A-B

        the difficulties lies in case 3 where `out of range` is hard to judge. criteria are like:
        a) locate in the square range
        b) distance (chosen)
        """
        drop_indexes = []
        port_pair_list = []
        for longtitude, latitude, trace in zip(chunk[3], chunk[4], chunk[12]):
            s_trace = trace.split('-')
            if trace in trace_set:
                port_pair_list.append(s_trace)
            else:
                if (s_trace[0], s_trace[-1]) in trace_split_list:
                    port_pair_list.append(s_trace)
                else:
                    for pair in trace_split_list:
                        if pair[0] in s_trace and pair[1] in s_trace:
                            sp_index = s_trace.index(pair[0])
                            ep_index = s_trace.index(pair[1])
                            # radical judgement
                            sp = port_to_loc[pair[0]]
                            ep = port_to_loc[pair[1]]
                            if getDistance(longitude, latitude, sp[0], sp[1]) + getDistance(longtitude, latitude, ep[0], ep[1]) \
                                < 2 * getDistancegetDistance(sp[0], sp[1], ep[0], ep[1]):
                                port_pair_list.append(pair)

                            break  # assume no overlap trace
        return pd.concat([chunk.drop(labels=drop_indexes, axis=0), pd.DataFrame({'portPair': port_pair_list})], axis=1)

    def _generate_test_data_oriented_eta(self, gps_record_reader, port_to_loc, cached=True):

        if cached:
            with open(os.path.join(self.cache_dir, 'order_to_route_eta.h5'), 'rb') as f:
                order_to_route_eta = pickle.load(f)
            with open(os.path.join(self.cache_dir, 'start_dicts.h5'), 'rb') as f:
                start_dicts = pickle.load(f)
            with open(os.path.join(self.cache_dir, 'dest_dicts.h5'), 'rb') as f:
                dest_dicts = pickle.load(f)
            # test eta
            start_pn = ['CNYTN', 'CNSHK', 'CNHKG', 'CNSHA', 'COBUN', 'HKHKG']
            dest_pn = ['MXZLO', 'PAONX', 'CLVAP', 'ARENA', 'MYTPP', 'MATNG', 'GRPIR', 'CAVAN', 'SGSIN', 'RTM', 'SIKOP', 'HKHKG',
                'FRFOS', 'NZAKL', 'ESALG', 'ZADUR', 'PAMIT', 'PKQCT', 'LBBEY', 'MTMLA']
            routes = ['CNYTN-MXZLO', 'CNYTN-PAONX', 'CNSHK-CLVAP', 'CNYTN-ARENA', 'CNSHK-MYTPP', 'CNYTN-MATNG', 'CNSHK-GRPIR', 'CNYTN-CAVAN', 'CNHKG-MXZLO',
                'CNSHK-SGSIN', 'CNYTN-RTM', 'CNSHA-SGSIN', 'CNSHK-SIKOP', 'COBUN-HKHKG', 'HKHKG-FRFOS', 'CNYTN-NZAKL', 'CNSHK-ESALG', 'CNSHK-ZADUR',
                'CNSHA-PAMIT', 'CNSHK-PKQCT', 'CNSHK-LBBEY', 'CNYTN-MTMLA']
            etas = {}
            for i in range(6) :
                orders = list(start_dicts[i].keys())
                for order in orders :
                    for j in range(20) :
                        if order in dest_dicts[j] :
                            route = start_pn[i] + '-' + dest_pn[j]
                            if route in etas :
                                etas[route].append(dest_dicts[j][order] - start_dicts[i][order])
                            else :
                                etas.update({route: [dest_dicts[j][order] - start_dicts[i][order]]})
            with open('data/preprocessed/etas.txt', 'w') as f:
                for i in routes :
                    if i in etas :
                        f.write('{} {} \n\n'.format(i, etas[i]))
            return order_to_route_eta, start_dicts, dest_dicts

        #起始港口的位置、名称
        start_ports = [port_to_loc['CNYTN'], port_to_loc['CNSHK'], port_to_loc['CNHKG'], port_to_loc['CNSHA'], port_to_loc['COBUN'], port_to_loc['HKHKG']]
        start_pn = ['CNYTN', 'CNSHK', 'CNHKG', 'CNSHA', 'COBUN', 'HKHKG']
        #终点港口的位置、名称
        dest_ports = [port_to_loc['MXZLO'], port_to_loc['PAONX'], port_to_loc['CLVAP'], port_to_loc['ARENA'], port_to_loc['MYTPP'], port_to_loc['MATNG'],
                    port_to_loc['GRPIR'], port_to_loc['CAVAN'], port_to_loc['SGSIN'], port_to_loc['RTM'], port_to_loc['SIKOP'], port_to_loc['HKHKG'],
                    port_to_loc['FRFOS'], port_to_loc['NZAKL'], port_to_loc['ESALG'], port_to_loc['ZADUR'], port_to_loc['PAMIT'], port_to_loc['PKQCT'],
                    port_to_loc['LBBEY'], port_to_loc['MTMLA']]
        dest_pn = ['MXZLO', 'PAONX', 'CLVAP', 'ARENA', 'MYTPP', 'MATNG', 'GRPIR', 'CAVAN', 'SGSIN', 'RTM', 'SIKOP', 'HKHKG',
                'FRFOS', 'NZAKL', 'ESALG', 'ZADUR', 'PAMIT', 'PKQCT', 'LBBEY', 'MTMLA']
        #所有路线
        routes = ['CNYTN-MXZLO', 'CNYTN-PAONX', 'CNSHK-CLVAP', 'CNYTN-ARENA', 'CNSHK-MYTPP', 'CNYTN-MATNG', 'CNSHK-GRPIR', 'CNYTN-CAVAN', 'CNHKG-MXZLO',
                'CNSHK-SGSIN', 'CNYTN-RTM', 'CNSHA-SGSIN', 'CNSHK-SIKOP', 'COBUN-HKHKG', 'HKHKG-FRFOS', 'CNYTN-NZAKL', 'CNSHK-ESALG', 'CNSHK-ZADUR',
                'CNSHA-PAMIT', 'CNSHK-PKQCT', 'CNSHK-LBBEY', 'CNYTN-MTMLA']
        # CNSHA-PAMIT HKHKG-FRFOS (special)
        CNSHA_idx = start_pn.index('CNSHA')
        HKHKG_idx = start_pn.index('HKHKG')
        PAMIT_idx = dest_pn.index('PAMIT')
        FRFOS_idx = dest_pn.index('FRFOS')
        CNSHA_loc = start_ports[start_pn.index('CNSHA')]
        HKHKG_loc = start_ports[start_pn.index('HKHKG')]
        PAMIT_loc = dest_ports[dest_pn.index('PAMIT')]
        FRFOS_loc = dest_ports[dest_pn.index('FRFOS')]
        #记录订单在对应港口出发的时间
        start_dicts = []
        for i in range(6) :
            dic = {}
            start_dicts.append((dic))
        #记录订单到达对应港口的时间
        dest_dicts = []
        for i in range(20) :
            dic = {}
            dest_dicts.append(dic)
        
        for chunk in gps_record_reader:
            # chunk.drop([0], axis=1, inplace=True)   # additional row index rendered by data-filtering
            chunk.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                     'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                     'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], infer_datetime_format=True)
            for loadingOrder, longitude, latitude, speed, timestamp in zip(chunk['loadingOrder'], chunk['longitude'], chunk['latitude'],
                                                               chunk['speed'], chunk['timestamp']) :
                if speed > 0 :#速度不为0，则在航行中
                    # CNSHA-PAMIT HKHKG-FRFOS
                    for port, i in zip((CNSHA_loc, HKHKG_loc), (CNSHA_idx, HKHKG_idx)):
                        if abs(longitude - port[0]) < 0.5 and abs(latitude - port[1]) < 0.5 :
                            if loadingOrder in start_dicts[i] :
                                start_dicts[i][loadingOrder] = timestamp
                            else :
                                start_dicts[i].update({loadingOrder: timestamp})
                    for port, i in zip((PAMIT_loc, FRFOS_loc), (PAMIT_idx, FRFOS_idx)):
                        if abs(longitude - port[0]) < 0.5 and abs(latitude - port[1]) < 0.5 :
                            if loadingOrder not in dest_dicts[i] :
                                dest_dicts[i].update({loadingOrder: timestamp})
                else:
                    #判断是否从某港口出发或到达某港口
                    for i in range(6) :
                        port = start_ports[i]
                        if abs(longitude - port[0]) < 0.1 and abs(latitude - port[1]) < 0.1 :
                            if loadingOrder in start_dicts[i] :
                                start_dicts[i][loadingOrder] = timestamp
                            else :
                                start_dicts[i].update({loadingOrder: timestamp})
                            break
                    for i in range(20) :
                        port = dest_ports[i]
                        if abs(longitude - port[0]) < 0.1 and abs(latitude - port[1]) < 0.1 :
                            if loadingOrder not in dest_dicts[i] :
                                dest_dicts[i].update({loadingOrder: timestamp})
                            break
        #找出所有出发、到达的组合
        # etas = {}
        order_to_route_eta = {}
        for i in range(6) :
            orders = list(start_dicts[i].keys())
            for order in orders :
                for j in range(20) :
                    if order in dest_dicts[j] :
                        route = start_pn[i] + '-' + dest_pn[j]
                        # if route in etas :
                        #     etas[route].append(dest_dicts[j][order] - start_dicts[i][order])
                        # else :
                        #     etas.update({route: [dest_dicts[j][order] - start_dicts[i][order]]})
                        if order in order_to_route_eta:
                            order_to_route_eta[order].append((route, dest_dicts[j][order] - start_dicts[i][order]))
                        else:
                            order_to_route_eta[order] = [(route, dest_dicts[j][order] - start_dicts[i][order])]
                        
        with open(os.path.join(self.cache_dir, 'order_to_route_eta.h5'), 'wb') as f:
            pickle.dump(order_to_route_eta, f)
        with open(os.path.join(self.cache_dir, 'start_dicts.h5'), 'wb') as f:
            pickle.dump(start_dicts, f)
        with open(os.path.join(self.cache_dir, 'dest_dicts.h5'), 'wb') as f:
            pickle.dump(dest_dicts, f)

        return order_to_route_eta, start_dicts, dest_dicts
        # #找到所需路径的ATA
        # with open('data/preprocessed/etas.txt', 'w') as f:
        #     for i in routes :
        #         if i in etas :
        #             f.write('{} {} \n\n'.format(i, etas[i]))

    def _generate_etas(self, port_loc, gps_record_reader, cached=True, test_mode=False):
        """
        补充缺失的vesselNextPort和vesselNextPortETA进行的工作（广义上属于补充缺失数据部分）
        回填port，之后查eta
        """
        if cached:
            port_mapping = pickle.load(open(os.path.join(self.cache_dir, 'port_mapping'), 'rb'))
            eta_mapping = pickle.load(open(os.path.join(self.cache_dir, 'eta_mapping'), 'rb'))
            port_standardizing = pickle.load(open(os.path.join(self.cache_dir, 'port_standardizing'), 'rb'))
            return port_mapping, eta_mapping, port_standardizing

        port_mapping = {} # order -> [current nextPort, distance, timestamp, speed, start port (?)]
        eta_mapping = {} # ["{order} from {start_port} to {end_port}"] -> timestamp
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
                        str = order + ' from ' + port_mapping[order][4] + ' to ' + port_mapping[order][0]
                        if str in eta_mapping :
                            if speed < port_mapping[order][3] : #slowing, preparing for stop
                                eta_mapping[str] = timestamp
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
                        port_mapping[order][4] = port_mapping[order][0]
                        port_mapping[order][0] = nextport
                        #port_mapping[order][0] = [port_loc[nextport][0], port_loc[nextport][1]]
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
                        port_mapping.update({order: [nextport, cur_dis, timestamp, speed, 'start']})
                        #port_mapping.update({order: [[port_loc[nextport][0], port_loc[nextport][1]], cur_dis, timestamp, speed, [longitude, latitude]]]})

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

    def _loading_order_specific_trajectory_information_extraction(
        self, gps_record_reader, port_loc, order_to_route_eta,
        cached=True, test_mode=False
        ):
        """
        not so necessary
        since trajectory information is not fully given in testing set
        """
        on_disk_set = {}
        in_memory_set = {}
        order_cache_dir = os.path.join(self.cache_dir, 'order_cache')

        # extract info
        # for chunk in gps_record_reader:
        #     for loadingOrder, ... in zip(chunk[0], ...):
        #         if loadingOrder not in order_to_route_eta:
        #             continue
        
        # concat dumps

    def dump_test_data(self, port_path, test_data_path, rewrite=True, debug=True):
        # prepare data
        test_df = pd.read_csv(test_data_path, low_memory=False)
        test_trace_dict = test_df['TRANSPORT_TRACE'].value_counts().to_dict()
        test_trace_split_list = [x.split('-') for x in test_trace_dict]
        self.vesselMMSI_encoding = encode(test_df['vesselMMSI'].value_counts().to_dict(), strategy='uniform', size=9)   # might be learned
        self.carrier_name_encoding = encode(test_df['carrierName'].value_counts().to_dict(), strategy='one-hot')  # might be learned
        port_df = pd.read_csv(port_path)
        port_loc = {r['TRANS_NODE_NAME'] : [r['LONGITUDE'], r['LATITUDE']] for _, r in port_df.iterrows()}
        # load
        test_data_reader = pd.read_csv(test_data_path, low_memory=False, chunksize=CHUNK_SIZE)
        count = 0
        for chunk in test_data_reader:
            feature_narr, trace_narr = self._generate_feature(chunk, port_loc, None, source='test')  # todo: modify
            if rewrite: # for debugging
                feature_file = open(os.path.join(self.cache_dir, 'test_features.csv'), 'wb')
                trace_file = open(os.path.join(self.cache_dir, 'test_trace.csv'), 'wb')
                rewrite = False
            else:
                feature_file = open(os.path.join(self.cache_dir, 'test_features.csv'), 'ab')
                trace_file = open(os.path.join(self.cache_dir, 'test_trace.csv'), 'ab')
            np.savetxt(feature_file, feature_narr, delimiter=',')
            np.savetxt(trace_file, trace_narr, delimiter=',', fmt='%s')
            count += 1
            if debug and count > 0:   # for debugging
                break

    def seperate_dataset_by_trace(self, source, trace_map_path, feature_path=None, target_path=None, rewrite=True):
        """
        only for feature or target
        """
        if feature_path is not None:
            mode = 'feature'
        elif target_path is not None:
            mode = 'target'
        else:
            raise NotImplementedError

        def dump_chunks(chunk, trace_to_indexes, trace_to_file):
            for trace, index_list in trace_to_indexes.items():
                chunk.iloc[index_list][:].to_csv(trace_to_file[trace], mode='a', header=False, index=0) # index=0 means no index

        test_traces = ['CNYTN-MXZLO', 'CNYTN-PAONX', 'CNSHK-CLVAP', 'CNYTN-ARENA', 'CNSHK-MYTPP', 'CNYTN-MATNG', 'CNSHK-GRPIR', 'CNYTN-CAVAN', 'CNHKG-MXZLO',
                'CNSHK-SGSIN', 'CNYTN-RTM', 'CNSHA-SGSIN', 'CNSHK-SIKOP', 'COBUN-HKHKG', 'HKHKG-FRFOS', 'CNYTN-NZAKL', 'CNSHK-ESALG', 'CNSHK-ZADUR',
                'CNSHA-PAMIT', 'CNSHK-PKQCT', 'CNSHK-LBBEY', 'CNYTN-MTMLA']
        if rewrite:
            for trace in test_traces:
                f = open(os.path.join(self.cache_dir, 'trace_specific_dataset/{}_{}_{}.csv'.format(trace, source, mode)), 'wb')
                f.close()
        # files = {trace: open(os.path.join(self.cache_dir, 'trace_specific_dataset/{}_{}_{}.csv'.format(trace, mode)), 'ab') for trace in test_traces}
        files = {trace: os.path.join(self.cache_dir, 'trace_specific_dataset/{}_{}_{}.csv'.format(trace, source, mode)) for trace in test_traces}
        trace_reader = pd.read_csv(trace_map_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)
        if mode == 'feature':
            data_reader = pd.read_csv(feature_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)
        else:
            data_reader = pd.read_csv(target_path, header=None, low_memory=False, chunksize=CHUNK_SIZE)
        for trace_chunk, data_chunk in zip(trace_reader, data_reader):
            trace_mapping_cache = {}
            for i, trace in enumerate(trace_chunk[0]):
                if trace in trace_mapping_cache:
                    trace_mapping_cache[trace].append(i)
                else:
                    trace_mapping_cache[trace] = [i]
            dump_chunks(data_chunk, trace_mapping_cache, files)
        # close files
        # for f in files.values():
        #     f.close()
            
    def form_results(self, test_data, pred_deltas):
        """
        results have exactly the same rows as test data
        ..., ETA, creatDate
        """
        pass
        # ETA = timestamp + pred_delta[x]
        test_data['creatDate'] = datetime.datetime.now().strftime(RECORD_TIME_FORMAT)

    def _generate_feature_and_target(
            self,
            chunk: pd.DataFrame, 
            port_to_loc: dict,
            order_to_route_eta: dict, 
            start_dicts: dict, 
        ):
        """
        the function to generate the training data
        :param chunk: gps data chunk
        """
        test_routes = set(['CNYTN-MXZLO', 'CNYTN-PAONX', 'CNSHK-CLVAP', 'CNYTN-ARENA', 'CNSHK-MYTPP', 'CNYTN-MATNG', 'CNSHK-GRPIR', 'CNYTN-CAVAN', 'CNHKG-MXZLO',
                'CNSHK-SGSIN', 'CNYTN-RTM', 'CNSHA-SGSIN', 'CNSHK-SIKOP', 'COBUN-HKHKG', 'HKHKG-FRFOS', 'CNYTN-NZAKL', 'CNSHK-ESALG', 'CNSHK-ZADUR',
                'CNSHA-PAMIT', 'CNSHK-PKQCT', 'CNSHK-LBBEY', 'CNYTN-MTMLA'])
        start_pn = ['CNYTN', 'CNSHK', 'CNHKG', 'CNSHA', 'COBUN', 'HKHKG']
        chunk.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                    'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                    'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']

        feature_dict = {col: [] for col in [
            'carrier_name_code', 'MMSI_code', 'status_code', 'datasource_code',
            'longitude', 'latitude', 'speed', 'direction',
            'start_port_location', 'end_port_location',
            'time_delta_from_start', 'distance_from_start', 'distance_to_end']}
        target_list = []
        trace_list = []
        
        # features: [先最简单的开始]

        cached_orders = {}  # [order -> route, eta, start_time]. cached for chunk. faster mapping
        # count = 0
        for loadingOrder,carrierName,timestamp,longitude,latitude,vesselMMSI,speed,\
            direction,vesselNextport,vesselNextportETA,vesselStatus,vesselDatasource,TRANSPORT_TRACE in zip(
                chunk['loadingOrder'], chunk['carrierName'], chunk['timestamp'], chunk['longitude'],
                chunk['latitude'], chunk['vesselMMSI'], chunk['speed'], chunk['direction'], chunk['vesselNextport'],
                chunk['vesselNextportETA'], chunk['vesselStatus'], chunk['vesselDatasource'], chunk['TRANSPORT_TRACE']
            ):
            if loadingOrder not in cached_orders:
                if loadingOrder not in order_to_route_eta:
                    continue
                else:
                    route_eta_list = order_to_route_eta[loadingOrder].copy()
                    route_eta = None
                    for tr in route_eta_list:
                        if tr[0] in test_routes:
                            route_eta = list(tr)
                            break
                    if not route_eta:
                        # del order_to_route_eta[getattr(row, 'loadingOrder')]
                        continue
                    sp_idx = start_pn.index(route_eta[0].split('-')[0])
                    route_eta.append(start_dicts[sp_idx][loadingOrder])
                    cached_orders[loadingOrder] = route_eta
            route_eta_stime = cached_orders[loadingOrder]
            trace_list.append(route_eta_stime[0])
            carrier_name_code = self.carrier_name_encoding[carrierName] if carrierName in self.carrier_name_encoding else self.carrier_name_encoding['<unk>']
            MMSI_code = self.vesselMMSI_encoding[vesselMMSI] if vesselMMSI in self.vesselMMSI_encoding else self.vesselMMSI_encoding['<unk>']
            status_code = self.vessel_status_encoding[vesselStatus] if vesselStatus in self.vessel_status_encoding else self.vessel_status_encoding['<unk>']
            datasource_code = self.vessel_datasource_encoding[vesselDatasource] if vesselDatasource in self.vessel_datasource_encoding else vessel_datasource['<unk>']
            # longitude, latitude, speed, direction INHERIT
            route = route_eta_stime[0].split('-')
            target_list.append(abs(route_eta_stime[1].total_seconds()))
            start_port_location = port_to_loc[route[0]]
            end_port_location = port_to_loc[route[1]]
            time_delta_from_start = abs(pd.to_datetime(timestamp, infer_datetime_format=True) - route_eta_stime[2]).total_seconds()
            distance_from_start = getDistance(start_port_location[0], start_port_location[1],
                                        longitude, latitude)
            distance_to_end = getDistance(end_port_location[0], end_port_location[1],
                                        longitude, latitude)
            feature_dict['carrier_name_code'].append(carrier_name_code)
            feature_dict['MMSI_code'].append(MMSI_code)
            feature_dict['status_code'].append(status_code)
            feature_dict['datasource_code'].append(datasource_code)
            feature_dict['longitude'].append(longitude)
            feature_dict['latitude'].append(latitude)
            feature_dict['speed'].append(speed)
            feature_dict['direction'].append(direction)
            feature_dict['start_port_location'].append(start_port_location)
            feature_dict['end_port_location'].append(end_port_location)
            feature_dict['time_delta_from_start'].append(time_delta_from_start)
            feature_dict['distance_from_start'].append(distance_from_start)
            feature_dict['distance_to_end'].append(distance_to_end)
            # count += 1
            # if count % 1024 * 4 == 0:
            #     print(count)
            #     break

        feature_df = pd.DataFrame.from_dict(feature_dict)
        # feature_df = pd.concat([
        #     feature_df['carrier_name_code'].apply(pd.Series),
        #     feature_df['MMSI_code'].apply(pd.Series),
        #     feature_df['status_code'].apply(pd.Series),
        #     feature_df['datasource_code'].apply(pd.Series),
        #     feature_df[['longitude', 'latitude', 'speed', 'direction']],
        #     feature_df['start_port_location'].apply(pd.Series),
        #     feature_df['end_port_location'].apply(pd.Series),
        #     feature_df[['time_delta_from_start', 'distance_from_start', 'distance_to_end']]
        # ], axis=1)
        feature_narr = np.concatenate([
            feature_df['carrier_name_code'].apply(pd.Series).values,
            feature_df['MMSI_code'].apply(pd.Series).values,
            feature_df['status_code'].apply(pd.Series).values,
            feature_df['datasource_code'].apply(pd.Series).values,
            feature_df[['longitude', 'latitude', 'speed', 'direction']].values,
            feature_df['start_port_location'].apply(pd.Series).values,
            feature_df['end_port_location'].apply(pd.Series).values,
            feature_df[['time_delta_from_start', 'distance_from_start', 'distance_to_end']].values
        ], axis=1)
        
        target_narr = np.array(target_list)
        trace_narr = np.array(trace_list)
        assert target_narr.shape[0] == feature_narr.shape[0]
        return feature_narr, target_narr, trace_narr

    def _generate_target(
            self,
            chunk: pd.DataFrame, 
            port_to_loc: dict,
            order_to_route_eta: dict, 
            start_dicts: dict, 
        ):
        """
        补救之前target算的是seconds而不是total_seconds
        """
        test_routes = set(['CNYTN-MXZLO', 'CNYTN-PAONX', 'CNSHK-CLVAP', 'CNYTN-ARENA', 'CNSHK-MYTPP', 'CNYTN-MATNG', 'CNSHK-GRPIR', 'CNYTN-CAVAN', 'CNHKG-MXZLO',
                'CNSHK-SGSIN', 'CNYTN-RTM', 'CNSHA-SGSIN', 'CNSHK-SIKOP', 'COBUN-HKHKG', 'HKHKG-FRFOS', 'CNYTN-NZAKL', 'CNSHK-ESALG', 'CNSHK-ZADUR',
                'CNSHA-PAMIT', 'CNSHK-PKQCT', 'CNSHK-LBBEY', 'CNYTN-MTMLA'])
        start_pn = ['CNYTN', 'CNSHK', 'CNHKG', 'CNSHA', 'COBUN', 'HKHKG']
        chunk.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                    'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                    'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']

        feature_dict = {col: [] for col in [
            'carrier_name_code', 'MMSI_code', 'status_code', 'datasource_code',
            'longitude', 'latitude', 'speed', 'direction',
            'start_port_location', 'end_port_location',
            'time_delta_from_start', 'distance_from_start', 'distance_to_end']}
        target_list = []
        # features: [先最简单的开始]

        cached_orders = {}  # [order -> route, eta, start_time]. cached for chunk. faster mapping
        # count = 0
        for loadingOrder,carrierName,timestamp,longitude,latitude,vesselMMSI,speed,\
            direction,vesselNextport,vesselNextportETA,vesselStatus,vesselDatasource,TRANSPORT_TRACE in zip(
                chunk['loadingOrder'], chunk['carrierName'], chunk['timestamp'], chunk['longitude'],
                chunk['latitude'], chunk['vesselMMSI'], chunk['speed'], chunk['direction'], chunk['vesselNextport'],
                chunk['vesselNextportETA'], chunk['vesselStatus'], chunk['vesselDatasource'], chunk['TRANSPORT_TRACE']
            ):
            if loadingOrder not in cached_orders:
                if loadingOrder not in order_to_route_eta:
                    continue
                else:
                    route_eta_list = order_to_route_eta[loadingOrder].copy()
                    route_eta = None
                    for tr in route_eta_list:
                        if tr[0] in test_routes:
                            route_eta = list(tr)
                            break
                    if not route_eta:
                        # del order_to_route_eta[getattr(row, 'loadingOrder')]
                        continue
                    sp_idx = start_pn.index(route_eta[0].split('-')[0])
                    route_eta.append(start_dicts[sp_idx][loadingOrder])
                    cached_orders[loadingOrder] = route_eta
            route_eta_stime = cached_orders[loadingOrder]
            target_list.append(abs(route_eta_stime[1].total_seconds()))
            
        target_narr = np.array(target_list)
        return target_narr

    def _generate_feature(
            self,
            chunk: pd.DataFrame, 
            port_to_loc: dict, 
            order_to_route_eta: Optional[dict],
            source='test'
        ):
        """
        the primary function for feature engineering
        used for both training data and testing data
        """
        
        feature_dict = {col: [] for col in [
            'carrier_name_code', 'MMSI_code', 'status_code', 'datasource_code',
            'longitude', 'latitude', 'speed', 'direction',
            'start_port_location', 'end_port_location',
            'time_delta_from_start', 'distance_from_start', 'distance_to_end']}
        trace_list = []

        if source == 'train':
            raise NotImplementedError

        elif source == 'test':
            chunk.columns = ['loadingOrder', 'timestamp', 'longitude', 'latitude',
                'speed', 'direction', 'carrierName', 'vesselMMSI',
                'onboardDate', 'TRANSPORT_TRACE']
            for loadingOrder,timestamp,longitude,latitude,speed,direction,carrierName,vesselMMSI,onboardDate,TRANSPORT_TRACE in zip(
                chunk['loadingOrder'], chunk['timestamp'], chunk['longitude'], chunk['latitude'],
                chunk['speed'], chunk['direction'], chunk['carrierName'], chunk['vesselMMSI'],
                chunk['onboardDate'], chunk['TRANSPORT_TRACE']
            ):
                trace_list.append(TRANSPORT_TRACE)
                carrier_name_code = self.carrier_name_encoding[carrierName]
                MMSI_code = self.vesselMMSI_encoding[vesselMMSI]
                status_code = self.vessel_status_encoding['under way using engine'] if speed != 0 else self.vessel_status_encoding['at anchor']
                # ^^^^^^^^ or self.vessel_status_encoding['<unk>']
                datasource_code = self.vessel_datasource_encoding['<unk>']
                route = TRANSPORT_TRACE.split('-')
                start_port_location = port_to_loc[route[0]]
                end_port_location = port_to_loc[route[1]]
                time_delta_from_start = abs(pd.to_datetime(timestamp, infer_datetime_format=True) - pd.to_datetime(onboardDate, infer_datetime_format=True).tz_localize('utc')).total_seconds()
                distance_from_start = getDistance(start_port_location[0], start_port_location[1],
                                            longitude, latitude)
                distance_to_end = getDistance(end_port_location[0], end_port_location[1],
                                            longitude, latitude)
                feature_dict['carrier_name_code'].append(carrier_name_code)
                feature_dict['MMSI_code'].append(MMSI_code)
                feature_dict['status_code'].append(status_code)
                feature_dict['datasource_code'].append(datasource_code)
                feature_dict['longitude'].append(longitude)
                feature_dict['latitude'].append(latitude)
                feature_dict['speed'].append(speed)
                feature_dict['direction'].append(direction)
                feature_dict['start_port_location'].append(start_port_location)
                feature_dict['end_port_location'].append(end_port_location)
                feature_dict['time_delta_from_start'].append(time_delta_from_start)
                feature_dict['distance_from_start'].append(distance_from_start)
                feature_dict['distance_to_end'].append(distance_to_end)
        
        else:
            raise NotImplementedError

        feature_df = pd.DataFrame.from_dict(feature_dict)
        feature_narr = np.concatenate([
            feature_df['carrier_name_code'].apply(pd.Series).values,
            feature_df['MMSI_code'].apply(pd.Series).values,
            feature_df['status_code'].apply(pd.Series).values,
            feature_df['datasource_code'].apply(pd.Series).values,
            feature_df[['longitude', 'latitude', 'speed', 'direction']].values,
            feature_df['start_port_location'].apply(pd.Series).values,
            feature_df['end_port_location'].apply(pd.Series).values,
            feature_df[['time_delta_from_start', 'distance_from_start', 'distance_to_end']].values
        ], axis=1)
        trace_narr = np.array(trace_list)
        return feature_narr, trace_narr
    

if __name__ == '__main__':
    main()