# TODO:
# 1. load data - 通用（训练、预测），相关数据结构
# 2. 数据集拆分 - train_test_split, shuffle
# 3. 数据清洗 - 简单部分，复杂的放到extract_feature.py

def foo():
    print(233)


import numpy as np
import pandas as pd

import random
import datetime

import matplotlib.pyplot as plt
import seaborn as sns


# contants (TODO: define enum)
from enum import Enum

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

event_code = {nan, 'TRANSIT PORT ETD', 'UPDATE SHIPMENT ETA', 'TRANSIT PORT ETA', 'SHIPMENT MIT INBOUND DATE', 'ESTIMATED ARRIVAL TIME TO PORT', 'TRANSIT PORT DECLARATION BEGIN', 'ESTIMATED ARRIVAL TO PORT', 'DISCHARGED', 'ESTIMATED ONBOARD DATE', 'PLANNED PICK UP DATE', 'SHIPMENT ONBOARD DATE', 'RDC ATD', 'CONTAINER LOADED ON BOARD', 'DAILY TRACK AND TRACE', 'TRANSIT PORT ATA', 'ARRIVAL AT CFS OR CY', 'CARGO ARRIVAL AT DESTINATION', 'IMP CUSTOMS CLEARANCE START', 'IMP CUSTOMS CLEARANCE FINISHED', 'TRANSIT PORT CUSTOMS RELEASE', 'ARRIVAL AT PORT', 'TRANSIT PORT ATD', 'PICKED UP'}



# support functions

def parse_trace(trace_string):
    trace = trace_string.split('-')


def main():
    pass


from types import Optional, Union
import tensorflow as tf

class DataProcessor:
    """
    data processor for NN
    """
    def __init__(self):
        self.something = None

    @classmethod
    def load_gps(self):
        # load
        # get paths
        # vectorize
        # return
    
    @classmethod
    def load_test_data(self):
        # load
        # get paths
        # vectorize
        # return
    
    def _get_paths(self, df):
        # path by order

    def _get_query(self, df):
        # extract query from test data
    

if __name__ == '__main__':
    main()