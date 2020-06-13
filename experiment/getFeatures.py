import numpy as np
import pandas as pd
import operator
import csv
import time

def main():
    trainset=pd.read_csv('../head10m - 复件.csv', header=None, low_memory=False, nrows=100000, chunksize=1024 * 1024)

    #处理后的数据
    dataset = pd.DataFrame(columns=['loadingOrder', 'vesselNextport',
                                    'mmax', 'count', 'mmin', 'label', 'latitude_min',
                                    'latitude_max', 'latitude_mean', 'latitude_median',
                                    'longitude_min', 'longitude_max', 'longitude_mean',
                                    'longitude_median', 'speed_min', 'speed_max',
                                    'speed_mean', 'speed_median', 'direction_min',
                                    'direction_max', 'direction_mean', 'direction_median'])

    count = 0
    for chunk in trainset :#将空的Nextport补充完整
        chunk.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                            'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                            'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
        dict = {}#记录Nextport为空，且暂不确定的订单GPS数据
        currentnextport = {}#记录订单当前的Nextport
        train_gps_index = 0#记录订单索引
        for loadingOrder, speed, vesselNextport in zip(chunk['loadingOrder'], chunk['speed'], chunk['vesselNextport']):

            if vesselNextport!=vesselNextport:#当前数据Nextport为空时，查询currentnextport根据订单号给Nextport赋值，如果不在该字典中，就按定单号将索引值存入dict

                if speed==0:#订单当前速度为0代表到达港口,查询对应的currentnextport，然后删除该字典项。同一订单号连续几条speed为0的记录，可能只有第一条有Nextport
                    if loadingOrder in currentnextport:
                        chunk.loc[train_gps_index, 'vesselNextport']=currentnextport[loadingOrder]
                        del currentnextport[loadingOrder]

                else:
                    if loadingOrder in currentnextport:
                        chunk.loc[train_gps_index, 'vesselNextport']=currentnextport[loadingOrder]

                    else:
                        if loadingOrder not in dict:
                            dict[loadingOrder]=[train_gps_index]

                        else:
                            dict[loadingOrder].append(train_gps_index)

            else:#当前数据Nextport非空时，如果速度为0，在currentnextport中删除该字典项，否则更新字典项。按照当前订单号将存在dict中之前未确定Nextport的记录补全
                if speed!=0:
                    currentnextport[loadingOrder]=vesselNextport      

                else:
                    if loadingOrder in currentnextport:
                        del currentnextport[loadingOrder]

                if loadingOrder in dict:
                    temp=dict[loadingOrder]
                    for i in range(len(temp)):
                        chunk.loc[temp[i], 'vesselNextport']=vesselNextport
                    del dict[loadingOrder]

            train_gps_index += 1
            count += 1

        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], infer_datetime_format=True)
        #求得每条订单在每个航行段的起止时间
        etaTemp = chunk.groupby(['loadingOrder', 'vesselNextport'])['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        etaTemp['label'] = (etaTemp['mmax'] - etaTemp['mmin']).dt.total_seconds()#此处label即ETA
        #求得每条订单在每个航行段经纬度、速度大小、方向的最小值、最大值、平均值与中位值
        agg_function = ['min', 'max', 'mean', 'median']
        agg_col = ['latitude', 'longitude', 'speed', 'direction']
        groupdata = chunk.groupby(['loadingOrder', 'vesselNextport'])[agg_col].agg(agg_function).reset_index()
        groupdata.columns = ['loadingOrder', 'vesselNextport'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
        #将以上数据合并
        etaTemp = etaTemp.merge(groupdata, on=['loadingOrder', 'vesselNextport'], how='left')
        dataset = pd.concat([dataset, etaTemp])

        if count > 1024 * 1024 * 2 :
            break



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)