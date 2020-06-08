import pandas as pd
import math
import time


def editDistance(word1, word2) : #used to find the most similar string
    m, n = len(word1), len(word2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1) :
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(m):
        for j in range(n):
            dp[i][j]=min(dp[i][j]+(0 if word1[i] == word2[j] else 1),
                         dp[i][j + 1]+1,
                         dp[i + 1][j]+1,
                         )
    return dp[m][n]


def getDistance(lo1, la1, lo2, la2) : #calculate distance by coordinates
    lng1, lat1, lng2, lat2 = map(math. radians, [float(lo1), float(la1), float(lo2), float(la2)])
    return 6371 * math.acos(math.sin(lng1) * math.sin(lng2) + math.cos(lng1) * math.cos(lng2) * math.cos(lat1 - lat2))


#load in port statistics
train_data = pd.read_csv('../port.csv')
port_loc = {} #portName: location(coordinates)

for _, i in train_data.iterrows() :
    port_loc.update({i['TRANS_NODE_NAME'] : [i['LATITUDE'], i['LONGITUDE']]})

gps_record_reader = pd.read_csv('../head10m - 复件.csv', header=None, low_memory=False, chunksize=1024 * 1024) # 重新设置Reader指针
start_time = time.time()


#get ETAs
port_mapping = {} #order: current nextPort
eta_mapping = {} #order+nextPort: timestamp
port_standardizing = {} #informal port name: most similar formal port name #NOT PRECISE

count = 0
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
    if count > 1024 * 1024 * 2 :
        break

print(eta_mapping)

end_time = time.time()
print(end_time - start_time)