# ML2020-ETA
Code for the contest “华为云大数据挑战赛”——[船运到达时间预测](https://competition.huaweicloud.com/information/1000037843/circumstance?track=107)

## Working Directory
```
├─data
│  ├─GPS.csv
│  ├─loadingOrderEvent.csv
│  ├─testData.csv
│  └─preprocessed
│      ├─backup
│      ├─order_cache
│      └─trace_specific_dataset
├─experiment
│  ├─baseline.ipynb
│  ├─data_selection.ipynb
│  └─directpredict.ipynb
├─models
├─train.py
└─predict.py
```
## Requirements
In consistent with HUAWEI ModelArts Platform's configuration
- python 3.7
- Tensorflow 2.0.0-beta1
- xgboost 1.1.0
- sklearn 0.23.1
- lightgbm 2.3.0

## Team Members
- Yifan Mao
- Xin Wei
- Zian Su

## References
- Wang et al., Didi AI Labs, [Learning to Estimate the Travel Time](https://dl.acm.org/doi/10.1145/3219819.3219900)
- Saet et al., [Predicting Estimated Time of Arrival for Commercial Flights](https://dl.acm.org/doi/10.1145/3219819.3219874)