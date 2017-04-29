# -*-coding:utf8-*-
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#
#
'''
@version: ??
@author: xiholix
@contact: x123872842@163.com
@software: PyCharm
@file: buildFeature.py
@time: 17-4-26 上午10:14
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from evaluation import RMSD


def get_train_dataframe(_path):
    dfTrain = pd.read_csv(_path)
    return dfTrain


def operate_miss_data(_dfData):
    nullNum = _dfData.isnull().sum().sort_values(ascending=False)

    columnsName = nullNum[nullNum>1].index
    dfData = _dfData.drop(columnsName, 1)
    dfData = dfData.drop(dfData.loc[dfData['Electrical'].isnull()].index)
    return dfData


def get_most_related_variable(_dfData, _k):
    corrmat = _dfData.corr()
    cols = corrmat.nlargest(_k, 'SalePrice')['SalePrice']
    # print(cols)
    return cols.index

def remove_linearity(_dfData):
    pass


def get_train_and_test_data():
    df_train = get_train_dataframe('data/train.csv')
    # test_RMSD()
    # print(df_train.info())
    df_train = operate_miss_data(df_train)
    mostVariabel = get_most_related_variable(df_train, 20)
    print(mostVariabel)
    trainData = df_train[mostVariabel]
    explore_linearity(trainData, mostVariabel)
    trainData = trainData.values
    labels = df_train['SalePrice'].values
    # print(trainData.info())
    kFold = KFold(trainData.shape[0], 5)
    # print(trainData.head(2))
    rfm = RandomForestRegressor()
    print('start training')

    evaluate_model_param(rfm, trainData, labels)

def evaluate_model_param(_model, _datas, _labels):
    kFold = KFold(_datas.shape[0], 5)
    evaluation = 0

    for trainIndices, testIndices in kFold:
        _model.fit(_datas[trainIndices, :], _labels[trainIndices])
        pre = _model.predict(_datas[testIndices, :])
        t = RMSD(np.log(_labels[testIndices]), np.log(pre) )
        evaluation += t

    evaluation /= 5
    print(evaluation)
    return evaluation


def score_function(_model, _datas, _labels):
    pre = _model.predict(_datas)
    rmsd = RMSD(np.log(_labels), np.log(pre) )
    return -rmsd


def test_grid_search():
    from sklearn.model_selection import GridSearchCV

    df_train = get_train_dataframe('data/train.csv')
    # test_RMSD()
    # print(df_train.info())
    df_train = operate_miss_data(df_train)
    mostVariabel = get_most_related_variable(df_train, 20)
    trainData = df_train[mostVariabel]
    trainData = trainData.drop(['TotRmsAbvGrd', 'GarageArea', '1stFlrSF'], 1)
    trainData = trainData.values
    labels = df_train['SalePrice'].values

    rfm = RandomForestRegressor()
    param_grid = {
        'n_estimators':(10, 20 ,30),
        'min_samples_split':(2,4,6, 8),
        # 'criterion':('mse', 'mae')
    }#最佳参数为n_estimator=20, min_samples_split=2, mse
    clf = GridSearchCV(rfm, cv=3, scoring=score_function, param_grid=param_grid)
    clf.fit(trainData, labels)
    t = sorted(clf.cv_results_.keys() )
    result = pd.DataFrame(clf.cv_results_)
    print(clf.cv_results_['params'])
    result['mean_test_score'].plot.line()
    plt.show()
    print(clf.best_score_)
    print(clf.best_estimator_)


def explore_linearity(_data, _names):
    cm = np.corrcoef(_data.values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size':10}, yticklabels=_names, xticklabels=_names)
    # plt.show()
    linearThanHalf = cm>0.7
    print(linearThanHalf)
    indices = get_true_index(linearThanHalf)

    for i, j in indices:
        print(_names[i]+'\t'+_names[j])


def get_true_index(_data):
    indices = []
    for i in xrange(_data.shape[0]):
        for j in xrange(_data.shape[1]):
            if _data[i, j] == True:
                if i<j:
                   indices.append((i,j))
    print(indices)
    return indices

if __name__ == "__main__":
    # get_train_and_test_data()
    test_grid_search()