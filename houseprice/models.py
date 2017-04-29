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
@file: models.py
@time: 17-4-26 上午10:44
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from evaluation import RMSD

def model_persistence(_model, _path):
    from sklearn.externals import joblib
    joblib.dump(_model, _path)


def model_evaluation(_path, _dataX, _dataY=None):
    from sklearn.externals import joblib
    clf = joblib.load(_path)

    predictions = clf.predict(_dataX)
    evaluatedValue = RMSD(_dataY, predictions)

    print(evaluatedValue)
    return evaluatedValue


def test_RMSD():
    a = [1,1,1,1]
    b = [1,2,1,2]
    print(RMSD(a,b))


if __name__ == "__main__":
    # get_train_dataframe('data/train.csv')
    test_RMSD()