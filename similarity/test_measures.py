from calendar import leapdays
from email.errors import HeaderMissingRequiredValue
import os
import sys

sys.path.append(os.path.dirname(__file__))

import pickle

import numpy as np
from tqdm import tqdm

from sim_measures import *
import pandas as pd

def precentage_analysis_sim_matrix(matrix,precision=10):
    """
    按照百分比计算统计相似度矩阵的分布
    """
    mid = np.max(matrix)-np.min(matrix)
    matrix = matrix-np.min(matrix)
    matrix = (matrix/mid*precision).astype(np.int32)
    m, n = matrix.shape
    totol = [0]*(precision+1)
    for i in tqdm(range(m), ncols=100):
        for j in range(n):
            totol[matrix[i, j]] += 1
    print(np.divide(totol, m*n))


def precentage_analysis_sim_method(sim_method,data_size=10000,min_items=10,max_items=200,min_corr=1,max_corr=10,step=10):
    """
        使用fake数据对相似度方法进行分析
        分析内容有均值和标准差
    """
    sim_means=dict()
    sim_stds=dict()
    for n_items in range(min_items,max_items+1,step):
        fake_datas=np.random.randint(min_corr,max_corr+1,size=(data_size,n_items))
        sim=sim_method(fake_datas,data_size,dtype=tf.float32)
        sim_means[n_items]=np.mean(sim)
        sim_stds[n_items]=np.std(sim)
    print(sim_means)
    print(sim_stds)
    return sim_means,sim_stds

def analysis_all_measures():
    res=dict()
    # methods = [
    #     SMD,  #0.99？
    #     COS_measure,
    #     Jacc_measure,
    #     ADF,
    #     MSD,
    #     PSS,  #为啥会大于1
    #     JMSD, # 得益于Jacc
    #     Mjacc,
    #     URP_measure,
    #     NHSM,
    #     SMD,
    #     HSMD,
    # ]

    methods = [
        PSS,
        PNCR,
        OS,
        NHSM,
        HSMD
    ]
    methods_name = []
    length  = []
    means   = []
    stds    = []
    for one in methods:
        sim_means, sim_stds = precentage_analysis_sim_method(one)
        length.extend(list(sim_means.keys()))
        means.extend(list(sim_means.values()))
        stds.extend(list(sim_stds.values()))
        methods_name.extend([one.__name__] * len(sim_means))
    df = pd.DataFrame({"method": methods_name,
                      "lenght": length,
                       "mean": means,
                       "std": stds
                       })
    print(df)
    df.to_csv("res/res4.csv",index=False)

if __name__=="__main__":
    analysis_all_measures()
