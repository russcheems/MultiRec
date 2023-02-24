from copyreg import pickle
import os
import sys
from cmath import exp, sqrt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tokenize import group

import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle

def get_user_ij_same_items(pair,range_r,user_his):
    i,j=pair
    total=0
    for r in range(1, range_r+1):
        his_i = user_his[i][r]
        his_j = user_his[j][r]
        total+=len(his_i & his_j)
    return i,j,total

class User_adj_matrix():
    def __init__(self, n_users, range_r, data) -> None:
        self.n_users = n_users
        self.range_r = range_r
        self.user_adj_matrix = np.zeros((n_users, n_users))

        self.build_user_his(data)
        # self.build_user_adj_matrix()

    def build_user_adj_matrix(self):
        """ 非零值表示了两个用户之间具有相同评分物品的数量"""

        for i in tqdm(range(self.n_users), ncols=100):
            # for j in tqdm(range(i+1, self.n_users), ncols=100):
            for j in range(i+1,self.n_users):
                for r in range(1, self.range_r+1):
                    his_i = self.user_his[i][r]
                    his_j = self.user_his[j][r]
                    self.user_adj_matrix[i, j] += len(his_i & his_j)
                self.user_adj_matrix[j,i] = self.user_adj_matrix[i, j]

        # pairs=[[i,j] for i in range(self.n_users) for j in range(i+1,self.n_users)]

        # with ProcessPoolExecutor(max_workers=10) as exectuor:
        #     res = list(
        #         tqdm(
        #             exectuor.map(
        #                 get_user_ij_same_items,
        #                 pairs, [self.range_r] *
        #                 len(pairs), [self.user_his]*len(pairs),
        #                 chunksize=100
        #             ), ncols=100)
        #     )
        # for i,j,n in res:
        #     self.user_adj_matrix[i,j]=n
        #     self.user_adj_matrix[j,i]=n
        return self.user_adj_matrix

    def build_user_adj_matrix2(self):
        self.user_adj_matrix2=np.dot(self.rating_matrix,self.rating_matrix.T)
        return self.user_adj_matrix2
    
    def build_user_his(self, data):
        self.user_his = defaultdict(dict)
        ids = data[:, 1]
        self.n_items = np.max(ids)+1


        self.rating_matrix = np.zeros((self.n_users, self.n_items))
        for u in range(self.n_users):
            for r in range(1, self.range_r+1):
                self.user_his[u][r] = set()

        for [u, i, _, r] in tqdm(data, ncols=100):
            self.user_his[u][r].add(i)
            self.rating_matrix[u, i] = 1

    def build_user_cluster(self):
        pass

    def show(self):
        print(np.count_nonzero(self.user_adj_matrix))

        row=np.sum(self.user_adj_matrix,axis=1)
        print(np.count_nonzero(row))
        pass
    

    def build_eighs(self,subgraph=100):
        # 

        # 
        d_row=np.zeros((self.n_users,self.n_users))
        sum_row=np.sum(self.rating_matrix,axis=1)
        for i in range(self.n_users):
            d_row[i,i]=sum_row[i]
        # kene
        d_col=np.zeros((self.n_items,self.n_items))
        sum_col=np.sum(self.rating_matrix,axis=0)
        for j in range(self.n_items):
            d_col[j,j]=sum_col[j]


        d1=la.inv(la.sqrtm(d_row))
        d2=la.inv(la.sqrtm(d_col))

        S=np.dot(
            np.dot(d1,self.rating_matrix),d2
        )
        St=np.copy(S.T)
        #
        In=np.identity(self.n_users)
        Im=np.identity(self.n_items)
        
        #Get M
        m1=np.hstack((In,-S))
        m2=np.hstack((-St,Im))
        m=np.vstack((m1,m2))

        # 返回的特征值、特征向量是有序的
        evals,evecs=la.eigh(m)
        
        group_info=evecs[:,0:subgraph]

        group_info=np.argmax(group_info,axis=1)
        print(len(group_info))
        print(group_info[:10])
        return group_info
        pass

class UserSimMatrix:
    def __init__(self, n_users, n_items,range_r, data,save_path) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.range_r = range_r
        self.user_adj_matrix = np.zeros((n_users, n_users))
        self.save_path=os.path.join(save_path[:-5],"group_info.pkl")
        self.data=data
        # self.get_user_group()

    def build_user_his(self):
        data=self.data
        # 使用字典来记录用户历史记录
        self.user_his = defaultdict(defaultdict)
        # for u in range(self.n_users):
        #     for r in range(1, self.range_r+1):
        #         self.user_his[u][r] = set()        

        for [u, i, _, r] in tqdm(data, ncols=100):
            self.user_his[u][i]=r
    def get_os_sim(self):
        def get_adf(i,j,items):
            ADF=[]
            for item in items:

                r_i=self.user_his[i][item]
                r_j=self.user_his[j][item]
                ADF.append(
                    np.exp(-abs(r_i-r_j)/max(r_i,r_j))
                )
            return sum(ADF)
        for user_i in tqdm(self.user_his.keys(),ncols=100):
            i_items=self.user_his[user_i].keys()
            for user_j in self.user_his.keys():
                j_items=self.user_his[user_j].keys()
                and_items=list(set(i_items)& set(j_items))
                
                andL=len(and_items )
                if andL==0:
                    continue
                PNCR=np.exp(
                    -(self.n_items-andL)/self.n_items
                )
                ADF=get_adf(user_i,user_j,and_items)
                ADF/=andL
                self.user_adj_matrix[user_i,user_j]=PNCR*ADF

    def get_user_group(self,n=0):
        self.build_user_his()       
        self.get_os_sim()
        if n==0:
            n=int(np.sqrt(self.n_users))
        cluster=KMeans(n).fit(self.user_adj_matrix,self.n_users)
        with open(self.save_path,"wb") as f:
            pickle.dump(cluster.labels_,f)
        return cluster.labels_


"""
7089368

8502174

1131458

3932160

131392/2,042,041

"""
