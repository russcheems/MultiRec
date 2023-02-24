"""
    用户相似度计算方法
    设计范例：
    similarity_measure(matrix,n_users,batch_size=1024*8,dtype=tf.float16)
        # 存放结果的矩阵
        sim=np.zeros((n_users,n_users))
        
        # 将数据矩阵转化为Tensor，放入GPU中
        matrix=tf.convert_to_tensor(matrix,dtype=dtype)

        # 两两计算相似度，这里生成两两配对组合，用于batch计算
        pairs=list(combinations(range(n_users),2))
        totol=int(n_users*(n_users-1)/2)

        # batch计算
        for idx in tqdm(range(0,totol,batch_size),ncols=100):
            # 读取批处理数据
            ## 获取索引
            batch_pairs=pairs[idx:idx+batch_size]
            batch_u=[i[0] for i in batch_pairs]
            batch_v=[i[1] for i in batch_pairs]

            ## 取得GPU中的数据
            data_u=tf.nn.embedding_lookup(matrix,batch_u)
            data_v=tf.nn.embedding_lookup(matrix,batch_v)

            ## 相似度计算公式
            sum= similiart_measures process....

            ## 从GPU中取回结果
            sum = sum.numpy()
            for s, (u, v) in zip(sum, batch_pairs):
                sim[u, v] = s
        return sim



"""
import math
from itertools import combinations
from time import time

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.math import divide_no_nan
from tqdm import tqdm

WRAPPER_LEVER=0 

def wrapper(func):
    """
    一个带深度指示的装饰器，用于进行一些运行测试和说明
    """
    
    def inner(*args,**kwargs):
        
        global WRAPPER_LEVER

        start =time()
        print("\t"*WRAPPER_LEVER+"strat to run {}...".format(func.__name__))
        WRAPPER_LEVER+=1

        res=func(*args,**kwargs)

        end=time()
        WRAPPER_LEVER-=1
        print("\t"*WRAPPER_LEVER+"use {} sec".format(end-start))
        print("")
        return res
    inner.__name__=func.__name__
    return inner


@wrapper
def COS_measure(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    return cosine_similarity(matrix)

@wrapper
def PCC_measure(matirx,n_users,batch_size=1024*8,dtype=tf.float16):
    return pearsonr(matirx)



@wrapper
def Jacc_measure(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    # 
    sim=np.zeros((n_users,n_users))
    matrix=tf.convert_to_tensor(matrix,dtype=dtype)
    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)

    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        # 相似度计算公式
        max_w = tf.maximum(data_u, data_v)
        min_w = tf.minimum(data_u, data_v)
        sum = divide_no_nan(tf.math.count_nonzero(min_w, axis=1, dtype=dtype),
                            tf.math.count_nonzero(max_w, axis=1, dtype=dtype))
        sum = sum.numpy()
        for s, (u, v) in zip(sum, batch_pairs):
            sim[u, v] = s
    return sim


@wrapper
def ADF(matrix,n_users,batch_size=1024*8,dtype=tf.float16):

    sim=np.zeros((n_users,n_users))
    # 这个太大就放不下
    temp_one=tf.ones(shape=(matrix.shape[1],1),dtype=dtype) # 求和用的
    matrix=tf.convert_to_tensor(matrix,dtype=dtype)

    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)

    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        # 相似度计算公式
        max_w = tf.maximum(data_u, data_v)
        m1 = -tf.math.abs(data_u-data_v)
        m2 = tf.math.exp(tf.math.divide_no_nan(m1, max_w))
        m2 = divide_no_nan(m2*max_w, max_w)  # 去掉exp(0)
        sum = tf.reshape(tf.matmul(m2, temp_one), -1)
        sum = tf.math.divide_no_nan(sum, tf.cast(
            tf.math.count_nonzero(max_w, axis=1), dtype=dtype))
        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s
    return sim

@wrapper
def MSD(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    sim=np.zeros((n_users,n_users))
    max_rating=np.max(matrix)
    n_items=matrix.shape[1]
    matrix=tf.convert_to_tensor(matrix,dtype=dtype)

    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        # 相似度计算公式
        
        sub=(data_u-data_v)/max_rating
        mid=tf.pow(sub,2)
        sum = tf.reduce_sum(mid,axis=1)/n_items
        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s        
    return 1-sim

@wrapper
def PIP():
    pass

@wrapper
def PSS(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    sim=np.zeros((n_users,n_users))
    # 这个太大就放不下
    temp_one=tf.ones(shape=(matrix.shape[1],1),dtype=dtype) # 求和用的

    rm=np.sum(matrix)/np.sum(matrix!=0)
    rj=np.sum(matrix,axis=0)/np.sum(matrix!=0,axis=0)
    rj=tf.convert_to_tensor(rj,dtype=dtype)

    matrix=tf.convert_to_tensor(matrix,dtype=dtype)

    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        max_w = tf.maximum(data_u, data_v)
        mid = tf.exp(-tf.abs(data_u-data_v))
        prox = mid/(1+mid)

        sig = 1/(1+tf.exp(-tf.abs(
            data_u-rm)*tf.abs(data_v-rm)))

        mid = tf.exp(-tf.abs((data_u+data_v)/2-rj))
        sin = mid/(1+mid) 
        
        scores=prox *sig *sin
        scores=divide_no_nan(scores * max_w,max_w) # 去掉exp(0)

        sum = tf.reshape(tf.matmul(scores, temp_one), -1)
        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s                
    return sim
    
@wrapper
def JMSD(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    jacc_sim=Jacc_measure(matrix,n_users,batch_size=1024*8,dtype=tf.float16)
    msd_sim=MSD(matrix,n_users,batch_size=1024*8,dtype=tf.float16)
    return msd_sim*jacc_sim


@wrapper
def Mjacc(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    sim=np.zeros((n_users,n_users))

    matrix=tf.convert_to_tensor(matrix,dtype=dtype)

    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        r=tf.math.count_nonzero(tf.maximum(data_u,data_v),axis=1,dtype=dtype)

        g1=tf.math.count_nonzero(data_u,axis=1,dtype=dtype)
        g2=tf.math.count_nonzero(data_v,axis=1,dtype=dtype)
        g=g1*g2
        sum=divide_no_nan(r,g) 
        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s       
    return sim


@wrapper
def URP_measure(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    # 有更高效的版本
    sim=np.zeros((n_users,n_users))
    non_matrix=np.where(matrix>0,matrix,np.nan)
    mean_u=np.nanmean(non_matrix,axis=1)
    std_u=np.nanstd(non_matrix,axis=1)
    #
    mean_u=tf.convert_to_tensor(mean_u,dtype=dtype)
    std_u=tf.convert_to_tensor(std_u,dtype=dtype)
    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u_mean=tf.nn.embedding_lookup(mean_u,batch_u)
        data_u_std=tf.nn.embedding_lookup(std_u,batch_u)
        data_v_mean=tf.nn.embedding_lookup(mean_u,batch_v)
        data_v_std=tf.nn.embedding_lookup(std_u,batch_v)
        sum=1/(1 +tf.math.exp(
                               -tf.abs((data_u_mean-data_v_mean) *
                                       (data_u_std-data_v_std))
                           ))
        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s       
    return 1-sim


@wrapper
def NHSM(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    pss_sim=PSS(matrix,n_users,batch_size,dtype)
    mjcacc_sim=Mjacc(matrix,n_users,batch_size,dtype)
    jpss=pss_sim*mjcacc_sim
    urp_sim=URP_measure(matrix,n_users,batch_size,dtype)
    sim=jpss*urp_sim
    return sim
@wrapper
def SMD(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    sim=np.zeros((n_users,n_users))
    # 计算一些统计信息
    count_u=np.count_nonzero(matrix,axis=1)

    # 这个太大就放不下
    temp_one=tf.ones(shape=(matrix.shape[1],1),dtype=dtype) # 求和用的
    n_items=matrix.shape[1]
    matrix=tf.convert_to_tensor(matrix,dtype=dtype)
    count_u=tf.convert_to_tensor(count_u,dtype=dtype)
        
    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        # 等价于求交集
        N12 = tf.math.count_nonzero(tf.minimum(
            data_u, data_v), axis=1, dtype=dtype)
        # 等价于求补集
        F = tf.math.count_nonzero(tf.maximum(
            data_u, data_v), axis=1, dtype=dtype)-N12

        
        N1=tf.nn.embedding_lookup(count_u,batch_u)
        N2=tf.nn.embedding_lookup(count_u,batch_v)
        
        sum=(
            1-F/n_items+divide_no_nan((2*N12),(N1+N2))
            )/2
        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s       
    return sim

@wrapper
def HSMD(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    sim=np.zeros((n_users,n_users))
    # 计算一些统计信息
    sum_u=np.sum(matrix,axis=1)
    
    # 这个太大就放不下
    temp_one=tf.ones(shape=(matrix.shape[1],1),dtype=dtype) # 求和用的
    n_items=matrix.shape[1]
    matrix=tf.convert_to_tensor(matrix,dtype=dtype)
    sum_u=tf.convert_to_tensor(sum_u,dtype=dtype)
        
    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        r=tf.maximum(data_u,data_v)
        r1=tf.reshape(tf.matmul(r-data_v, temp_one), -1)
        r2=tf.reshape(tf.matmul(r-data_u, temp_one), -1)
        
        g1=tf.nn.embedding_lookup(sum_u,batch_u)
        g2=tf.nn.embedding_lookup(sum_u,batch_v)
        g=g1*g2
        sum=divide_no_nan((r1*r2+1),g)
        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s       
    return 1-sim

@wrapper
def PNCR(matrix,n_users,batch_size=1024*8,dtype=tf.float16):
    sim=np.zeros((n_users,n_users))

    n_items=matrix.shape[1]

    matrix=tf.convert_to_tensor(matrix,dtype=dtype)
    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        r = tf.math.count_nonzero(tf.minimum(data_u, data_v), axis=1)
        sum = tf.math.exp(
            -(n_items-r)/n_items
        )   # 理论上不存在r==n_items

        sum = sum.numpy()
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s       
    return sim

@wrapper
def OS(matrix, n_users, batch_size=1024*8, dtype=tf.float16):
    pncr_sim = PNCR(matrix, n_users, batch_size, dtype)
    adf_sim = ADF(matrix, n_users, batch_size, dtype)
    return pncr_sim * adf_sim


def BS(matrix, n_users, batch_size=1024*8, dtype=tf.float16):
    sim=np.zeros((n_users,n_users))
    n_items=matrix.shape[1]

    matrix=np.where(matrix>0,matrix,np.nan)
    scale=int(np.max(matrix))
    p=[0]*(scale+1)
    alpha=[0]*(scale+1)
    n2=scale*scale
    
    num_ratings=len(np.nonzero(matrix)[1])

    # p[i]表示分数i的概率 i =[1,scale]
    for i in range(1,scale+1):
        p[i]=len(np.where(matrix==i)[1])/num_ratings
    # alpha[j]表示间距j的概率 j=[0,scale-1],为了保持与i相同，这里进行了映射
    for j in range(1,scale+1):
        alpha[1]+=n2*p[j]*p[j]
    for i in range(2,scale+1):
        for j in range(1,scale-i+1+1):
            alpha[i]+=n2*p[j]*p[j+i-1]
        alpha[i]=2*alpha[i]
    alpha[0]=np.sum(alpha)
    E0=np.array(alpha)/alpha[0]

    std_items=np.nanstd(matrix,axis=0)
    std_=np.nanstd(matrix)
    c=1/std_
    up_limit=2*c*std_items
    for u in tqdm(range(n_users),ncols=100):
        for v in range(u+1,n_users):
            epcl_0_is=[]
            # 计算间距
            # 为与原论文符号一致，这里i表示间距
            duv=0
            sum_di=0
            for i in range(scale):
                epcl_0_i=0
                epcl_k_is=[]
                for k in range(n_items):
                    dk=np.abs(matrix[u,k]-matrix[v,k])
                    if dk>up_limit[k]:
                        eik=-1
                    else:
                        eik=1-dk/c*std_items[k]
                    epcl_k_i =1 if dk==i else 0
                    epcl_k_is.append(epcl_k_i)
                    epcl_0_i+=eik*epcl_k_i
                epcl_0_is.append(epcl_0_i)

                epcl_0=np.sum(epcl_0_i*np.array(epcl_k_is))
                E1=(alpha[i]+epcl_0_i)/(alpha[0]+epcl_0)
                
                wi=E1-E0[i]
                duv+=wi*i
                sum_di+=np.abs(wi)
            duv=duv/sum_di

            suv1=1-duv/scale
            suv2=1
            # 为与论文保持一致，此处j表示间距 
            for j in range(scale):
                suv2*=pow(E0[i],epcl_0_is[i])
            suv=max(suv1-suv2,0)
            sim[u,v]=suv
            sim[v,u]=suv
    return sim

def BS2(matrix, n_users, batch_size=1024*8, dtype=tf.float16):
    sim=np.zeros((n_users,n_users))

    # 预处理
    matrix=np.where(matrix>0,matrix,np.nan)
    scale=int(np.max(matrix))
    p=[0]*(scale+1)
    alpha=[0]*(scale+1)
    n2=scale*scale
    num_ratings=len(np.nonzero(matrix)[1])
    n_items=matrix.shape[1]
    # 获取先验信息
    for i in range(1,scale+1):
        # p[i]=len(np.where(matrix==i)[1])/num_ratings
        p[i]=0.2
    for j in range(1,scale+1):
        alpha[1]+=n2*p[j]*p[j]
    for i in range(2,scale+1):
        for j in range(1,scale-i+1+1):
            alpha[i]+=n2*p[j]*p[j+i-1]
        alpha[i]=2*alpha[i]
    alpha[0]=np.sum(alpha)
    

    std_items=np.nanstd(matrix,axis=0)
    std_=np.nanstd(matrix)
    c=1/std_

    up_limit=c*std_items

    e=np.zeros((n_items,scale))
    for k in range(n_items):
        for i in range(scale):
            e[k,i]=1-np.min([i/std_items[k],2])
    # 放入显存 
    matrix=tf.convert_to_tensor(matrix,dtype=dtype)
    up_limit=tf.convert_to_tensor(up_limit,dtype=dtype)
    pairs=list(combinations(range(n_users),2))
    totol=int(n_users*(n_users-1)/2)
    epcl_emb=tf.convert_to_tensor(np.eye(scale),dtype=dtype)
    alpha2=tf.convert_to_tensor(alpha[1:],dtype=dtype)
    E0=tf.convert_to_tensor(np.array(alpha[1:])/alpha[0],dtype=dtype)
    scale_list=tf.reshape(tf.convert_to_tensor(list(range(scale)),dtype=dtype),[scale,1])
    e_emb=tf.convert_to_tensor(e,dtype=dtype)
    # 开始分批计算用户相似度
    for idx in tqdm(range(0,totol,batch_size),ncols=100):
        # 批处理数据
        batch_pairs=pairs[idx:idx+batch_size]
        batch_u=[i[0] for i in batch_pairs]
        batch_v=[i[1] for i in batch_pairs]
        data_u=tf.nn.embedding_lookup(matrix,batch_u)
        data_v=tf.nn.embedding_lookup(matrix,batch_v)

        d1=tf.abs(data_u-data_v)
        d2=tf.cast(d1,tf.int32)
        epcl=tf.nn.embedding_lookup(epcl_emb,d2)
        # e=tf.reshape(1-tf.minimum(divide_no_nan(d1,up_limit),2),[batch_size,-1,1])
        e=tf.nn.embedding_lookup(e_emb,d2)
        epcl_0_i=tf.reduce_sum(epcl*e,axis=1)
        
        epcl_0=tf.reduce_sum(epcl_0_i,axis=1,keepdims=True)
        epcl_0=tf.tile(epcl_0,[1,scale])
        E1=(epcl_0_i+alpha2)/(epcl_0+alpha[0])
        wi=E1-E0
        duv1=tf.reshape(tf.matmul(wi,scale_list),-1)
        duv2=tf.reduce_sum(tf.abs(wi),axis=1)
        duv=duv1/duv2
        suv1=1-duv/(scale-1)
        suv2=tf.exp(
            tf.reduce_sum( tf.math.log(alpha2/alpha[0])* epcl_0_i ,axis=1
            ))
        
        suv=tf.maximum(suv1-suv2,0)
        sum = suv.numpy()
        
        for s,(u,v) in zip(sum,batch_pairs):
            sim[u,v]=s
            sim[v,u]=s       

        # 还是有问题。




def test():
    matrix=np.random.randint(1,6,size=(10000,100))

    # SMD(matrix,10000)
    # COS_measure(matrix,10000)
    # Jacc_measure(matrix,10000)        # use 152.88838815689087 sec 
    # ADF(matrix,10000)                 # use 162.94708609580994 sec
    # MSD(matrix,10000)                 # use 176.28174424171448 sec
    # PSS(matrix,10000)                 # use 162.89934062957764 sec 
    # JMSD(matrix,10000)                # use 327.1638400554657 sec
    # res=Mjacc(matrix,10000)               # use 162.25598764419556 sec
    # res=URP_measure(matrix,10000)           # use 105.3256323337555 sec  
    # res=NHSM(matrix,10000)                  # use 360 sec
    # SMD(matrix,10000)
    # HSMD(matrix,10000)
    # PSS(matrix,10000)
    # OS(matrix,10000)
    BS2(matrix,10000,dtype=tf.float32)
if __name__=="__main__":
    test()
