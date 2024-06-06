"""
基于自适应分组的参数共享的模型
"""

import os
import sys
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as K_layers
from tensorflow.keras.layers import (Conv1D, Dense, Embedding, Input,Layer,
                                     concatenate)
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.python.keras import Model
from untils.gpu_config import *
from Routing import build_shared_layer
from Ind_reg import IND_reg,IndReg
TEST_FLAG=1



def wrapper(func):
    """
    一个装饰器，用于进行一些运行测试和说明
    """
    def inner(*args,**kwargs):
        res=func(*args,**kwargs)
        # print("[!] build module {} OK".format(sys._getframe().f_code.co_name))
        if TEST_FLAG:
            print("[!] build module {} OK".format(func.__name__),end="\t")
            try:
                print(res.shape)
                # if res and res.hasattr("shape"):
                #     print(res.shape)
                # else:
                #     print()
            except:
                print()
        return res
    return inner


class MuADSE():
    """
    基于共享的模型: 对所有用户和物品的向量进行了共享。
    共享信息包括：review,interaction,rating
    要求分组信息是提前计算好的。
    """
    name="DoubleEmbeddingModel2"

    def __init__(self,flags,data_loader,group_info,expert):
        print("this is group model")
        self.num_class    = flags.num_class
        self.emb_size     = flags.emb_size
        self.batch_size   = flags.batch_size
        self.epochs       = flags.epoch
        self.share_mode   = flags.mode # 用来控制偏好的影响
        self.doc_layers   = flags.doc_layers
        self.doc_keep     = 1-flags.doc_dropout
        self.routing_mode = flags.routing_mode
        self.reg          = IndReg(flags.routing_lr) if flags.use_reg else None
        # 分组信息

        self.group_info=group_info
        self.num_user_group=group_info["num_user_group"]
        self.num_item_group=group_info["num_user_group"]

        self.vocab_size = data_loader.vocab_size
        self.num_user   = data_loader.num_user
        self.num_item   = data_loader.num_item
        self.t_num      = data_loader.t_num
        self.maxlen     = data_loader.maxlen
        self.data_size  = data_loader.data_size+1
        self.vec_texts  = data_loader.vec_texts # 151255, 60
        self.expert1 = expert
        # self.expert2 = expert2

        self.ckpt_dir=os.path.join(flags.ckpt_dir,self.name) 
    
    def init_input(self):

        self.utext     = Input(name="utext", dtype=tf.int32, shape=self.t_num)
        self.itext     = Input(name="itext", dtype=tf.int32, shape=self.t_num)
        self.keep_prob = 0.5

        # 用户信息
        self.user_group_rating = Input(
            name="user_group_rating", dtype=tf.int32, shape=())
        self.user_group_interc = Input(
            name="user_group_interc", dtype=tf.int32, shape=())
        self.user_group_review = Input(
            name="user_group_review", dtype=tf.int32, shape=())

        # 物品信息
        self.item_group_rating = Input(
            name="item_group_rating", dtype=tf.int32, shape=())
        self.item_group_interc = Input(
            name="item_group_interc", dtype=tf.int32, shape=())
        self.item_group_review = Input(
            name="item_group_review", dtype=tf.int32, shape=())
        

    def get_model(self,summary=False):
        """
        写成显示的数据流会更好分析
        """
        
        # 获取输入
        self.init_input()

        # 构建v,d的相关结构

        pred_fc,u_rating_latent_fc,i_rating_latent_fc=self.expert1.build(self.user_group_rating,
                                   self.item_group_rating,
                                   self.user_group_review,
                                   self.item_group_review,
                                   self.user_group_interc,
                                   self.item_group_interc,self.utext,self.itext)


        rating_latent_fc,r_w_fc=self.predict_by_d(pred_fc,u_rating_latent_fc,i_rating_latent_fc)

        model = Model(inputs=[self.user_group_rating,
                              self.user_group_interc,
                              self.user_group_review,
                              self.item_group_rating,
                              self.item_group_interc,
                              self.item_group_review,
                              self.utext, self.itext],
                      outputs=[r_w_fc])

        self.model = model

        if summary:
            model.summary()



    




    def predict_by_d(self,w,u_latent,i_latent):
        """
        return P(r|w)
        """
        self.rd_layers = self.expert1.get_rd()
        layer=concatenate([w,u_latent,i_latent])
        for one_layer in self.rd_layers[:-1]:
            layer=one_layer(layer)
        res=self.rd_layers[-1](layer)    
        return layer,res
        # for one_dim in layers:
        # for i in [3,2,1]:
        #     one_dim=i*self.emb_size
        #     layer=Dense(one_dim,activation="relu",name="Pr_w_{}".format(one_dim))(layer)
        
        # Pred = Dense(1,bias_initializer=tf.constant_initializer(3),name="P_r_w")(layer)
        # return Pred

    def predict_gmf(self,u_latent,i_latent,layers):
        layer=concatenate([u_latent,i_latent])
        for i,one in enumerate(layers):
            layer=Dense(one,activation="relu",name="GMF_{}_{}".format(i,one))(layer)
        layer=Dense(1,activation="relu",name="P_r_ui")(layer)
        return layer
    
    def final_pred(self,w,ulatent,ilatent,layers):
        """
        P(r|u,i)=P(r|w) *P(w|u,i)
        """
        layer = concatenate([w,ulatent,ilatent])
        for i,one in enumerate(layers):
            layer=Dense(one,activation="relu",name="final_pred_{}_{}".format(i,one))(layer)

        pred = Dense(1, name="P_r_uiw")(layer)

        return pred

    def train(self,data_loader):
        
        # 存储路径
        checkpoint_dir=self.ckpt_dir
        checkpoint_path=os.path.join(checkpoint_dir,"{}.h5".format(self.name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 通过回调进行保存
        cp_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="mean_absolute_error",
            save_best_only=True, verbose=1, save_weights_only=True, period=1)
        # 配置优化器
        # self.model.compile(optimizer="adam",loss="mean_squared_error",
        #     metrics = [RootMeanSquaredError(), "mean_absolute_error"])
        self.model.compile(optimizer="adam",loss="mean_squared_error",
            metrics = [RootMeanSquaredError(), "mean_absolute_error"])

        # 读取训练数据
        (   user_group_ratings,
            user_group_intercs,
            user_group_reviews,
            item_group_ratings,
            item_group_intercs,
            item_group_reviews,
            utext, itext, label) = data_loader.all_train_data_with_group_info(self.group_info)
        train_data = {
            "user_group_rating": user_group_ratings,
            "user_group_interc": user_group_intercs,
            "user_group_review": user_group_reviews,
            "item_group_rating": item_group_ratings,
            "item_group_interc": item_group_intercs,
            "item_group_review": item_group_reviews,
            "utext": utext,
            "itext": itext
        }

        (v_user_group_ratings,
         v_user_group_intercs,
         v_user_group_reviews,
         v_item_group_ratings,
         v_item_group_intercs,
         v_item_group_reviews,
         v_utext, v_itext, v_label) = data_loader.eval_with_group_info()
        # valid_data = {"u_input": v_u_input,
        #               "i_input": v_i_input,
        #               "text": v_text,
        #               "utext": v_utext,
        #               "itext": v_itext
        #               }

        valid_data = [v_user_group_ratings,
                      v_user_group_intercs,
                      v_user_group_reviews,
                      v_item_group_ratings,
                      v_item_group_intercs,
                      v_item_group_reviews,
                      v_utext, v_itext]
        valid=(valid_data,v_label)
        
        # 训练模型
        # self.model.summary()
        t0=time()
        
        history = self.model.fit(train_data, label, epochs=self.epochs, verbose=1,batch_size=self.batch_size,
                                 callbacks=[cp_callback],validation_data=valid, validation_freq=1)
        # 返回训练的历史评价结果
        # 
        each_epoch=(time()-t0)/self.epochs
        print("fit each epoch use {} sec".format(each_epoch))
        t1=time()
        self.model.predict(valid_data)
        print("test use {} sec".format(time()-t1))      
        return history.history

    def train_subfit(self,data_loader,epoch_size=5):
        
        # 存储路径
        checkpoint_dir=self.ckpt_dir
        checkpoint_path=os.path.join(checkpoint_dir,"{}.h5".format(self.name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 通过回调进行保存
        cp_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="mean_absolute_error",
            save_best_only=True, verbose=1, save_weights_only=True, period=1)
        # 配置优化器
        # self.model.compile(optimizer="adam",loss="mean_squared_error",
        #     metrics = [RootMeanSquaredError(), "mean_absolute_error"])
        self.model.compile(optimizer="adam",loss="mean_squared_error",
            metrics = [RootMeanSquaredError(), "mean_absolute_error"])

        # 读取训练数据
        (   user_group_ratings,
            user_group_intercs,
            user_group_reviews,
            item_group_ratings,
            item_group_intercs,
            item_group_reviews,
            utext, itext, label) = data_loader.all_train_data_with_group_info(self.group_info)
        train_data = {
            "user_group_rating": user_group_ratings,
            "user_group_interc": user_group_intercs,
            "user_group_review": user_group_reviews,
            "item_group_rating": item_group_ratings,
            "item_group_interc": item_group_intercs,
            "item_group_review": item_group_reviews,
            "utext": utext,
            "itext": itext
        }

        (v_user_group_ratings,
         v_user_group_intercs,
         v_user_group_reviews,
         v_item_group_ratings,
         v_item_group_intercs,
         v_item_group_reviews,
         v_utext, v_itext, v_label) = data_loader.eval_with_group_info()
        # valid_data = {"u_input": v_u_input,
        #               "i_input": v_i_input,
        #               "text": v_text,
        #               "utext": v_utext,
        #               "itext": v_itext
        #               }

        valid_data = [v_user_group_ratings,
                      v_user_group_intercs,
                      v_user_group_reviews,
                      v_item_group_ratings,
                      v_item_group_intercs,
                      v_item_group_reviews,
                      v_utext, v_itext]
        valid=(valid_data,v_label)
        
        # 训练模型
        # self.model.summary()
        sub_epoch=int(self.epochs/epoch_size)
        
        dfs=[]
        for i in range(sub_epoch):
            t0=time()
            history = self.model.fit(train_data, label, epochs=epoch_size, verbose=1,batch_size=self.batch_size,
                                    callbacks=[cp_callback],validation_split=0.1, validation_freq=5)
            # 返回训练的历史评价结果
            # 
            each_epoch=(time()-t0)/epoch_size
            print("fit each epoch use {} sec".format(each_epoch))
            t1=time()
            print(self.model.evaluate(valid_data,v_label))
            print("test use {} sec".format(time()-t1))      
            dfs.append(
                pd.DataFrame(history.history)
            )
        df=pd.concat(dfs)

        return df

    def train_subfit_subdata(self,data_loader,validation_split=0.1):
        
        # 存储路径
        checkpoint_dir=self.ckpt_dir
        checkpoint_path=os.path.join(checkpoint_dir,"{}.h5".format(self.name))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 通过回调进行保存
        cp_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="mean_absolute_error",
            save_best_only=True, verbose=1, save_weights_only=True, period=1)
        # 配置优化器
        # self.model.compile(optimizer="adam",loss="mean_squared_error",
        #     metrics = [RootMeanSquaredError(), "mean_absolute_error"])
        self.model.compile(optimizer="adam",loss="mean_squared_error",
            metrics = [RootMeanSquaredError(), "mean_absolute_error"])

        # 读取训练数据
        (   user_group_ratings,
            user_group_intercs,
            user_group_reviews,
            item_group_ratings,
            item_group_intercs,
            item_group_reviews,
            utext, itext, label) = data_loader.all_train_data_with_group_info(self.group_info)
        train_data = {
            "user_group_rating": user_group_ratings,
            "user_group_interc": user_group_intercs,
            "user_group_review": user_group_reviews,
            "item_group_rating": item_group_ratings,
            "item_group_interc": item_group_intercs,
            "item_group_review": item_group_reviews,
            "utext": utext,
            "itext": itext
        }

        (v_user_group_ratings,
         v_user_group_intercs,
         v_user_group_reviews,
         v_item_group_ratings,
         v_item_group_intercs,
         v_item_group_reviews,
         v_utext, v_itext, v_label) = data_loader.eval_with_group_info()
        # valid_data = {"u_input": v_u_input,
        #               "i_input": v_i_input,
        #               "text": v_text,
        #               "utext": v_utext,
        #               "itext": v_itext
        #               }

        valid_data = [v_user_group_ratings,
                      v_user_group_intercs,
                      v_user_group_reviews,
                      v_item_group_ratings,
                      v_item_group_intercs,
                      v_item_group_reviews,
                      v_utext, v_itext]
        valid=(valid_data,v_label)
        
        # 训练模型
        # self.model.summary()
        
        res=[]
        trian_times=[]
        test_times=[]
        for i in range(self.epochs):
            t0=time()
            print("Epoch {}".format(i))
            history = self.model.fit(train_data, label, epochs=1, verbose=1,batch_size=self.batch_size,
                                    callbacks=[cp_callback],validation_split=validation_split, validation_freq=5)
            # 返回训练的历史评价结果
            # 
            
            trian_times.append(time()-t0)
            t1=time()
            res.append(self.model.evaluate(valid_data,v_label))
            test_times.append(time()-t1) 
            print(res[-1])
        
        df = {
            "val_loss": [i[0] for i in res],
            "val_root_mean_squared_error":  [i[1] for i in res],
            "val_mean_absolute_error": [i[2] for i in res],
        }
        print("fit each epoch use {} sec".format(np.mean(trian_times)))
        print("test use {} sec".format(np.mean(test_times)))
        return df


if __name__=="__main__":
    print("hello,this is a work of RS")

