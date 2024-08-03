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


class ADSE():
    """
    基于共享的模型: 对所有用户和物品的向量进行了共享。
    共享信息包括：review,interaction,rating
    要求分组信息是提前计算好的。
    """
    name="DoubleEmbeddingModel2"

    def __init__(self,flags,data_loader,group_info):
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

    
    def build_ui_laten(self):
        """
        构建u和i的隐向量
        """

        ## 评论级别
        self.user_review_embed = Embedding(
            self.num_user_group, self.emb_size, name="u_emb")
        self.item_review_embed = Embedding(
            self.num_item_group, self.emb_size, name="i_emb")
        self.u_review_latent = self.user_review_embed(self.user_group_review)
        self.i_review_latent = self.item_review_embed(self.item_group_review)

        # # 交互级别
        self.user_interc_embed = Embedding(
            self.num_user_group, self.emb_size, name="wu_embed")
        self.item_interc_embed = Embedding(
            self.num_item_group, self.emb_size, name="wi_embed")
        self.u_interc_latent = self.user_review_embed(self.user_group_interc)
        self.i_interc_latent = self.item_review_embed(self.item_group_interc)

        # 评分级别
        self.u_rating = Embedding(
            self.num_user_group, self.emb_size, name="u_rating")
        self.i_rating = Embedding(
            self.num_item_group, self.emb_size, name="i_rating")
        # Embeddin
        self.u_rating_latent = self.u_rating(self.user_group_rating)
        self.i_rating_latent = self.i_rating(self.item_group_rating)

        return

    def build_ui_fc(self):
        self.user_review_fc =build_shared_layer(self.routing_mode,(self.num_user_group,self.emb_size),reg=self.reg,name="user_review_embed")
        self.item_review_fc =build_shared_layer(self.routing_mode,(self.num_item_group,self.emb_size),reg=self.reg,name="item_review_embed")
 
        # # 交互级别
        self.user_interc_fc = build_shared_layer(self.routing_mode,(self.num_user_group,self.emb_size),reg=self.reg)#,name="user_interc_embed"
        self.item_interc_fc = build_shared_layer(self.routing_mode,(self.num_item_group,self.emb_size),reg=self.reg)#,name="item_interc_embed"

        # 评分级别
        self.user_rating_fc = build_shared_layer(self.routing_mode,(self.num_user_group,self.emb_size),reg=self.reg)#,name="u_rating_embed"
        self.item_rating_fc = build_shared_layer(self.routing_mode,(self.num_item_group,self.emb_size),reg=self.reg)#,name="u_rating_embed"
        

    def get_model(self,summary=False):
        """
        写成显示的数据流会更好分析
        """

        # 获取输入
        self.init_input()

        # 构建v,d的相关结构
        self.build_word_level_layers()
        self.build_document_level_layers()

        # 构造基本的u|i 隐变量
        self.build_ui_laten()
        self.build_ui_fc()
        self.build_interc_layers()

        # 获取u,i的文本特征 6,100
        # 这里使用到了u_laten, i_laten
        docs_u, doc_u = self.get_w_u(self.utext)
        docs_i, doc_i = self.get_w_i(self.itext)
        # 基于注意力的w_u,w_i
        w_u, w_i = self.get_doc_level_att(doc_u, docs_u, doc_i, docs_i)
        doc_u_fc=tf.reshape(self.user_review_fc(w_u),shape=(-1,1,self.emb_size))
        doc_i_fc=tf.reshape(self.item_review_fc(w_i),shape=(-1,1,self.emb_size))
        w_u_fc, w_i_fc = self.get_doc_level_att(doc_u_fc, docs_u, doc_i_fc, docs_i)

        # 预测的w
        pred_d = self.predict_value_d(self.u_interc_latent, w_u_fc, self.i_interc_latent, w_i_fc)
        u_interc_latent_fc=self.user_interc_fc(pred_d)
        i_interc_latent_fc=self.item_interc_fc(pred_d)
        pred_fc=self.predict_value_d(u_interc_latent_fc, w_u_fc, i_interc_latent_fc, w_i_fc)

        rating_latent,r_w = self.predict_by_d(pred_fc,self.u_rating_latent,self.i_rating_latent)

        u_rating_latent_fc=self.user_rating_fc(rating_latent)
        i_rating_latent_fc=self.item_rating_fc(rating_latent)

        rating_latent_fc,r_w_fc=self.predict_by_d(pred_fc,u_rating_latent_fc,i_rating_latent_fc)
        # 加一个sigmoid把r_w_fc转化为概率
        # r_w_fc = tf.nn.sigmoid(r_w_fc)

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



    def build_word_level_layers(self):
        # 构造vocab
        self.build_vocab()              # 构造v
        
    def build_document_level_layers(self):
        # 构造d
        self.build_document_latent()    # 构造P(d)
        self.build_document_user()      # 构造P(d|u)
        self.build_document_item()      # 构造P(d|i)
        self.budild_r_d()               # 构造P(r|d)
    
    @wrapper
    def build_vocab(self):
        """

        """
        # assert self.emb_size==self.vec_texts.shape[1],"shape not same"
        # 加载词嵌入-这里使用了glove,且不可训练
        self.vec_embed = Embedding(self.vec_texts.shape[0],self.vec_texts.shape[1], name="build_w",
                                 trainable=False, weights=[self.vec_texts])
        
        # 1. 这里是个点
        self.w_embed=Embedding(self.vocab_size,self.emb_size,name="w_emb")
        self.w_att = Dense(self.emb_size*2, activation = 'tanh')
       
        # 
        
        layers=[(128,3),(128,5),(self.emb_size,3)]


        self.convs=[]
        for a, b in layers:
            self.convs.append(Conv1D(a, b, padding="same"))

        # self.conv1=Conv1D(128,3,padding="same")
        # self.conv2=Conv1D(128,5,padding="same")
        # self.conv3=Conv1D(self.emb_size,3,padding="same")


        # 问题是，如何构建语义向量空间
    
    def get_wui_convs(self,latent):
        latent=tf.reshape(latent,[-1,self.maxlen,self.emb_size])

        # for conv in self.convs[:-1]:
        #     latent=conv(latent)
        conv1=self.convs[0](latent)
        conv2=self.convs[1](conv1)

        hidden = tf.nn.relu(tf.concat([conv1,conv2], axis=-1))
        hidden = tf.nn.dropout(hidden, self.keep_prob)

        conv3 = tf.nn.relu(self.convs[2](hidden))
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        return conv3

    def build_document_latent(self): 
        """
        定义计算得到document的结构
        """
        pass

    def build_document_user(self):
        """
        定义计算用户先验的document的结构，即P(d|u)
        """
        pass  
    def build_document_item(self):
        """
        定义计算物品先验的document的结构，即P(d|i)
        """
        pass  

    def get_document_user(self):
        pass  
    def get_document_item(self):
        pass  

    @wrapper
    def budild_r_d(self):
        """
        通过词空间预测评分
        """
        self.rd_layers=[]
        # for one_dim in layers:
        for i in [3,2,1]:
            one_dim=i*self.emb_size
            layer=Dense(one_dim,activation="elu",name="Pr_w_{}".format(one_dim))
            self.rd_layers.append(layer)
        Pred = Dense(1, bias_initializer=tf.constant_initializer(2),
                     activation="elu", name="P_r_w")
        self.rd_layers.append(Pred)
        


    
    def get_certainty(self,alpha): # ?,6,60
        # alpha_sort = tf.sort(alpha, axis=-1)
        alpha_mean = tf.reduce_mean(alpha, axis=-1, keepdims = True) # ?,6,1
        # alpha_sort = alpha
        upper_mask = alpha>alpha_mean
        upper_mask = tf.cast(upper_mask, tf.float32)
        lower_mask = 1.-upper_mask   # ?,6,60

        alpha_lower = tf.reduce_mean(alpha*lower_mask, axis=-1, keepdims = True) # ?,6,1
        alpha_upper = tf.reduce_mean(alpha*upper_mask, axis=-1, keepdims = True)

        certainty = tf.nn.sigmoid((alpha_upper-alpha_mean)*(alpha_mean-alpha_lower))
        certainty = 2*certainty - 1
        # certainty = tf.expand_dims(certainty, axis=-1)
        return certainty

    @wrapper
    def get_word_level_att(self,uit_cnn, u_latent, i_latent, name='user'):

        uit_cnn_rsh = tf.reshape(uit_cnn, [-1, self.t_num, self.maxlen, self.emb_size])

        trans = self.w_att(uit_cnn_rsh) #[?,8,60,200]

        ui_latent = tf.concat([u_latent, i_latent], axis=-1)
        latent = tf.expand_dims(tf.expand_dims(ui_latent,1),1) #[?,1,1,200]
        alpha = tf.reduce_sum(trans*latent,axis=-1) #[?,8,60]


        alpha = tf.nn.softmax(alpha, axis=-1)
        if name == 'user':
            self.word_user_alpha = alpha
        else:
            self.word_item_alpha = alpha

        # certainty = self.get_certainty(alpha)
        # self.certainty = certainty

        alpha = tf.expand_dims(alpha, axis=-1) #[?,8,60,1]

        hidden = tf.reduce_sum(alpha*uit_cnn_rsh, axis=2) #[?,8,100]

        # print(certainty.shape, alpha.shape)

        return hidden #*certainty        

    @wrapper
    def doc_level_att(self, vec_1, vec_2, layer, name='user'):

        dist = tf.reduce_mean(tf.square(vec_1 - vec_2), axis=-1)*(layer+1)*10
        dist = -dist
        if layer == 0:
            self.vec_1 = vec_1
            self.vec_2 = vec_2
        alpha_1 = tf.nn.softmax(dist, axis=-1) # ?,6
        alpha_2 = tf.expand_dims(alpha_1, axis=-1)
        # if name == 'user':
        #     self.doc_user_alpha.append(alpha_1)
        # else:
        #     self.doc_item_alpha.append(alpha_1)

        return tf.reduce_sum(alpha_2*vec_2, axis=1, keepdims = True)


    @wrapper
    def get_doc_level_att(self, doc_user,docs_user, doc_item,docs_item):

        layers = self.doc_layers
        doc_att_layers=[]
        for i in range(layers):
            if i==0:
                i_temp = self.doc_level_att(doc_user, docs_item, i, 'item')
                u_temp = self.doc_level_att(doc_item, docs_user, i, 'user')
            else:
                i_temp = self.doc_keep* i_temp+ (1-self.doc_keep)*  self.doc_level_att(doc_user, docs_item, i, 'item')
                u_temp =  self.doc_keep* u_temp+(1-self.doc_keep)*  self.doc_level_att(doc_item, docs_user, i, 'user')                
            # self.doc_item.append(i_temp)
            # self.doc_user.append(u_temp)
        u_temp=tf.squeeze(u_temp,axis=1)
        i_temp=tf.squeeze(i_temp,axis=1)
        return u_temp,i_temp
    
    # get 方法似乎没有必要
    def get_w(index):
        """
        return w
        """

    @wrapper
    def get_w_u_i(self,uitext,name="user"):
        """
        计算P(w|u)和P(w|i)的通用模板
        """
        # 加载用户评论
        uidocs=self.vec_embed(uitext)
        # 计算用户对词空间的偏好

        uiw_emb=self.w_embed(uidocs)
        ui_cnn=self.get_wui_convs(uiw_emb)

        # !!! 这里需要进一步细化
        ui_watt=self.get_word_level_att(ui_cnn,self.u_review_latent,self.i_review_latent,name)
        doc_ui = tf.reduce_mean(ui_watt, axis=1, keepdims = True)

        return ui_watt,doc_ui        
    
    def get_w_u(self,utext):
        """
        return P(w|u)
        """
        return self.get_w_u_i(utext,"user")
        

    def get_w_i(self,itext):
        """
        return P(w|i)
        """
        # 加载物品评论
        return self.get_w_u_i(itext,"item")

    def build_interc_layers(self):
        self.interc_layers=[]
        for i in [3, 2, 1]:
            one_dim = i*self.emb_size
            self.interc_layers.append(
                Dense(one_dim, activation="relu", name="predcit_w_{}".format(one_dim)))

    def predict_value_d(self,u_latent,w_u,i_latent,w_i):
        
        """
        直接P(w|u)和P(w|i)得到P(w|u,i)
        预测用户u对i的评论w
        return value of w
        """
        # 交互级偏置
        
        laten=concatenate([u_latent,w_u,i_latent,w_i])

        layer=laten
        for i in range(3):
            layer=self.interc_layers[i](layer)
        
        return layer


    def predict_by_d(self,w,u_latent,i_latent):
        """
        return P(r|w)
        """
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
            utext, itext, label,ctr) = data_loader.all_train_data_with_group_info(self.group_info)
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
         v_utext, v_itext, v_label,v_ctr) = data_loader.eval_with_group_info()
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

