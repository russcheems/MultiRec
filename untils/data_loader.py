import os
import pickle
import random
import sys
from os.path import join
from textwrap import shorten

import numpy as np
import tensorflow as tf
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix

# from untils import build_data
from .process import build_data,get_neighbors_1
from.sample import negative_review_sampler_random,negative_review_sampler_sam_pos,negative_review_sampler_sim_neg,negative_review_sampler_sim_neg2,negative_review_sampler_revise_pos


class Data_Loader():
    Glove_path="glove"
    def __init__(self, flags): #是否将负样本写入init方法ysq

        # 创建独立的文件夹来保存处理后的文件
        self.dir=flags.filename.split('.')[0]
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.pro_path=os.path.join(self.dir,"data")

        if not os.path.exists(self.pro_path+'.pkl'):
            build_data(flags.filename,self.pro_path)
        data = pickle.load(open(self.pro_path+'.pkl','rb')) # 将文件中的数据解析为一个python对象
        # data为字典
 

        # print(len(data))
        vec_texts 	= data['vec_texts']
        vocab 		= data['vocab']
        vec_uit 	= data['vec_uit']
        vec_u_text 	= data['vec_u_text']
        vec_i_text 	= data['vec_i_text']
        self.user2idx 	= data['user2idx']
        self.item2idx 	= data['item2idx']

        self.Glove_path = flags.glovepath
        self.filename 	= flags.filename
        self.word2idx 	= data['word2idx']
        self.batch_size = flags.batch_size
        self.num_user 	= len(self.user2idx)
        self.num_item 	= len(self.item2idx)
        self.vocab_size = len(vocab)+1
        self.data_size 	= len(vec_uit)
        self.emb_size 	= flags.emb_size

        # self.test_size  = flags.test_size

        self.t_num 	= 6 # 这里参数是什么意思？
        self.maxlen = 60
        self.vec_texts 	= pad_sequences(vec_texts, maxlen = self.maxlen) # 将评论长度统一
        # self.vec_u_text = pad_sequences(vec_u_text, maxlen = self.t_num, padding='post')
        # self.vec_i_text = pad_sequences(vec_i_text, maxlen = self.t_num, padding='post')
        self.vec_u_text = vec_u_text[:,:self.t_num]
        print(self.vec_u_text.shape) #（10261，6）
        self.vec_i_text = vec_i_text[:,:self.t_num]

        self.vec_uit = np.array(vec_uit)

        pmtt = np.random.permutation(self.data_size)
        pmtt_file = self.pro_path+'_pmtt.npy'

        if not os.path.exists(pmtt_file):
            np.save(pmtt_file, pmtt)
        else:
            print('pmtt file exist')
            pmtt = np.load(pmtt_file).astype(np.int32)

        # self.vec_uit = np.random.permutation(self.vec_uit)

        self.vec_uit = self.vec_uit[pmtt]
        self.vec_u_text = self.vec_u_text[pmtt]
        self.vec_i_text = self.vec_i_text[pmtt]

        # print(len(self.vec_uit))
        # print(len(self.vec_u_text))

        self.train_size = int(self.data_size*(1-flags.test_size))
        # self.test_size = int(self.data_size* flags.test_size )
        # 固定测试集的大小去验证
        self.test_size = int(self.data_size* 0.1)
        # 划分训练集 测试集
        
        self.train_uit = self.vec_uit[:self.train_size]
        self.train_u_text = self.vec_u_text[:self.train_size]
        self.train_i_text = self.vec_i_text[:self.train_size]
        # self.test_uit = self.vec_uit[self.train_size:]
        self.test_uit = self.vec_uit[-self.test_size:]
        self.test_u_text = self.vec_u_text[-self.test_size:]
        self.test_i_text = self.vec_i_text[-self.test_size:]

        self.pointer = 0

        self.get_embedding()


    def next_batch(self):
        begin 	= self.pointer*self.batch_size
        end		= (self.pointer+1)*self.batch_size
        self.pointer+=1
        if end >= self.train_size:
            end = self.train_size
            self.pointer = 0

        labels = self.train_uit[begin:end][:,3:] #评分，是否有帮助 ，是否有帮助可用分类问题处理？
        users = self.train_uit[begin:end][:,0]
        items = self.train_uit[begin:end][:,1]
        texts = self.train_uit[begin:end][:,2]

        utexts = self.train_u_text[begin:end]
        itexts = self.train_i_text[begin:end]
        # print(i_texts)
        return users, items, labels, utexts,itexts, texts

    def set_mode(self,mode):
        if mode == "r":
            self.mode = "r" # 相反
        elif mode == "n":
            self.mode = "n" # 相似或不相似
        else:
            self.mode = "rd" # 随机

    def all_train_data(self, neg_sample_ratio=2):
        users = self.train_uit[:, 0]
        items = self.train_uit[:, 1]
        labels = self.train_uit[:, 3:]
        utexts = self.train_u_text
        itexts = self.train_i_text
        texts = self.train_uit[:, 2]
        lable_ctr = np.zeros((len(users), 1))

        for i in range(len(labels)):
            if labels[i][0] > 3:
                lable_ctr[i][0] = 1
            else:
                lable_ctr[i][0] = 0

        all_users = np.unique(users)
        all_items = np.unique(items)

        ui_info = {}
        for row in self.train_uit:
            user = row[0]
            item = row[1]
            if user not in ui_info:
                ui_info[user] = [item]
            else:
                ui_info[user].append(item)

        neg_users = np.tile(users, neg_sample_ratio)
        neg_items = np.tile(items, neg_sample_ratio)
        neg_labels = np.zeros((len(users) * neg_sample_ratio, 1))

        neg_utexts = np.tile(utexts, (neg_sample_ratio, 1))
        neg_itexts = np.tile(itexts, (neg_sample_ratio, 1))
        neg_texts = np.zeros(len(users) * neg_sample_ratio)
        neg_ratings = np.zeros((len(users) * neg_sample_ratio, 1))

        for i in range(len(neg_labels)):
            flag = np.random.random()
            if flag < 0.2:
                neg_ratings[i] = 3
            else:
                neg_ratings[i] = 4

        for i in range(len(users)):
            interacted_items = ui_info.get(users[i], [])
            non_interacted_items = np.setdiff1d(all_items, interacted_items)
            
            if len(non_interacted_items) > 0:
                for j in range(neg_sample_ratio):
                    random_item = np.random.choice(non_interacted_items)
                    neg_index = i * neg_sample_ratio + j
                    neg_items[neg_index] = random_item
                    idx = np.where(items == random_item)[0][0]
                    neg_itexts[neg_index] = itexts[idx]

        users = np.concatenate((users, neg_users), axis=0)
        items = np.concatenate((items, neg_items), axis=0)
        labels = np.concatenate((labels, neg_ratings), axis=0)
        texts = np.concatenate((texts, neg_texts), axis=0)
        lable_ctr = np.concatenate((lable_ctr, neg_labels), axis=0)
        # 先降维
        users = users.reshape(-1)
        items = items.reshape(-1)
        labels = labels.reshape(-1)

        ui_mat = csr_matrix((labels, (users, items)), shape=(max(users)+1, max(items)+1), dtype='int8')
        u_neighbor_mat = get_neighbors_1(ui_mat)
        i_neighbor_mat = get_neighbors_1(ui_mat.T)


        new_train_uit = np.zeros((len(users), 4))
        new_train_uit[:, 0] = users
        new_train_uit[:, 1] = items
        new_train_uit[:, 2] = texts
        labels = labels.reshape(-1, 1)
        new_train_uit[:, 3:] = labels
        print("new_train_uit",new_train_uit.shape)
        print("self.train_uit",self.train_uit.shape)
        # 创建neg_train_uit
        neg_train_uit = np.zeros((len(neg_users), 4))
        neg_train_uit[:, 0] = neg_users
        neg_train_uit[:, 1] = neg_items
        neg_train_uit[:, 2] = neg_texts
        neg_train_uit[:, 3:] = neg_ratings

        if self.mode == "r":
            neg_utexts = negative_review_sampler_revise_pos(new_train_uit, list(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='user')
            neg_itexts = negative_review_sampler_revise_pos(new_train_uit,(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='item')
        elif self.mode == "n":
            neg_utexts = negative_review_sampler_sim_neg(new_train_uit, list(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='user')
            neg_itexts = negative_review_sampler_sim_neg(new_train_uit, list(range(len(items)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='item')
        elif self.mode == "rd":
            neg_utexts = negative_review_sampler_random(new_train_uit, list(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='user')
            neg_itexts = negative_review_sampler_random(new_train_uit, list(range(len(items)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='item')
        
        
        neg_utexts = neg_utexts.reshape(-1, 6)
        neg_itexts = neg_itexts.reshape(-1, 6)
        print(neg_utexts.shape,"neg_utexts")
        


        utexts = np.concatenate((utexts, neg_utexts), axis=0)
        itexts = np.concatenate((itexts, neg_itexts), axis=0)

        print("shape of utexts", utexts.shape)
        print("shape of itexts", itexts.shape)
        print("shape of texts", texts.shape)
        print("users", users.shape)
        print("items", items.shape)

        return users, items, labels, utexts, itexts, texts, lable_ctr
    





    def eval(self, neg_sample_ratio=2):
        labels = self.test_uit[:, 3:]
        users = self.test_uit[:, 0]
        items = self.test_uit[:, 1]
        texts = self.test_uit[:, 2]
        utexts = self.test_u_text  # (1026, 6)
        itexts = self.test_i_text

        lable_ctr = np.ones((len(users), 1))  # 默认测试集中每个样本的标签都为1（正样本）
        all_users = np.unique(users)
        all_items = np.unique(items)

        ui_info = {}
        for row in self.train_uit:
            user = row[0]
            item = row[1]
            if user not in ui_info:
                ui_info[user] = [item]
            else:
                ui_info[user].append(item)

        neg_users = np.tile(users, neg_sample_ratio)
        neg_items = np.tile(items, neg_sample_ratio)
        neg_labels = np.zeros((len(users) * neg_sample_ratio, 1))
        neg_utexts = np.tile(utexts, (neg_sample_ratio, 1))
        neg_itexts = np.tile(itexts, (neg_sample_ratio, 1))
        neg_texts = np.zeros(len(users) * neg_sample_ratio)
        neg_ratings = np.zeros((len(users) * neg_sample_ratio, 1))

        for i in range(len(users)):
            user = users[i]
            interacted_items = ui_info.get(user, [])
            non_interacted_items = np.setdiff1d(all_items, interacted_items)

            if len(non_interacted_items) > 0:
                for j in range(neg_sample_ratio):
                    random_item = np.random.choice(non_interacted_items)
                    neg_index = i * neg_sample_ratio + j
                    neg_items[neg_index] = random_item
                    idx = np.where(items == random_item)[0][0]
                    neg_itexts[neg_index] = itexts[idx]

        users = np.concatenate((users, neg_users), axis=0)
        items = np.concatenate((items, neg_items), axis=0)
        labels = np.concatenate((labels, neg_ratings), axis=0)
        # utexts = np.concatenate((utexts, neg_utexts), axis=0)
        # itexts = np.concatenate((itexts, neg_itexts), axis=0)
        texts = np.concatenate((texts, neg_texts), axis=0)
        lable_ctr = np.concatenate((lable_ctr, neg_labels), axis=0)
        
        users = users.reshape(-1)
        items = items.reshape(-1)
        labels = labels.reshape(-1)

        ui_mat = csr_matrix((labels, (users, items)), shape=(max(users)+1, max(items)+1), dtype='int8')
        u_neighbor_mat = get_neighbors_1(ui_mat)
        i_neighbor_mat = get_neighbors_1(ui_mat.T)


        new_train_uit = np.zeros((len(users), 4))
        new_train_uit[:, 0] = users
        new_train_uit[:, 1] = items
        new_train_uit[:, 2] = texts
        labels = labels.reshape(-1, 1)
        new_train_uit[:, 3:] = labels
        print("new_train_uit",new_train_uit.shape)
        print("self.train_uit",self.train_uit.shape)
        # 创建neg_train_uit
        neg_train_uit = np.zeros((len(neg_users), 4))
        neg_train_uit[:, 0] = neg_users
        neg_train_uit[:, 1] = neg_items
        neg_train_uit[:, 2] = neg_texts
        neg_train_uit[:, 3:] = neg_ratings



        # neg_utexts = negative_review_sampler_revise_pos(new_train_uit, list(range(8208)), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='user')
        # neg_itexts = negative_review_sampler_revise_pos(new_train_uit,list(range(8208)), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='item')

        if self.mode == "r":
            neg_utexts = negative_review_sampler_revise_pos(new_train_uit, list(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='user')
            neg_itexts = negative_review_sampler_revise_pos(new_train_uit,(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='item')
        elif self.mode == "n":
            neg_utexts = negative_review_sampler_sim_neg(new_train_uit, list(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='user')
            neg_itexts = negative_review_sampler_sim_neg(new_train_uit, list(range(len(items)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='item')
        elif self.mode == "rd":
            neg_utexts = negative_review_sampler_random(new_train_uit, list(range(len(users)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='user')
            neg_itexts = negative_review_sampler_random(new_train_uit, list(range(len(items)*neg_sample_ratio//(neg_sample_ratio+1))), neg_train_uit, u_neighbor_mat, i_neighbor_mat, name='item')
        
        
        neg_utexts = neg_utexts.reshape(-1, 6)
        neg_itexts = neg_itexts.reshape(-1, 6)
        print(neg_utexts.shape,"neg_utexts")
        


        utexts = np.concatenate((utexts, neg_utexts), axis=0)
        itexts = np.concatenate((itexts, neg_itexts), axis=0)

        return users, items, labels, utexts, itexts, texts, lable_ctr




    def assert_doc(self, idxs, rating): #检查给定的用户和物品在训练集中是否有指定评分的记录。
        res = []
        for idx in idxs:
            # label = self.vec_uit[self.vec_uit[:,2] == doc_idx][0][3]
            utexts = self.train_u_text[idx]
            itexts = self.train_i_text[idx]
            u_flag = 0
            for utext in utexts:
                if utext == 0:
                    continue
                label = self.vec_uit[self.vec_uit[:,2] == utext][0][3]
                if label == rating:
                    u_flag +=1
            i_flag = 0
            for itext in itexts:
                if itext == 0:
                    continue
                label = self.vec_uit[self.vec_uit[:,2] == itext][0][3]
                if label == rating:
                    i_flag +=1
            if u_flag >1 and i_flag > 1:
                res.append(idx)
        return res

    def assert_docs(self, pos_idxs, neg_idxs): #没用到
        pos = []
        neg = []
        pos = self.assert_doc(pos_idxs, 5)
        neg = self.assert_doc(neg_idxs, 1)
        if len(pos)!=0 and len(neg)!=0:
            print(len(pos), len(neg))
        if len(pos) == 0 or len(neg) == 0:
            return False
        else:
            return pos[0], neg[0]
        

    
    def find_a_user(self): 
        # while True:
        #     idx  = np.random.randint(self.train_size)
        all_users=list(range(self.train_size))
        # np.random.shuffle(all_users)

        for idx in all_users:
            user = self.train_uit[idx][0]
            sub_indices = np.where(self.train_uit[:,0] == user)[0] #where(condition)返回一个元组

            pos_indices = np.where(self.train_uit[sub_indices,3] == 5)[3] #所有用户a喜欢的列表
            neg_indices = np.where(self.train_uit[sub_indices,3] == 1)[3]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                pass
            else:
                pos_idxs = sub_indices[pos_indices]
                neg_idxs = sub_indices[neg_indices]

                res = self.assert_docs(pos_idxs, neg_idxs)
                if res == False:
                    continue
                else:
                    pos_idx, neg_idx = res

                # pos_idx = np.random.choice(pos_idxs)
                # neg_idx = np.random.choice(neg_idxs)

                pos = self.train_uit[pos_idx]
                neg = self.train_uit[neg_idx]

                uit = np.array([pos,neg])
                idx = np.array([pos_idx, neg_idx])

                break
        user  = uit[:,0]
        item  = uit[:,1]
        text  = uit[:,2]
        label = uit[:,3:]
        utexts = self.train_u_text[idx]
        itexts = self.train_i_text[idx]
        for utext in utexts[1]:
            # print(utexts)
            if utext == 0:
                continue
            # print(np.where(self.vec_uit[:,2] == utext))
            ulabel = self.vec_uit[self.vec_uit[:,2] == utext][0][3]
            print("utext label: ", ulabel)
        for itext in itexts[1]:
            if itext == 0:
                continue
            ilabel = self.vec_uit[self.vec_uit[:,2] == itext][0][3]
            print('itext label: ', ilabel)


        return user,item,label,utexts,itexts,text


    def sample_point(self): 
        # seed = np.random.random()
        seed = 0.6
        index = []
        for i in range(self.train_size):
            if seed<0.5 and self.train_uit[i][3]==5:
                index.append(i)
            elif seed>0.5 and self.train_uit[i][3]==1:
                index.append(i)
        idx = np.random.choice(index)

        # idx = np.random.randint(self.train_size)
        print('random index: ', idx)

        sample_data = self.train_uit[idx:idx+1]

        user  = sample_data[:,0]
        item  = sample_data[:,1]
        text  = sample_data[:,2]
        label = sample_data[:,3:]

        utexts = self.train_u_text[idx:idx+1]
        itexts = self.train_i_text[idx:idx+1]

        # print(text)
        # print(utexts, itexts)


    


        return user,item, label, utexts, itexts, text

    def reset_pointer(self):
        self.pointer = 0



    def get_embedding(self):
        # emb_file = self.filename.split('.')[0]+'_'+str(self.emb_size)+'d.emb'
        emb_file=self.pro_path+'d.emb'
        if not os.path.exists(emb_file):
            self.w_embed = np.random.uniform(-0.25,0.25,(self.vocab_size, self.emb_size))
            self.w_embed[0] = 0
            file = os.path.join(self.Glove_path,'glove.6B.'+str(self.emb_size)+'d.txt')
            fr = open(file)
            for line in fr.readlines():
                line = line.strip()
                listfromline = line.split()
                word = listfromline[0]
                if word in self.word2idx:
                    vect = np.array(list(map(np.float32,listfromline[1:])))
                    idx = self.word2idx[word]
                    self.w_embed[idx] = vect
            np.savetxt(emb_file, self.w_embed, fmt='%.8f')
        else:
            self.w_embed = np.genfromtxt(emb_file)
    


    def validate(self):
        total = self.data_size//self.batch_size
        rand = np.random.randint(total)
        for i in range(rand):
            self.next_batch()
        u,i,l,uts,its,t = self.next_batch()
        print(t,l)
        widxs = self.vec_texts[t]
        idx2word = {v[1]:v[0] for v in self.word2idx.items()}
        
        for widx in widxs:
            words = [idx2word[idx] for idx in widx if idx!=0]
            print(' '.join(words))
            print('\n\n')

        


    

if __name__ == '__main__':
    prefix = 'amazon_data'
    filename = os.path.join(prefix,'Musical_Instruments_5.json')


    flags = tf.flags.FLAGS 	
    tf.flags.DEFINE_string('filename',filename,'name of file')
    tf.flags.DEFINE_integer('batch_size',4,'batch size')
    tf.flags.DEFINE_integer('emb_size',100, 'embedding size')
    tf.flags.DEFINE_string('glovepath',"glove"," path of glove")
    # tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
    flags(sys.argv)

    data_loader = Data_Loader(flags)

    data_loader.sample_point()
    data_loader.next_batch()
    data_loader.validate()
    data_loader.eval()
    res = data_loader.find_a_user()
    print(res)


