import math
import os
import sys
sys.path.append(".")
sys.path.append(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle
import pandas as pd
import tensorflow as tf
# from tensorflow.compat.v1 import flags as tf_flags
from absl import flags as tf_flags
from model import ADSE
from GroupDataLoader import GData_Loader
# from untils.build_user_adj_maxtrix import UserSimMatrix
import numpy as np
from ExpertTSE import ExpertTSE
from MuModel import MuADSE
ReviewShare=4
IntercShare=8
RatingShare=12

def prepare_group_info(data_loader,p,mode=12):
    """
        待完善
        """
    p=p[:-4]
    group_info=dict()
    group_info["num_user_group"]=int(math.sqrt(data_loader.num_user))
    group_info["num_item_group"]=int(math.sqrt(data_loader.num_item))

    sim_path_fmt="/mnt/Disk3/ysq/localFile/TSE/sim_res/{}_{}_{}_{}.pkl"
    
    # 在保存rating分组时发生了命名错误。
    with open(sim_path_fmt.format(p,"rating","os","user"),"rb") as f:
        group_info["user_rating"]=pickle.load(f) 
    with open(sim_path_fmt.format(p,"interc","Jacc","user"),"rb") as f:
        group_info["user_interc"]=pickle.load(f) 
    with open(sim_path_fmt.format(p,"review","ADF","user"),"rb") as f:
        group_info["user_review"]=pickle.load(f) 

    with open(sim_path_fmt.format(p,"rating","os","item"),"rb") as f:
        group_info["item_rating"]=pickle.load(f) 
    with open(sim_path_fmt.format(p,"interc","Jacc","item"),"rb") as f:
        group_info["item_interc"]=pickle.load(f) 
    with open(sim_path_fmt.format(p,"review","ADF","item"),"rb") as f:
        group_info["item_review"]=pickle.load(f) 
    return group_info


def prepare_group_info_random(data_loader,mode=12):
    group_info=dict()
    group_info["num_user_group"]=int(math.sqrt(data_loader.num_user))
    group_info["num_item_group"]=int(math.sqrt(data_loader.num_item))
    
    group_info["user_rating"]=np.random.randint(0,group_info["num_user_group"],data_loader.num_user)
    group_info["user_interc"]=np.random.randint(0,group_info["num_user_group"],data_loader.num_user)
    group_info["user_review"]=np.random.randint(0,group_info["num_user_group"],data_loader.num_user)
    group_info["item_rating"]=np.random.randint(0,group_info["num_item_group"],data_loader.num_item)
    group_info["item_interc"]=np.random.randint(0,group_info["num_item_group"],data_loader.num_item)
    group_info["item_review"]=np.random.randint(0,group_info["num_item_group"],data_loader.num_item)
    
    
    return group_info

if __name__=="__main__":
    
    # filename = os.path.join(prefix,'Grocery_and_Gourmet_Food_5.json')
    
    paths = [
        'Musical_Instruments_5.json',
        "Office_Products_5.json",
        'Grocery_and_Gourmet_Food_5.json',
        "Video_Games_5.json",
        "Sports_and_Outdoors_5.json",
    ]
    prefix = 'amazon_data'
    # filename = os.path.join(prefix,paths[0])
    filename=paths[0]
    flags = tf_flags.FLAGS 	
    tf_flags.DEFINE_string('filename', filename, 'name of file')
    tf_flags.DEFINE_string("res_dir", filename, "name of dir to store result")
    tf_flags.DEFINE_integer('batch_size', 128, 'batch size')
    tf_flags.DEFINE_integer('emb_size', 100, 'embedding size')
    tf_flags.DEFINE_integer('num_class', 5, "num of classes")
    tf_flags.DEFINE_integer('epoch',20, 'epochs for training')
    tf_flags.DEFINE_string('ckpt_dir', os.path.join(
        "CKPT_DIR", "HTI_"+filename.split('.')[0]), 'directory of checkpoint')
    tf_flags.DEFINE_string('train_test', 'train', 'training or test')
    tf_flags.DEFINE_string("glovepath", "glove", "glove path")
    tf_flags.DEFINE_string("res_path", "res/res.csv", "save predict res")
    tf_flags.DEFINE_float('test_size', "0.2", "set test size to split data")
    tf_flags.DEFINE_string('res', "/mnt/Disk3/ysq/localFile/TSE/res/temp.csv", "res path to save")
    tf_flags.DEFINE_integer('mode', -1, "2,4,8表示三种层次的共享")
    tf_flags.DEFINE_integer('doc_layers', 3, "doc层注意力的层数")
    tf_flags.DEFINE_float('doc_dropout', .3, "doc层注意力的层数")
    tf_flags.DEFINE_float("group_scale",1, "group scale based on sqrt(|U|)")
    tf_flags.DEFINE_float('epoch_size', 0.1, 'use sub epoch to fit model')
    # 
    tf_flags.DEFINE_string("routing_mode", "soft", "the routing mode of shared embedding,hard or soft")
    tf_flags.DEFINE_integer("use_reg",0,"use reguler 1 or not 0")
    tf_flags.DEFINE_float('routing_lr', 0.001, 'routing_lr')

    flags(sys.argv)
    
    p=flags.filename
    flags.filename=os.path.join(prefix,flags.filename)
    data_loader = GData_Loader(flags)
    
    group_info=prepare_group_info(data_loader,p)
    # group_info=prepare_group_info_random(data_loader)
    expert = ExpertTSE(flags,data_loader,group_info=group_info,layer = "layer1")
    expert2 = ExpertTSE(flags,data_loader,group_info=group_info,layer = "layer2")
    expert3 = ExpertTSE(flags,data_loader,group_info=group_info,layer = "layer3")
    # model = ADSE(flags,data_loader,group_info=group_info)
    model=MuADSE(flags,data_loader,group_info=group_info,expert=expert,expert2=expert2,expert3=expert3,task = "ctsr")

    model.get_model(summary=True)
    # res=model.train(data_loader)
    if flags.epoch_size==-1:
        res=model.train(data_loader)
        
        res_Df=pd.DataFrame(res)

        print(res_Df)
        if flags.res!="":
            res_Df.to_csv(flags.res,index=False)
    elif flags.epoch_size==1:
        res=model.train_subfit_subdata(data_loader)
        res_Df=pd.DataFrame(res)
        print(res_Df)
        if flags.res!="":
            res_Df.to_csv(flags.res,index=False)      
    elif flags.epoch_size>=0 and  flags.epoch_size<1:
        res=model.train_subfit_subdata(data_loader,flags.epoch_size)
        res_Df=pd.DataFrame(res)
        print(res_Df)
        if flags.res!="":
            res_Df.to_csv(flags.res,index=False)           
    else:
        print(flags.epoch_size,"epoch_size error")
        res=model.train_subfit(data_loader,epoch_size=flags.epoch_size)
        print(res)