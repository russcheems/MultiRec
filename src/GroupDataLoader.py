from multiprocessing.sharedctypes import RawValue
from untils import Data_Loader
import numpy as np

REVIEWSHARED=4
INTERCSHARED=2 # 历史遗留问题
RATINGSHARED=8


class GData_Loader(Data_Loader):
    def all_train_data_with_group_info(self,group_info:dict,mode=14):
        u_input, i_input, label, utext, itext, text, ctr= super().all_train_data()
        # 只要前一半
        # u_input=u_input[:len(u_input)//2]
        # i_input=i_input[:len(i_input)//2]
        # label=label[:len(label)//2]
        # utext=utext[:len(utext)//2]
        # itext=itext[:len(itext)//2]
        # text=text[:len(text)//2]
        # ctr=ctr[:len(ctr)//2]


        self.dict_user_group_ratings=group_info["user_rating"]
        self.dict_user_group_intercs=group_info["user_interc"]
        self.dict_user_group_reviews=group_info["user_review"]

        self.dict_item_group_ratings=group_info["item_rating"]
        self.dict_item_group_intercs=group_info["item_interc"]
        self.dict_item_group_reviews=group_info["item_review"]        
        #
        if mode & RATINGSHARED:
            user_group_ratings=np.array([self.dict_user_group_ratings[i] for i in u_input])
            item_group_ratings=np.array([self.dict_item_group_ratings[i] for i in i_input])
        else:
            user_group_ratings=u_input
            item_group_ratings=i_input
        #
        if mode & INTERCSHARED:
            user_group_intercs=np.array([self.dict_user_group_intercs[i] for i in u_input])
            item_group_intercs=np.array([self.dict_item_group_intercs[i] for i in i_input])
        else:
            user_group_intercs=u_input
            item_group_intercs=i_input   
        # 
        if mode & REVIEWSHARED:
            user_group_reviews=np.array([self.dict_user_group_reviews[i] for i in u_input])
            item_group_reviews=np.array([self.dict_item_group_reviews[i] for i in i_input])
            
        else:
            user_group_reviews=u_input
            item_group_reviews=i_input

        return (user_group_ratings,
                user_group_intercs,
                user_group_reviews,
                item_group_ratings,
                item_group_intercs,
                item_group_reviews,
                utext, itext, label,ctr)
    
    def eval_with_group_info(self,mode=14):
        u_input, i_input, label, utext, itext, text,ctr_eval= super().eval()

        # u_input=u_input[:len(u_input)//2]
        # i_input=i_input[:len(i_input)//2]
        # label=label[:len(label)//2]
        # utext=utext[:len(utext)//2]
        # itext=itext[:len(itext)//2]
        # text=text[:len(text)//2]
        # ctr_eval= ctr_eval[:len(ctr_eval)//2]

        #
        if mode & RATINGSHARED:
            user_group_ratings=np.array([self.dict_user_group_ratings[i] for i in u_input])
            item_group_ratings=np.array([self.dict_item_group_ratings[i] for i in i_input])
        else:
            user_group_ratings=u_input
            item_group_ratings=i_input
        #
        if mode & INTERCSHARED:
            user_group_intercs=np.array([self.dict_user_group_intercs[i] for i in u_input])
            item_group_intercs=np.array([self.dict_item_group_intercs[i] for i in i_input])
        else:
            user_group_intercs=u_input
            item_group_intercs=i_input   
        # 
        if mode & REVIEWSHARED:
            item_group_reviews=np.array([self.dict_item_group_reviews[i] for i in i_input])
            user_group_reviews=np.array([self.dict_user_group_reviews[i] for i in u_input])
        else:
            item_group_reviews=u_input
            user_group_reviews=i_input  


        return (user_group_ratings,
                user_group_intercs,
                user_group_reviews,
                item_group_ratings,
                item_group_intercs,
                item_group_reviews,
                utext, itext, label,ctr_eval)

class GData_Loader_signal(Data_Loader):
    def all_train_data_with_group_info(self,group_info:dict,mode):
        u_input, i_input, label, utext, itext, text= super().all_train_data()
        
        self.dict_user_group_ratings=group_info["user_rating"]
        self.dict_user_group_intercs=group_info["user_interc"]
        self.dict_user_group_reviews=group_info["user_review"]

        self.dict_item_group_ratings=group_info["item_rating"]
        self.dict_item_group_intercs=group_info["item_interc"]
        self.dict_item_group_reviews=group_info["item_review"]        
        #
        if mode == RATINGSHARED:
            the_user_group=np.array([self.dict_user_group_ratings[i] for i in u_input])
            the_item_group=np.array([self.dict_item_group_ratings[i] for i in i_input])
        elif  mode==INTERCSHARED:
            the_user_group=np.array([self.dict_user_group_intercs[i] for i in u_input])
            the_item_group=np.array([self.dict_item_group_intercs[i] for i in i_input])           
        #
        elif mode==REVIEWSHARED:
            the_user_group=np.array([self.dict_user_group_reviews[i] for i in u_input])
            the_item_group=np.array([self.dict_user_group_reviews[i] for i in i_input])    

        else:
            assert "Singal shared do not support this mode"
        user_group_ratings=the_user_group
        user_group_intercs=the_user_group
        user_group_reviews=the_user_group

        item_group_ratings=the_item_group
        item_group_intercs=the_item_group
        item_group_reviews=the_item_group


        return (user_group_ratings,
                user_group_intercs,
                user_group_reviews,
                item_group_ratings,
                item_group_intercs,
                item_group_reviews,
                utext, itext, label)
    
    def eval_with_group_info(self,mode):
        u_input, i_input, label, utext, itext, text= super().eval()
        
        if mode == RATINGSHARED:
            the_user_group=np.array([self.dict_user_group_ratings[i] for i in u_input])
            the_item_group=np.array([self.dict_item_group_ratings[i] for i in i_input])
        elif  mode==INTERCSHARED:
            the_user_group=np.array([self.dict_user_group_intercs[i] for i in u_input])
            the_item_group=np.array([self.dict_item_group_intercs[i] for i in i_input])           
        #
        elif mode==REVIEWSHARED:
            the_user_group=np.array([self.dict_user_group_reviews[i] for i in u_input])
            the_item_group=np.array([self.dict_user_group_reviews[i] for i in i_input])    

        else:
            assert "Singal shared do not support this mode"
        user_group_ratings=the_user_group
        user_group_intercs=the_user_group
        user_group_reviews=the_user_group

        item_group_ratings=the_item_group
        item_group_intercs=the_item_group
        item_group_reviews=the_item_group


        return (user_group_ratings,
                user_group_intercs,
                user_group_reviews,
                item_group_ratings,
                item_group_intercs,
                item_group_reviews,
                utext, itext, label)