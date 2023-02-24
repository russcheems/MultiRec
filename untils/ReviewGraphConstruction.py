import pickle
from re import U
import numpy as np
# import networkx as nx
from tqdm  import tqdm
from collections import defaultdict


def unique_edge(a):
    """对收集到的word-word边进行唯一化
        """
    temp = ["_".join(map(str, x)) for x in a
            ]
    temp = list(set(temp))
    b = [
        list(map(int, x.split("_"))) for x in temp
    ]

    return b


class ReviewGraphConstruction:
    def __init__(self) -> None:
        pass
    def _build_a_word_graph(self,reviews,window_size=3):
        # 构建词网络
        graph=[]
        for one_review in reviews:
            for idx,w in enumerate(one_review):
                # 循环边
                graph.append([w,2,w])
                for i in range(1,window_size+1):
                    if idx+i>=len(one_review):
                        break
                    # 前向边
                    graph.append([w, 0, one_review[idx+i]])
                    graph.append([one_review[idx+i], 1, w])


        graph=unique_edge(graph)
        return graph
    def build_word_nets(self,texts,u_texts,i_texts):
        """
            texts:[[w1,w2,...],...]     记录了评论中的单词
            u_texts:[[t1,t2,...],...]   记录了所有用户的评论
            i_texts:[[t1,t2,...],...]   记录了所有物品的评论
        """
        self.u_nets=self._build_one_side_word_net(texts, u_texts)
        self.i_nets=self._build_one_side_word_net(texts, i_texts)
        return self.u_nets,self.i_nets

    def _build_one_side_word_net(self, texts, u_texts):
        u_nets=[]
        # 遍历每一个用户
        for u in tqdm(u_texts,ncols=100):
            reviews=[]
            for i in u:
                if i==0:
                    continue
                else:
                    reviews.append(texts[int(i)])
            u_nets.append(self._build_a_word_graph(reviews))
        return u_nets




def test_unique():
    a=ReviewGraphConstruction()
    data=[[1,2],[1,3],[1,2],[1,2,3],[3,2,1]]
    print(unique_edge(data))

def test_ReviewGraphConstruction():
    a=ReviewGraphConstruction()
    with open("amazon_data/Musical_Instruments_5/data.pkl","rb") as f:
        data=pickle.load(f)

    l_users,l_items=papre_data(data)
    a.build_word_nets(data["vec_texts"],l_users,l_items)


def papre_data(review_data):
    uit=review_data["vec_uit"]
    users=defaultdict(list)
    items=defaultdict(list)
    for one in uit:
        u,i,re,ra=one
        users[u].append(re)
        items[i].append(re)
    max_u=max(users.keys())
    max_i=max(items.keys())
    l_users=[users[i] for i in range(max_u+1)]
    l_items=[items[i] for i in range(max_i+1)]
    return l_users,l_items

if __name__=="__main__":

    # test_unique()
    test_ReviewGraphConstruction()
