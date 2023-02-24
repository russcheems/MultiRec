# 说明
This is the source code for 《》

## 1. file structure

> amazon_data [training data]  
> sim_res  [save the similarity matrixs]  
> similarity  [similarity measures]  
> untils  [some tools to papre training data]  
> src  


## 2. run example
1. calculate user similarities and item similarities in three levels


>+ sudo python src/get_review_sim.py  
>+ sudo python src/get_user_sim.py

> [Note!] We use seta to calculate the review sentiment scores.  
> The package of seta need to used at root user. 

2. training the model
> python src/main.py



