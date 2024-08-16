<h1 align="center">
A Multi-task Factor Sharing Method for Review-based Recommendation
</h1>

<p align="center">
  <a href="#2-quick-start🚀">Quick Start</a>
</p>

Repo for the paper: A Multi-task Factor Sharing Method for Review-based Recommendation

## 1. Introduction✨

 Reviews are a form of user feedback on item features, which
 play a crucial role in helping recommender systems under
stand user preferences. Existing review-based methods typi
cally utilize deep neural networks to extract high-level fea
tures of reviews, and then implicitly model user-item inter
actions using these features along with ID embeddings. How
ever, this implicit approach lacks supervision, which often re
sults in the model capturing only partial interaction features.
 Moreover, these methods tend to focus on achieving a single
 objective at a time—either optimizing for the click-through
 rate prediction task to satisfy user requirements or focusing
 on the rating prediction task to capture user interests. As a
 result, they overlook the deep connections between these two
 goals. In this paper, we propose a Multi-task Factor Sharing
 method (MFS) for review-based recommendations. It lever
ages the inherent relationship between supervision informa
tion of interaction features and rating features, enhancing the
 model performance through multi-task learning. MFS uses an
 individual expert model for each task to learn special factors
 from reviews and a shared expert model to learn the com
mon factors. To make predictions, the method combines lin
ear, shared, and high-level relationships between factors of
 two tasks. To address the challenges of different tasks requir
ing distinct training data, we design a data constructor that
 samples negative samples for the CTR task and generates
 corresponding rating labels. Experiments on five real-world
 datasets demonstrate that MFS outperforms the best baseline
 by 9.19%, 9.80%, 0.69%, 7.95%, 1.92% in terms of Accu
racy, Precision, Recall, F1-score, and MAE, respectively.

## 2. Quick Start🚀

1. File Structure

```
.\amazon_data [training data]  
.\sim_res  [save the similarity matrices]  
.\similarity  [similarity measures]  
.\utils  [some tools to prepare training data]  
.\src  
```

2. Calculate user similarities and item similarities in three levels

```sh
sudo python src/get_review_sim.py  
sudo python src/get_user_sim.py
```

> [Note!] We use seta to calculate the review sentiment scores.  
> The package of seta needs to be used by the root user. 

3. Train the model

```sh
python src/main.py
```
