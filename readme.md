<h1 align="center">
A Multi-task Method for Review-based Recommendation
</h1>

<p align="center">
  <a href="#2-quick-start🚀">Quick Start</a>
</p>

Repo for the paper: A Multi-task Method for Review-based Recommendation

## 1. Introduction✨

Reviews are a form of user feedback on item features, which can help recommender systems understand user preferences and product features. Existing review-based methods utilize deep neural networks to extract high-level features of reviews, and then implicitly model user-item interactions based on the features and ID embeddings. However, this implicit method lacks supervision, so the final learning result can only learn part of the interaction features. 

In this paper, we propose a multi-task factorization sharing method (MFS) for review-based recommendations, which considers the inherent relationship between supervision information of interactive features and rating features, improving the model performance through multi-task learning. MFS uses an individual expert model for each task to learn special factorization features from reviews and a shared expert model to learn the common feature. To make predictions, the method combines linear, shared, and high-level relationships between two task features. 

To deal with the challenges of training data, we design a data constructor that samples negative samples for the CTR task and completes their rating labels for the rating prediction task. Experiments on five real-world datasets demonstrate that the predictions of MTP significantly outperform baselines in multi-tasks.

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
