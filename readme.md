<h1 align="center">
A Two-tier Shared Embedding Method (TSE)
</h1>

<p align="center">
  <a href="#2-quick-start🚀">Quick Start</a> •
  <a href="https://dl.acm.org/doi/10.1145/3583780.3614770">Paper</a> •
  <a href="#3-citation☕️">Citation</a>
</p>



Repo for CIKM 2023 paper: [A Two-tier Shared Embedding Method for Review-based Recommender Systems](https://dl.acm.org/doi/10.1145/3583780.3614770)

## 1. Introduction✨

Reviews are valuable resources that have been widely researched
and used to improve the quality of recommendation services. 
Recent methods use multiple full embedding layers to model various
levels of individual preferences,increasing the risk of the data sparsity issue. Although it is a potential way to deal with this issue
that models homophily among users who have similar behaviors,
the existing approaches are implemented in a coarse-grained way.
They calculate user similarities by considering the homophily in
their global behaviors but ignore their local behaviors under a specific context. In this paper, we propose a two-tier shared embedding model (TSE), which fuses coarse- and fine-grained ways of
modeling homophily. It considers global behaviors to model homophily in a coarse-grained way, and the high-level feature in
the process of each user-item interaction to model homophily in
a fine-grained way. TSE designs a whole-to-part principle-based
process to fuse these ways in the review-based recommendation.
Experiments on five real-world datasets demonstrate that TSE significantly outperforms state-of-the-art models. It outperforms the
best baseline by 20.50% on the root-mean-square error (RMSE) and
23.96% on the mean absolute error (MAE), respectively.
## 2. Quick Start🚀

1. File Structure

```
.\amazon_data [training data]  
.\sim_res  [save the similarity matrixs]  
.\similarity  [similarity measures]  
.\untils  [some tools to papre training data]  
.\src  
```

2. calculate user similarities and item similarities in three levels

```sh
sudo python src/get_review_sim.py  
sudo python src/get_user_sim.py
```

> [Note!] We use seta to calculate the review sentiment scores.  
> The package of seta need to used at root user. 

3. training the model

```sh
python src/main.py
```





## 3. Citation☕️

If you find this repository helpful, please consider citing our paper:
```
@inproceedings{liu-ickm-2023-tse,
  author       = {Zhen Yang and
                  Junrui Liu and
                  Tong Li and
                  Di Wu and
                  Shiqiu Yang and
                  Huan Liu},
  editor       = {Ingo Frommholz and
                  Frank Hopfgartner and
                  Mark Lee and
                  Michael Oakes and
                  Mounia Lalmas and
                  Min Zhang and
                  Rodrygo L. T. Santos},
  title        = {A Two-tier Shared Embedding Method for Review-based Recommender Systems},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information
                  and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom,
                  October 21-25, 2023},
  pages        = {2928--2938},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3583780.3614770},
  doi          = {10.1145/3583780.3614770},
  timestamp    = {Wed, 22 Nov 2023 13:37:55 +0100},
}
```
