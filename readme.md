# Introduction
Repo for CIKM 2023 paper: [A Two-tier Shared Embedding Method for Review-based Recommender Systems](https://dl.acm.org/doi/10.1145/3583780.3614770)

## 1. File Structure

> amazon_data [training data]  
> sim_res  [save the similarity matrixs]  
> similarity  [similarity measures]  
> untils  [some tools to papre training data]  
> src  


## 2. Quick Start
1. calculate user similarities and item similarities in three levels


>+ sudo python src/get_review_sim.py  
>+ sudo python src/get_user_sim.py

> [Note!] We use seta to calculate the review sentiment scores.  
> The package of seta need to used at root user. 

2. training the model
> python src/main.py




## Citation

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
  biburl       = {https://dblp.org/rec/conf/cikm/0004LLWYL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
