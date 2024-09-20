# dataset-distillation阅读笔记

# reference:
Yu, Ruonan, et al. “Dataset Distillation: A Comprehensive Review.” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 1, 1 Jan. 2024, pp. 150–170, https://doi.org/10.1109/tpami.2023.3323376. Accessed 18 Mar. 2024.


## what is dataset distillation
dataset distillation (DD), also known as dataset condensation (DC), was introduced and has recently attracted much research attention in the community. Given an original dataset, DD aims to derive a much smaller dataset containing synthetic samples, based on which the trained models yield performance comparable with those trained on the original dataset.


![image](https://github.com/user-attachments/assets/7683316a-25f4-477d-a551-1065ac443f3c)

A relatively straightforward method, known as core-set or instance selection,to obtain smaller datasets is to select the most representative or valuable samples from original datasets so that models trained on these subsets can achieve as good performance as the original ones.

Another line of work that focuses on generating new training data for compression has been proposed, known as dataset distillation (DD) or dataset condensation (DC). It aims at synthesizing original datasets into a limited number of samples such that they are learned or optimized to represent the knowledge of original datasets.

## General workflow of dataset distillation
![image](https://github.com/user-attachments/assets/835b65f8-2ca9-4a9b-ad28-13aeee5808b3)

Two key steps in current DD methods are to train neural networks and compute L via these networks. They perform alternately in an iterative loop to optimize synthetic datasets S, which formulates ageneral workflow for DD as shown in Alg. 1

Firstly, S is initialized before the optimization loop,which may have crucial impacts on the convergence and final performance of condensation methods. Typically,the synthetic dataset S is usually initialized in two ways:randomly initialization e.g., from Gaussian noise, and randomly selected real samples from the original dataset T .Moreover, some coreset methods, e.g., K-Center, can also be applied to initialize the synthetic dataset.

To avoid overfitting problems and make the output synthetic dataset more generalizable, the network θ will be periodically fetched at the beginning of the loop.The networks can be randomly initialized or loaded from some cached checkpoints.

The fetched network is updated via Sor T for some steps if needed. Then, the synthetic dataset is updated through some dataset distillation objective L based on the network

##  Optimization Objectives in DD

### Performance Matching

Performance matching aims to optimize a synthetic dataset such that neural networks trained on it could have the lowest loss on the original dataset, and thus the performance of models trained by synthetic and real datasets is matched

#### Meta Learning Based Methods. 







