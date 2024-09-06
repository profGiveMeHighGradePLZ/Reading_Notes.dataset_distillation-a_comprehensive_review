# dataset-distillation

## what is dataset distillation
dataset distillation (DD), also known as dataset condensation (DC), was introduced and has recently attracted much research attention in the community. Given an original dataset, DD aims to derive a much smaller dataset containing synthetic samples, based on which the trained models yield performance comparable with those trained on the original dataset.


![image](https://github.com/user-attachments/assets/7683316a-25f4-477d-a551-1065ac443f3c)

A relatively straightforward method, known as core-set or instance selection,to obtain smaller datasets is to select the most representative or valuable samples from original datasets so that models trained on these subsets can achieve as good performance as the original ones.

another line of work that focuses on generating new training data for compression has been proposed, known as dataset distillation (DD) or dataset condensation (DC). It aims at synthesizing original datasets into a limited number of samples such that they are learned or optimized to represent the knowledge of original datasets.


# reference:
Yu, Ruonan, et al. “Dataset Distillation: A Comprehensive Review.” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 1, 1 Jan. 2024, pp. 150–170, https://doi.org/10.1109/tpami.2023.3323376. Accessed 18 Mar. 2024.
