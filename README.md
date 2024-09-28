# dataset-distillation阅读笔记

# reference:
Yu, Ruonan, et al. “Dataset Distillation: A Comprehensive Review.” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 1, 1 Jan. 2024, pp. 150–170, https://doi.org/10.1109/tpami.2023.3323376. Accessed 18 Mar. 2024.


# What is dataset distillation

Dataset distillation (DD), also known as dataset condensation (DC), was introduced and has recently attracted much research attention in the community. Given an original dataset, DD aims to derive a much smaller dataset containing synthetic samples, based on which the trained models yield performance comparable with those trained on the original dataset.

![image](https://github.com/user-attachments/assets/7683316a-25f4-477d-a551-1065ac443f3c)

A relatively straightforward method, known as core-set or instance selection, to obtain smaller datasets is to select the most representative or valuable samples from original datasets so that models trained on these subsets can perform as well as the original ones.

Another line of work that focuses on generating new training data for compression has been proposed, known as dataset distillation (DD) or dataset condensation (DC). It aims at synthesizing original datasets into a limited number of samples such that they are learned or optimized to represent the knowledge of original datasets.

One of the essential goals of dataset distillation is to synthesize informative datasets to improve training efficiency, given a limited storage budget. In other words, as for the same limited storage, more information on the original dataset is expected to be preserved so that the model trained on condensed datasets can achieve comparable and satisfactory performance.

# General workflow of dataset distillation
![image](https://github.com/user-attachments/assets/835b65f8-2ca9-4a9b-ad28-13aeee5808b3)

Two key steps in current DD methods are to train neural networks and compute L via these networks. They perform alternately in an iterative loop to optimize synthetic datasets S, which formulates a general workflow for DD as shown in Alg. 1

Firstly, S is initialized before the optimization loop, which may have crucial impacts on the convergence and final performance of condensation methods. Typically, the synthetic dataset S is usually initialized in two ways: randomly initialization e.g., from Gaussian noise, and randomly selected real samples from the original dataset T. Moreover, some coreset methods, e.g., K-Center, can also be applied to initialize the synthetic dataset.

To avoid overfitting problems and make the output synthetic dataset more generalizable, the network θ will be periodically fetched at the beginning of the loop. The networks can be randomly initialized or loaded from some cached checkpoints.

The fetched network is updated via Sor T for some steps if needed. Then, the synthetic dataset is updated through some dataset distillation objective L based on the network

#  Optimization Objectives in DD

## 1, Performance Matching

### Meta-Learning Based Methods. 

This approach aims to optimize a synthetic dataset such that neural networks trained on it could have the lowest loss on the original dataset, and thus the performance of models trained by synthetic and real datasets is matched.

![image](https://github.com/user-attachments/assets/f530476c-5583-4958-9a13-6a4a6fc07a85)

The objective of performance matching indicates a bilevel optimization algorithm: in inner loops, weights of a differentiable model with parameter θ are updated with S via gradient decent, and the recursive computation graph is cached; in outer loops, models trained after inner loops are validated on T and the validation loss is backpropagated through the unrolled computation graph to S.

 ### Kernel Ridge Regression Based Methods.

For the above method, outer optimization steps are computationally expensive, and the GPU memory required is proportional to the number of inner loops. Thus, the number of inner loops is limited, which results in insufficient inner optimization and bottlenecks in the performance. It is also inconvenient for this routine to be scaled up to large models

a class of methods based on kernel ridge regression (KRR) tackle this problem, which performs convex optimization and results in a closed-form solution for the linear model which avoids extensive inner loop training.

## 2, Parameter Matching
the key idea of parameter matching is to train the same network using synthetic datasets and original datasets for some steps, respectively, and encourage the consistency of their trained neural parameters. According to the number of training steps using S and T, parameter matching methods can be further divided into two streams: single-step parameter matching and multi-step parameter matching.

### Single-Step Parameter Matching

![image](https://github.com/user-attachments/assets/d05641c6-1c6d-463f-8f62-2dde1ebf8d46)

In single-step parameter matching, a network is updated using S and T for only 1 step, respectively, and their resultant gradients respective to θ are encouraged to be consistent, which is also known as gradient matching. After each step of updating synthetic data, the network used for computing gradients is trained on S for T steps.

This approach is memory-efficient compared with meta-learning-based performance matching. This method has some limitations, e.g., the distance metric between two gradients considers each class independently and ignores relationships underlain for different classes. Thus, class-discriminative features are largely neglected. Besides, since only a single-step gradient is matched, errors may be accumulated in evaluation where models are updated by synthetic data for multiple steps.

### Multi-Step Parameter Matching. 

![image](https://github.com/user-attachments/assets/d079378e-49da-4678-b1c9-9d53e060bb4c)

multi-step parameter matching approach, known as matching training trajectory (MTT), is proposed to remedy the accumulative error during evaluation in the Single-Step Parameter Matching. In this method, θ will be initialized and sampled from checkpoints of training trajectories on original datasets.

starting from θ(0), the algorithm trains the model on synthetic datasets for Ts steps and the original dataset for Tt steps, respectively, where
Ts and Tt are hyperparameters, and tries to minimize the distance of the endings of these two trajectories, i.e., θ(Ts)S and θ(Tt)T

Note that the total loss is normalized by the distance between the starting point θ(0) and the expert endpoint θ(Tt)T. This normalization helps get a strong signal where the expert does not move as much at later training epochs and self-calibrates the magnitude difference across neurons
and layers. It is demonstrated that this multi-step parameter matching strategy yields better performance than the single-step counterpart.

## 3, Distribution Matching

The distribution matching approach aims to obtain synthetic data whose distribution can approximate that of real data. Instead of matching training effects, e.g., the performance of models trained on S, distribution matching directly optimizes the distance between the two distributions using some metrics, e.g., Maximum Mean Discrepancy (MMD)

![image](https://github.com/user-attachments/assets/1d974f90-ba31-464b-a68b-1f40c9a80ee1)

Since directly estimating the real data distribution can be expensive and inaccurate as images are high-dimensional data, distribution matching adopts a set of embedding functions, i.e., neural networks, each providing a partial interpretation of the input and their combination providing a comprehensive interpretation, to approximate MMD.

# EXPERIMENTS

Two sets of quantitative experiments, i.e., performance and training cost evaluation are conducted on representative dataset distillation methods that cover three classes of primary condensation metrics, including DD, DC, DSA, DM, MTT, and FRePo.

## Experimental Setup

### Datasets

Five datasets widely used as benchmarks in existing dataset distillation works, including MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet, are adopted.

### Networks

1, using the default ConvNet architecture provided by the authors, which mainly consists of multiple Conv-ReLU-AvgPooling blocks. 

2, evaluating the performance of synthetic datasets across different architectures: 

- ConvNet with no normalization layer

- AlexNet with no normalization layer and instance normalization layer

- ResNet with instance normalization layer and batch normalization layer

- VGG with instance normalization layer and batch normalization layer

### Evaluation Protocol

- For performance evaluation:

1, first generating synthetic datasets through candidate methods and train target networks using these datasets. 

2, evaluating the performance of trained models by the corresponding test set of the original dataset. 

- For training cost evaluation:

1, all methods are evaluated under full batch training for fair comparisons, and adopt the default data augmentation strategies the authors provided for distillation performance evaluation. 

2, for generalization evaluation, the DSA data augmentation is adopted in the evaluation model training process for fair comparisons.

3, we measure the distillation performance of condensed datasets with 1, 10, and 50 images per class (IPC) for different datasets. The cross-architecture experiments are conducted on CIFAR-10 with 10 IPC. Efficiency evaluation experiments are conducted on the CIFAR-10 dataset with a wide range of IPC.

## Performance Evaluation

### Distillation Performance

- In most cases, FRePo achieves state-ofthe-art performance, especially for more complex datasets, e.g., CIFAR-10, CIFAR-100, and Tiny-ImageNet.
- MTT often achieves second-best performance.
- Comparing the performance results of DC and DSA, although dataset augmentation cannot guarantee to benefit the distillation performance, it influences the results significantly at most times.
- As for DD, it obtains the SOTA performance in the case of MNIST 1 IPC, Fashion-MNIST 1 IPC, and 10 IPC.
- The performance of DM is often not as good as other methods

### Cross Architecture Generalization

- the instance normalization layer seems to be a vital ingredient in several methods (DD, DC, DSA, DM, and MTT), which may be harmful to the transferability.
- The performance degrades significantly for most methods except FRePo when no normalization is adopted (Conv-NN, AlexNet-NN).
- If the normalization layer is inconsistent with the training architecture, the test results will drop significantly.
- As for DD, the distilled data highly rely on the training model, and thus the performance degrades significantly on heterogeneous networks with different normalization layers, e.g., batch normalization.
- Comparing the transferability of DC and DSA, we find that DSA data augmentation can help train models with different architectures, especially those with batch normalization layers.

## Training Cost Evaluation

### Run-Time Evaluation.

- As for DD and MTT, the process of network updating is unrolled, and thus the required time for the entire outer loop and the loss function is the same.
- as DM does not update the network, the required time for these two processes is the same.
- DD requires a significantly longer run time to update the synthetic dataset for a single iteration, as the gradient computation for the performance matching loss over the synthetic dataset involves bi-level optimization.
- DC, DSA, and FRePo implemented by PyTorch have similar required run times. Comparing DSA and DC, using DSA augmentation makes the running time slightly longer, but it can significantly improve the performance.
- When the IPC is small, FRePo (both the JAX and PyTorch versions) is significantly faster than other methods, but as the IPC increases, the PyTorch version is similar to the second efficient echelon methods, while the JAX version is close to DM.
- MTT runs the second slowest, but there is no more data to analyze due to out-of-memory.
