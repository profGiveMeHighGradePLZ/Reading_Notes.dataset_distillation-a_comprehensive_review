# dataset-distillation阅读笔记

# reference:
Yu, Ruonan, et al. “Dataset Distillation: A Comprehensive Review.” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 1, 1 Jan. 2024, pp. 150–170, https://doi.org/10.1109/tpami.2023.3323376. Accessed 18 Mar. 2024.


# What is dataset distillation

Dataset distillation (DD), also known as dataset condensation (DC), was introduced and has recently attracted much research attention in the community. Given an original dataset, DD aims to derive a much smaller dataset containing synthetic samples, based on which the trained models yield performance comparable with those trained on the original dataset.

![image](https://github.com/user-attachments/assets/7683316a-25f4-477d-a551-1065ac443f3c)

Dataset distillation (DD) or dataset condensation (DC) focuses on generating new training data for compression. It aims at synthesizing original datasets into a limited number of samples such that they are learned or optimized to represent the knowledge of original datasets.One of the essential goals of dataset distillation is to synthesize informative datasets to improve training efficiency, given a limited storage budget. In other words, as for the same limited storage, more information on the original dataset is expected to be preserved so that the model trained on condensed datasets can achieve comparable and satisfactory performance.

# Definition of Dataset Distillation 

The canonical dataset distillation problem involves learning a small set of synthetic data from an original large-scale dataset so that models trained on the synthetic dataset can perform comparably to those trained on the original. Formally, we formulate the problem as the following:

$$
S = \arg \min_S L(S, T)
$$


- $$T = (X_t, Y_t)$$ denoting a real dataset consisting of $|T|$ pairs of training images and
corresponding labels
  - $X_t \in \mathbb{R}^{N \times d}$, $N$ is the number of real samples, $d$ is the number of features
  - $Y_t \in \mathbb{R}^{N \times C}$, and $C$ is the number of output
entries
- $$S = (X_s, Y_s)$$ denoting the synthetic dataset
  - $$X_s \in \mathbb{R}^{M \times D}$$, $M$ is the number of synthetic samples
  - $$Y_s \in \mathbb{R}^{M \times C}$$, $$M \ll N$$,  and $D$ is the number of features
for each sample.
-  For the typical image classification tasks, $D$ is $height × width × channels$, and $y$ is a one-hot vector
whose dimension $C$ is the number of classes.

where $$L$$ is some objective for dataset distillation, which will be elaborated in the following contents.

# General workflow of dataset distillation
![image](https://github.com/user-attachments/assets/835b65f8-2ca9-4a9b-ad28-13aeee5808b3)


Two key steps in current DD methods are to train neural networks and compute $L$ via these networks. They perform alternately in an iterative loop to optimize synthetic datasets $S$, which formulates a general workflow for DD.

Firstly, $S$ is initialized before the optimization loop, which may have crucial impacts on the convergence and final performance of condensation methods. Typically, the synthetic dataset $S$ is usually initialized in two ways: randomly initialization e.g., from Gaussian noise, and randomly selected real samples from the original dataset $T$. Moreover, some coreset methods, e.g., K-Center, can also be applied to initialize the synthetic dataset.

To avoid overfitting problems and make the output synthetic dataset more generalizable, the network $θ$ will be periodically fetched at the beginning of the loop. The networks can be randomly initialized or loaded from some cached checkpoints. 

The fetched network is updated via $S$ or $T$ for some steps if needed. Then, the synthetic dataset is updated through some dataset distillation objective $L$ based on the network

#  Optimization Objectives in DD

Different works in DD propose different optimization objectives, denoted as $L$, to obtain synthetic datasets that are valid to replace original ones in downstream training.

## 1, Performance Matching

### Meta-Learning Based Methods. 

This approach aims to optimize a synthetic dataset such that neural networks trained on it could have the lowest loss on the original dataset, and thus the performance of models trained by synthetic and real datasets is matched:


$$
L(S, T) = \mathbb{E}_{\theta^{(0)} \sim \Theta} [l(T; \theta^{(T)})],
$$

$$
\theta^{(t)} = \theta^{(t-1)} - \eta \nabla l(S; \theta^{(t-1)})
$$



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

### Peak GPU Memory Usage.

- DM requires the lowest GPU memory, while MTT requires the most.
- When IPC is only 50, the operation of MTT encounters out-of-memory
- With the gradual growth of IPC, JAX version FRePo reveals the advantages of memory efficiency.
- The DD and PyTorch versions of FRePo require relatively large GPU memory. The former needs to go through the entire training graph during backpropagation, while the latter adopts a wider network.
- There is no significant difference in the GPU memory required by DC and DSA, indicating that DSA augmentation is memory-friendly

## Conclusions from the above experiments:

- The performance of DD has significantly improved thanks to the momentum item. However, DD has relatively poor generalizability, and the required runtime and GPU memory are considerable.
- Compared with DC and DSA, adopting DSA can significantly improve the distillation performance and generalizability while not increasing the running time and memory requirement too much.
- DM does not perform as well as other methods. However, DM has relatively good generalizability and significant advantages regarding run time and GPU memory requirements.
- MTT has the overall second-best performance but requires much running time and space due to unrolled gradient computation through backpropagation.
- FRePo achieves SOTA performance when IPC are small, regardless of performance or training cost. However, as IPC increases, FRePo does not have comparable performance.

# CHALLENGES AND POSSIBLE IMPROVEMENTS

## Computational Cost

Generating the synthetic dataset is typically expensive and that the required time will rapidly increase when distilling large-scale datasets. This high computational cost is mainly caused by backpropagating through unrolled computational graphs for updating synthetic datasets.

## Scaling Up

the scaling-up problem of dataset distillation is reflected in three aspects: hard to obtain an informative synthetic dataset with a larger IPC; challenging to perform the dataset distillation algorithm on a large-scale dataset such as ImageNet; not easy to adapt to a large, complex model.

### Larger Compression Ratios

As the synthetic dataset increases, the ability of the dataset to compress knowledge of the real dataset does not increase significantly, even
though the memory efficiency of some methods can support larger IPC. In other words, many existing dataset distillation methods are not suitable for high compression ratios.

### Larger Original Datasets

Many existing dataset distillation methods are challenging to apply to large-scale datasets due to high GPU memory requirements and extensive GPU hours

### Larger Networks

Larger and more complex network structures require more GPU memory and runtime to compute and store the gradients during the dataset distillation process. At the same time when the generated dataset is evaluated on complex networks, its performance is significantly lower
than that of simple networks. As the size of the synthetic dataset is small, complex networks are prone to overfitting problems trained on the small training dataset.

## Generalization across Different Architectures

The generalizability of the synthetic dataset across different network architectures is an essential standard to evaluate the performance of a DD method, as it indicates availability in practice. The evaluation results of the synthetic data generated by the existing DD methods on heterogeneous models have not reached homogeneous performance.

## Design for Other Tasks and Applications

The existing DD methods mainly focus on the classification task. Here, we expect future works to apply DD and conduct sophisticated designing in more tasks in computer vision.

## Security and Privacy
Existing works focus on improving the methods to generate more informative synthetic datasets while neglecting the potential security and privacy of DD.



