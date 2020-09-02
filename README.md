# Deep-learning-programs
In this repository, there are several jupyter notebook files. Each file is independent.
These files are all deep learning related researches that include not only unsupervised learning, supervised learning, even reinforcement learning.
Each file has a brief introduction in follow section.

## Neural process 
Neural process tries to use deep learning model to learn a mapping function like a gaussian process. In Neural_Process.ipynb, i implemented the idea of this paper. Here, the input is part of a image with some pixels masked. The mapping function should learns to predict the pixel by given its location and the context which is the existed pixel.

Here, two samples are shown for this implementation.
First image is the context, second one is the prediction, and third image is the ground-truth.

![](https://i.imgur.com/PXBboXj.png)

![](https://i.imgur.com/nA1l56f.png)



Reference [Neural Process](https://arxiv.org/pdf/1807.01622.pdf)

## VAE - Variatioinal autoencoder
VAE is a famous variatioinal or stochastic model that learns latent variables in given data. That uses variational inference to estimate the approximate posterior. And the overall learning criteria is evidence lower bound(ELBO).

Reference [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)


## IntroVAE - Introspective variational autoencoder
IntroVAE is a variant of a standard VAE. In previous implementation of VAE, the result of the prediction is usually blurring. To overcome this issue, IntroVAE uses a special learning scheme like adversarial learning that enable the encoder to encode a more accurate latent variables to generate a real-like data and decoder try to make the real-like data far away from the prior distribution. This interaction can enable encoder and decoder to know more about how to generate a more precise result. 

Reference [IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis](https://arxiv.org/pdf/1807.06358.pdf)


## RealNVP 
RealNVP is a flow-based model that exactly evaluate the log-likelihood instead of the evidence lower bound of the given data. Flow-based model is an invertiable mapping function. In this implementation, i use this model to transform a gaussian distribution into a two-moons distribution. 
Here, a sample is shown for this implementation.
The z denotes the random varables of standard gaussian and x denotes the random variable of two-moons distribution.
The estimation means the prediction of the flow-based model. For example Estimation p(x) means the flow-based model transforms z into the x. 

![](https://i.imgur.com/wFfHyQN.png)

Reference [DENSITY ESTIMATION USING REALNVP](https://arxiv.org/pdf/1605.08803.pdf)

## VAE-NF - Variational autoencoder with normalize flow
This is the combination of a VAE and a flow-based model. VAE estimates a approximate posterior distribution of the latent variables. In order to fit the real data distirubiton, flow-based model can help to transform this posterior distribution before computing the KL-divergence between the pre-defined prior distribution (usually gaussian)

Reference [Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770.pdf)



## CapsNet - Capsule network
Standard deep learning model extracts high-dimensional vector which is called hidden features but the value in each dimension can not store enough information to represent the data especially for cnn. For example, in human face classification task, the hidden features only represent if the particular feature existed in the data like a nose, eyes. But those values in the hidden features doesn't contain the position or rotation information. To enable the network can capture this information, capsule network use a vector of vector for hidden features. In short, original high-dimensional vector becomes high-dimensional matrix.

Reference [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)

## Styletransfer
In fact, this is an assignment of CS231 online course. 
A generated result.

![](https://i.imgur.com/YfOAJLU.png)

Reference [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)


## RL_Model_Test
This file contains several different Reinforcement learning algorithms like
*  Deep Q network 
*  Deep Q network with priority replay
*  Policy Gradient 
*  Actor-Critic

I use the openAI-gym environment to test the these algorithms.



