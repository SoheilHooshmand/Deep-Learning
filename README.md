# Deep Learning 

A comprehensive hands-on tutorial for learning **Deep Learning** from fundamental neural networks to advanced architectures and techniques.

This repository contains theoretical explanations, practical implementations, experiments, and Jupyter Notebooks covering the major concepts used in modern Deep Learning.

The goal of this repository is to provide a structured learning path that starts from the fundamentals and gradually moves toward advanced topics such as CNNs, RNNs, Transformers, Attention, Autoencoders, GANs, Diffusion Models, Transfer Learning, and Encoder-Decoder architectures.

---

## 📚 Contents

* [Prerequisites](#-prerequisites)
* [Installation](#-installation)
* [Repository Structure](#-repository-structure)
* [Learning Path](#-learning-path)

  * [1. Basics](#1-basics)
  * [2. Components of Neural Networks](#2-components-of-neural-networks)
  * [3. Optimization Algorithms](#3-optimization-algorithms)
  * [4. Regularization and Hyperparameter Tuning](#4-regularization-and-hyperparameter-tuning)
  * [5. Convolutional Neural Networks](#5-convolutional-neural-networks)
  * [6. Recurrent Neural Networks](#6-recurrent-neural-networks)
  * [7. Advanced Deep Learning Techniques](#7-advanced-deep-learning-techniques)
* [Datasets](#-datasets)
* [Frameworks and Libraries](#-frameworks-and-libraries)
* [How to Use](#-how-to-use)
* [Recommended Learning Order](#-recommended-learning-order)
* [Project Goals](#-project-goals)

---

## 🔧 Prerequisites

Before starting this tutorial, it is recommended to have a basic understanding of:

* Python programming
* Linear Algebra
* Calculus
* Probability and Statistics
* Basic Machine Learning concepts
* NumPy
* Pandas
* Matplotlib

Basic knowledge of Machine Learning algorithms such as Linear Regression, Logistic Regression, Decision Trees, and Gradient Descent is also helpful.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/DeepLearning.git
cd DeepLearning
```

Create a virtual environment:

```bash
python3 -m venv myvenv
```

Activate the virtual environment:

### Linux / macOS

```bash
source myvenv/bin/activate
```

### Windows

```bash
myvenv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

# 📂 Repository Structure

```text
DeepLearning/
│
├── Basics/
│   └── Neural_Network.ipynb
│
├── Components_of_Neural_Networks/
│   ├── Forward_Propagation.ipynb
│   └── Back_Propagation.ipynb
│
├── Optimization_Algorithm/
│   ├── Gradient_Descent.ipynb
│   ├── Stochastic_Gradient_Descent.ipynb
│   ├── Mini_Batch_Gradient_Descent.ipynb
│   ├── Momentum_based_Gradient_Optimizer.ipynb
│   ├── Adagrad_Optimizer.ipynb
│   ├── RMSProp_Optimizer.ipynb
│   └── Adam_Optimizer.ipynb
│
├── Regularization_and_HyperparameterTuning/
│   ├── Regularization.ipynb
│   ├── Dropout.ipynb
│   ├── Batch_Normalization.ipynb
│   ├── Early_Stopping.ipynb
│   ├── Epoch.ipynb
│   └── Hyperparameter_Tuning.ipynb
│
├── Convolution_Neural_Network/
│   ├── CNN.ipynb
│   ├── CNN_PyTorch.ipynb
│   ├── CNN_Tensorflow.ipynb
│   ├── Image_Classification.ipynb
│   ├── Pooling.ipynb
│   ├── LeNet-5_Architecture.ipynb
│   ├── AlexNet_Architecture.ipynb
│   ├── Mobilenet_V2_Architecture.ipynb
│   ├── ResNet.ipynb
│   └── U-Net_Architecture.ipynb
│
├── Recurrent_Neural_Networks/
│   ├── simple_RNN.ipynb
│   ├── Training_of_RNN.ipynb
│   ├── Vanishing_and_Exploding_Gradients.ipynb
│   ├── LSTM.ipynb
│   ├── GRU.ipynb
│   ├── Bidirectional_RNN.ipynb
│   └── BiLSTM.ipynb
│
├── Advanced_Deep_Learning_Techniques/
│   ├── Attention_Mechanism.ipynb
│   ├── seq2seq_Model.ipynb
│   ├── Encoder_Decoder_Models.ipynb
│   ├── Transfer_Learning.ipynb
│   ├── Autoencoders.ipynb
│   ├── Types_of_Autoencoders.ipynb
│   ├── Generative_Adversarial_Network.ipynb
│   └── Diffusion_Models.ipynb
│
├── requirements.txt
└── README.md
```

---

# 🧠 Learning Path

## 1. Basics

The tutorial starts with the fundamental concepts of Artificial Neural Networks.

### Topics

* Neural Networks
* Neurons
* Weights and Biases
* Activation Functions
* Forward Propagation
* Loss Functions
* Backpropagation
* Gradient Descent
* Training Neural Networks

### Notebook

`Basics/Neural_Network.ipynb`

This notebook introduces the basic architecture and operation of a neural network and provides a practical implementation.

---

# 2. Components of Neural Networks

This section focuses on the two fundamental processes used during neural network training.

## Forward Propagation

`Components_of_Neural_Networks/Forward_Propagation.ipynb`

Covers:

* Input layer
* Hidden layers
* Output layer
* Weighted sums
* Activation functions
* Prediction generation

### Backpropagation

`Components_of_Neural_Networks/Back_Propagation.ipynb`

Covers:

* Loss calculation
* Gradients
* Chain rule
* Gradient propagation
* Weight updates
* Bias updates

---

# 3. Optimization Algorithms

Optimization algorithms determine how neural network parameters are updated during training.

## Topics

| Algorithm                   | Notebook                                  |
| --------------------------- | ----------------------------------------- |
| Gradient Descent            | `Gradient_Descent.ipynb`                  |
| Stochastic Gradient Descent | `Stochastic_Gradient_Descent.ipynb`       |
| Mini-Batch Gradient Descent | `Mini_Batch_Gradient_Descent.ipynb`       |
| Momentum                    | `Momentum_based_Gradient_Optimizer.ipynb` |
| AdaGrad                     | `Adagrad_Optimizer.ipynb`                 |
| RMSProp                     | `RMSProp_Optimizer.ipynb`                 |
| Adam                        | `Adam_Optimizer.ipynb`                    |

These notebooks explain how different optimizers improve convergence, training stability, and learning speed.

---

# 4. Regularization and Hyperparameter Tuning

Deep neural networks can easily overfit training data. This section covers techniques for improving generalization and controlling the training process.

## Topics

* Regularization
* L1 and L2 Regularization
* Dropout
* Batch Normalization
* Early Stopping
* Epochs
* Hyperparameter Tuning

### Notebooks

```text
Regularization_and_HyperparameterTuning/
│
├── Regularization.ipynb
├── Dropout.ipynb
├── Batch_Normalization.ipynb
├── Early_Stopping.ipynb
├── Epoch.ipynb
└── Hyperparameter_Tuning.ipynb
```

---

# 5. Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are widely used for computer vision and image-based tasks.

This section covers CNN fundamentals and several important CNN architectures.

## Core CNN Concepts

* Convolution
* Filters / Kernels
* Feature Maps
* Stride
* Padding
* Pooling
* ReLU
* Fully Connected Layers
* Image Classification

### Notebooks

```text
Convolution_Neural_Network/
│
├── CNN.ipynb
├── CNN_PyTorch.ipynb
├── CNN_Tensorflow.ipynb
├── Image_Classification.ipynb
└── Pooling.ipynb
```

---

## CNN Architectures

### LeNet-5

`LeNet-5_Architecture.ipynb`

An early and influential CNN architecture originally designed for handwritten digit recognition.

### AlexNet

`AlexNet_Architecture.ipynb`

Introduces a deeper CNN architecture that played a major role in the development of modern deep learning for computer vision.

### MobileNetV2

`Mobilenet_V2_Architecture.ipynb`

A lightweight CNN architecture designed for efficient inference on devices with limited computational resources.

### ResNet

`ResNet.ipynb`

Covers Residual Networks and the idea of **skip connections**, which help train very deep neural networks.

### U-Net

`U-Net_Architecture.ipynb`

An encoder-decoder CNN architecture widely used for image segmentation.

---

# 6. Recurrent Neural Networks

Recurrent Neural Networks are designed for sequential and temporal data.

This section focuses on how neural networks process sequences and how different architectures address the limitations of standard RNNs.

## Topics

* Recurrent Neural Networks
* Hidden States
* Sequence Processing
* Backpropagation Through Time
* Vanishing Gradients
* Exploding Gradients
* LSTM
* GRU
* Bidirectional RNN
* Bidirectional LSTM

### Notebooks

```text
Recurrent_Neural_Networks/
│
├── simple_RNN.ipynb
├── Training_of_RNN.ipynb
├── Vanishing_and_Exploding_Gradients.ipynb
├── LSTM.ipynb
├── GRU.ipynb
├── Bidirectional_RNN.ipynb
└── BiLSTM.ipynb
```

The notebooks also include practical experiments using datasets such as IMDB and review datasets.

---

# 7. Advanced Deep Learning Techniques

This section introduces more advanced architectures and techniques used in modern Deep Learning.

---

## Attention Mechanism

`Attention_Mechanism.ipynb`

Explains how a neural network can dynamically focus on the most relevant parts of an input sequence.

Key concepts include:

* Query
* Key
* Value
* Attention Scores
* Attention Weights
* Context Vectors

---

## Sequence-to-Sequence

`seq2seq_Model.ipynb`

Introduces Sequence-to-Sequence models for tasks such as:

* Machine Translation
* Text Generation
* Sequence Transformation

---

## Encoder-Decoder Models

`Encoder_Decoder_Models.ipynb`

Explores the Encoder-Decoder architecture and how it is used to transform one sequence representation into another.

---

## Transfer Learning

`Transfer_Learning.ipynb`

Explains how pretrained neural networks can be reused for new tasks.

Topics include:

* Pretrained Models
* Feature Extraction
* Fine-Tuning
* Transfer Learning for Computer Vision

---

## Autoencoders

`Autoencoders.ipynb`

Introduces neural networks that learn compressed representations of data.

Topics include:

* Encoder
* Latent Representation
* Decoder
* Reconstruction
* Reconstruction Loss

---

## Types of Autoencoders

`Types_of_Autoencoders.ipynb`

Covers different types of Autoencoders, including:

* Basic Autoencoder
* Convolutional Autoencoder
* Denoising Autoencoder
* Sparse Autoencoder
* Variational Autoencoder (VAE)

---

## Generative Adversarial Networks

`Generative_Adversarial_Network.ipynb`

Introduces GANs and their two-network architecture:

* Generator
* Discriminator

The two networks are trained in an adversarial process to generate realistic data.

---

## Diffusion Models

`Diffusion_Models.ipynb`

Introduces diffusion-based generative models and the general idea behind:

* Forward Diffusion
* Noise Addition
* Reverse Diffusion
* Denoising
* Generative Modeling

---

# 📊 Datasets

Several datasets are included in the repository for practical experiments.

## MNIST

Used for handwritten digit classification and experiments with neural networks and optimization algorithms.

### Fashion-MNIST

Used for image classification experiments involving clothing and fashion items.

### CIFAR-10

Used for image classification and CNN experiments.

The dataset contains 10 image classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

### IMDB

Used for sentiment analysis and sequence modeling experiments.

### Clothing Reviews

A review dataset used for natural language processing and recurrent neural network experiments.

### Hindi-English Truncated Corpus

Used for sequence-to-sequence and machine translation experiments.

---

# 🛠️ Frameworks and Libraries

The notebooks use Python and popular Deep Learning and Machine Learning libraries, including:

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* PyTorch
* Jupyter Notebook

See `requirements.txt` for the complete list of dependencies.

---

# ▶️ How to Use

Each topic is implemented as a separate Jupyter Notebook.

A typical workflow is:

```text
1. Read the theoretical explanation
        ↓
2. Understand the mathematical intuition
        ↓
3. Study the implementation
        ↓
4. Run the notebook
        ↓
5. Experiment with the parameters
        ↓
6. Analyze the results
```

You can run individual notebooks using Jupyter:

```bash
jupyter notebook
```

Then navigate to the desired topic and open the corresponding `.ipynb` file.

---

# 🗺️ Recommended Learning Order

For beginners, the following order is recommended:

```text
Neural Networks
      ↓
Forward Propagation
      ↓
Backpropagation
      ↓
Gradient Descent
      ↓
SGD / Mini-Batch GD
      ↓
Momentum
      ↓
AdaGrad
      ↓
RMSProp
      ↓
Adam
      ↓
Regularization
      ↓
Dropout
      ↓
Batch Normalization
      ↓
Early Stopping
      ↓
CNN
      ↓
Pooling
      ↓
LeNet-5
      ↓
AlexNet
      ↓
ResNet
      ↓
MobileNetV2
      ↓
U-Net
      ↓
RNN
      ↓
Vanishing / Exploding Gradients
      ↓
LSTM
      ↓
GRU
      ↓
Bidirectional RNN / BiLSTM
      ↓
Seq2Seq
      ↓
Encoder-Decoder
      ↓
Attention
      ↓
Transfer Learning
      ↓
Autoencoders
      ↓
GANs
      ↓
Diffusion Models
```

---

# 🎯 Project Goals

The main goals of this repository are:

* Build a strong foundation in Deep Learning.
* Understand the mathematical intuition behind neural networks.
* Implement important algorithms from scratch where appropriate.
* Gain practical experience with TensorFlow and PyTorch.
* Understand CNN architectures and their evolution.
* Understand sequence modeling with RNNs, LSTMs, and GRUs.
* Learn attention-based architectures.
* Explore modern generative models.
* Practice Deep Learning using real datasets.
* Create a structured reference for future Deep Learning projects.

---

# 📌 Topics Covered

| Category          | Topics                                                   |
| ----------------- | -------------------------------------------------------- |
| Neural Networks   | Neural Networks, Forward Propagation, Backpropagation    |
| Optimization      | GD, SGD, Mini-Batch GD, Momentum, AdaGrad, RMSProp, Adam |
| Regularization    | L1/L2, Dropout, Batch Normalization, Early Stopping      |
| CNN               | Convolution, Pooling, Image Classification               |
| CNN Architectures | LeNet-5, AlexNet, MobileNetV2, ResNet, U-Net             |
| RNN               | Simple RNN, BPTT, Vanishing/Exploding Gradients          |
| Sequence Models   | LSTM, GRU, BiRNN, BiLSTM                                 |
| Attention         | Attention Mechanism                                      |
| Encoder-Decoder   | Seq2Seq, Encoder-Decoder Models                          |
| Transfer Learning | Feature Extraction, Fine-Tuning                          |
| Autoencoders      | Autoencoder, Convolutional AE, VAE                       |
| Generative AI     | GANs, Diffusion Models                                   |

---

# 📁 Data and Saved Models

Some sections contain local datasets and pretrained/trained model checkpoints used by the notebooks.

For example:

```text
Convolution_Neural_Network/
├── data/
├── images/
└── saved_models/
```

The `saved_models` directory contains trained Keras model checkpoints from experiments with ResNet on CIFAR-10.

---

# 🤝 Contributing

Contributions are welcome.

If you find an issue, have a suggestion, or want to improve an explanation or implementation:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Open a Pull Request.

---

# ⭐ Support

If this repository is useful for learning Deep Learning, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and learning purposes.
