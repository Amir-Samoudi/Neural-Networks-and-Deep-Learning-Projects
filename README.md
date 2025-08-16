# Neural Networks and Deep Learning Projects

This repository contains the projects for the graduate-level **"Neural Networks and Deep Learning"** course at the **University of Tehran**.  
The collection showcases practical implementations of various neural network architectures and deep learning techniques.

---

## Topics Covered
- Multi-Layer Perceptrons (MLPs) & Autoencoders  
- Convolutional Neural Networks (CNNs)  
- Region-Based CNNs for Object Detection and Segmentation  
- Recurrent Neural Networks (RNNs)  
- Transformers  
- Deep Generative Models  

---

## Projects
- **Homework 1:** Multi-Layer Perceptrons and Autoencoders  
- **Homework 2:** ....

---

## Homework 1: Multi-Layer Perceptrons and Autoencoders

This assignment focuses on the fundamentals of fully connected networks. It covers classification and regression using Multi-Layer Perceptrons (MLPs), implementing a simple neuron from scratch, and using autoencoders for feature extraction.

### Part 1: Credit Card Fraud Detection using MLP
- **Task:** Classification  
- **Dataset:** Credit Card Fraud Detection  
- **Description:**  
  A Multi-Layer Perceptron (MLP) was trained to distinguish between fraudulent and legitimate credit card transactions.  
  The primary challenge was handling the highly imbalanced nature of the dataset.  

---

### Part 2: Concrete Strength Regression using MLP
- **Task:** Regression  
- **Dataset:** Concrete Compressive Strength  
- **Description:**  
  An MLP was trained to predict the compressive strength of concrete based on its components.  
  Different loss functions (e.g., Mean Squared Error, Mean Absolute Error) and optimizers (e.g., Adam, SGD) were explored to analyze their impact on the model's performance.  

---

### Part 3: Adaline for Iris Dataset Classification
- **Task:** Binary Classification  
- **Dataset:** Iris Dataset  
- **Description:**  
  An **ADALINE (Adaptive Linear Neuron)** was implemented from scratch using only NumPy.  
  The neuron was trained to solve a binary classification problem on the Iris dataset, and the effect of the learning rate on the convergence of the training process was investigated.  

---

### Part 4: Training Autoencoder and Classification on MNIST
- **Task:** Feature Extraction & Classification  
- **Dataset:** MNIST Handwritten Digits  
- **Description:**  
  Two autoencoders with different latent space dimensions were trained on the MNIST dataset.  
  The trained encoders were then frozen and used as feature extractors.  
  Finally, separate MLP classifiers were trained on these extracted features to perform digit classification.  
  This experiment assessed how the dimensionality of the autoencoder's latent space affects the quality of the features for a downstream classification task.  

---
