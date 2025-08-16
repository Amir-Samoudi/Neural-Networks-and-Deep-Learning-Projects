# Neural Networks and Deep Learning Projects

This repository contains the projects for the graduate-level **"Neural Networks and Deep Learning"** course at the **University of Tehran**.  
The collection showcases practical implementations of various neural network architectures and deep learning techniques.

---

## Topics Covered
- Multi-Layer Perceptrons (MLPs) & Autoencoders  
- Convolutional Neural Networks (CNNs)  
- Region-Based CNNs for Object Detection  
- Recurrent Neural Networks (RNNs)  
- Transformers  
- Deep Generative Models  

---

## Projects
- [Homework 1: Multi-Layer Perceptrons and Autoencoders](#-homework-1-multi-layer-perceptrons-and-autoencoders)  
- [Homework 2: Convolutional Neural Networks](#-homework-2-convolutional-neural-networks)  
- Homework 3: (Coming Soon)  
- ... and so on for all 7 projects.  

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

## Homework 2: Convolutional Neural Networks

This assignment explores the application of **Convolutional Neural Networks (CNNs)** for image classification tasks, including medical imaging and vehicle recognition.

### Part 1: COVID-19 Disease Detection from X-Ray Images
- **Task:** Image Classification  
- **Paper:** [COVID-19 Disease Detection Based on X-Ray Image Classification](https://onlinelibrary.wiley.com/doi/10.1155/2021/6621607)  
- **Description:**  
  A custom CNN architecture was implemented to classify chest X-ray images into three categories: **COVID-19, Normal, and Pneumonia**.  
  The project involved preprocessing a highly imbalanced dataset by **downsampling**, followed by **data augmentation** to create a balanced and robust training set.  
  The model achieved high accuracy in detecting COVID-19 cases.  

---

### Part 2: A Hybrid VGG-16 and SVM Model for Vehicle Classification
- **Task:** Image Classification (Hybrid Model)  
- **Paper:** [A Hybrid Deep Learning VGG-16 Based SVM Model for Vehicle Type Classification](https://www.researchgate.net/publication/389011147_A_Hybrid_Deep_Learning_VGG-16_Based_SVM_Model_for_Vehicle_Type_Classification)  
- **Description:**  
  This project implements a **hybrid approach** for vehicle classification.  
  The pre-trained **VGG-16** model is used as a feature extractor to generate feature vectors from vehicle images.  
  These features are then fed into a **Support Vector Machine (SVM)** classifier.  
  This method combines the feature representation strength of deep learning with the classification efficiency of traditional machine learning.  

---
