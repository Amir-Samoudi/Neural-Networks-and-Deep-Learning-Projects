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
- [Homework 3: Region-Based CNNs](#-homework-3-region-based-cnns)  
- [Homework 4: Recurrent Neural Networks](#-homework-4-recurrent-neural-networks) 
- [Homework 5: Transformers and Vision Transformers](#-homework-5-transformers-and-vision-transformers)  
- [Homework 6: Deep Generative Models](#-homework-6-deep-generative-models)    

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

## Homework 3: Region-Based CNNs

This assignment delves into advanced computer vision tasks using **region-based CNNs**, focusing on **real-time semantic segmentation** and **object detection with oriented bounding boxes**.

### Part 1: Segmentation of Urban Scenes using Fast-SCNN
- **Task:** Semantic Segmentation  
- **Paper:** [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/2106.03146) 
- **Dataset:** CamVid 
- **Description:**  
  This project implements the **Fast-SCNN** architecture, designed for **high-speed, low-memory** semantic segmentation of high-resolution images.  
  - The model uses a **two-branch design**: a high-level feature extraction branch and a spatial detail branch.  
  - These branches are merged to produce the final segmentation map.  
  - It was trained on the **CamVid dataset** to classify pixels into categories like *road, car, pedestrian,* etc.  

---

### Part 2: Oriented R-CNN for Object Detection
- **Task:** Object Detection with Oriented Bounding Boxes  
- **Paper:** [Oriented R-CNN for Object Detection](https://arxiv.org/abs/2108.05699)  
- **Dataset:** [HRSC2016 (High-Resolution Ship Collection)](https://www.kaggle.com/datasets/guofeng/hrsc2016)  
- **Description:**  
  An implementation of **Oriented R-CNN** to detect ships in high-resolution satellite imagery.  
  - Unlike standard object detectors, this model predicts **oriented bounding boxes**, which are better suited for long, thin objects with arbitrary orientations.  
  - The architecture uses a **ResNet-50-FPN backbone** and a specialized **Region Proposal Network (oriented RPN)**.  
  - Proposals are refined using a **rotated RoI Align** layer for final detection.  

---

## Homework 4: Recurrent Neural Networks

This assignment explores **Recurrent Neural Networks (RNNs)**, including **LSTM** and **GRU** architectures, for sequence modeling tasks such as image captioning and time-series prediction.

### Part 1: Image Captioning with ResNet50 and Hybrid LSTM-GRU
- **Task:** Image Captioning  
- **Paper:** [Image captioning deep learning model using ResNet50 encoder and hybrid LSTM–GRU decoder optimized with beam search](https://www.tandfonline.com/doi/pdf/10.1080/00051144.2025.2485695)  
- **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
- **Description:**  
  This project implements an **encoder-decoder model** to generate captions for images.  
  - A pre-trained **ResNet-50** acts as the encoder to extract features.  
  - A hybrid **LSTM-GRU decoder** generates textual captions.  
  - Optimized with **beam search** to produce more coherent and contextually relevant sentences.  

---

### Part 2: Context-aware LSTM for Clinical Event Time-series Prediction
- **Task:** Time-series Prediction  
- **Paper:** [Recent Context-aware LSTM for Clinical Event Time-series Prediction](https://people.cs.pitt.edu/%7Emilos/research/2019/Lee_Hauskrecht_AIME_2019.pdf)  
- **Dataset:** [PhysioNet Sepsis Prediction Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)  
- **Description:**  
  This project focuses on **predicting future clinical events** from patient time-series data.  
  - Implements **LSTM, Bi-LSTM, GRU, and Bi-GRU** models.  
  - Trained to forecast **20 physiological variables** in the next time step.  
  - Aimed at early detection and prevention of **sepsis** using sequential patient data.  

---

## Homework 5: Transformers and Vision Transformers

This assignment focuses on the **Transformer architecture**, exploring its application in computer vision through **Vision Transformers (ViT)** for direct image classification and investigating the **adversarial robustness** of large-scale multi-modal models like **CLIP**.

### Part 1: Image Classification by ViT for Smart Agriculture
- **Task:** Image Classification  
- **Paper:** [ViT-SmartAgri: Vision Transformer and Smartphone-Based Plant Disease Detection for Smart Agriculture](https://www.sciencedirect.com/science/article/abs/pii/S0168169923000318)  
- **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Description:**  
  Implements a **Vision Transformer (ViT)** to classify plant leaf diseases.  
  - Unlike CNNs, ViT treats an image as a sequence of patches and applies a **Transformer encoder** to model relationships.  
  - Trained on the **PlantVillage dataset** to identify multiple plant diseases.  
  - Demonstrates the power of **attention-based architectures** in agriculture applications.  

---

### Part 2: Robust Zero-Shot Classification with CLIP
- **Task:** Zero-Shot Classification & Adversarial Robustness  
- **Paper:** [Understanding Zero-shot Adversarial Robustness for Large-Scale Models](https://arxiv.org/abs/2112.09301)  
- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
- **Description:**  
  Investigates the robustness of **OpenAI’s CLIP** (Contrastive Language–Image Pre-training).  
  - **Zero-Shot Classification:** Evaluated CLIP on CIFAR-10 by comparing image embeddings with class text embeddings (e.g., *“a photo of a car”*).  
  - **Adversarial Attacks:** Generated adversarial samples with a pre-trained ResNet and tested transferability to CLIP.  
  - **Adversarial Fine-Tuning:** Explored robustness improvements using **standard CE loss** and **TeCoA (Text-guided Contrastive Adversarial) loss**, with **LoRA** for parameter-efficient adaptation.  

---

## Homework 6: Deep Generative Models

This assignment explores **deep generative models**, focusing on **Generative Adversarial Networks (GANs)** for domain adaptation and **Variational Autoencoders (VAEs)** for medical image generation and reconstruction.

### Part 1: Unsupervised Pixel-Level Domain Adaptation with GANs
- **Task:** Unsupervised Domain Adaptation  
- **Paper:** [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/abs/1612.05424)  
- **Datasets:** [MNIST](http://yann.lecun.com/exdb/mnist/) & [MNIST-M](https://github.com/erictzeng/pixelda)  
- **Description:**  
  Addresses the **domain gap problem** by adapting a model trained on **grayscale MNIST** to work on **colored MNIST-M** without target labels.  
  - A **GAN generator** learns to translate source images into target style.  
  - A **discriminator** enforces realism in generated samples.  
  - A **task classifier** ensures digit identity is preserved.  
  - Effectively bridges the visual gap between the domains.  

---

### Part 2: Generating Endoscopic Images with EndoVAE
- **Task:** Image Generation & Reconstruction  
- **Paper:** [EndoVAE: Generating Endoscopic Images with a Variational Autoencoder](https://arxiv.org/abs/2203.09137)  
- **Dataset:** [Kvasir](https://datasets.simula.no/kvasir/)  
- **Description:**  
  Implements an **EndoVAE** to generate and reconstruct endoscopic images.  
  - **Encoder:** Compresses images into a probabilistic latent space (*mean* & *log-variance*).  
  - **Decoder:** Reconstructs images from sampled latent vectors.  
  - **Losses:** Combination of **Reconstruction Loss** (BCE/MSE) and **KL Divergence Loss** ensures realism and smooth latent space.  
  - Enables **novel image generation** for medical imaging applications.  

---
