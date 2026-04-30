# Atrial Fibrillation Detection with Arbitrary Leads via a Codebook\-Based Reconstruction\-Classification Framework

This repository contains the official implementation of the paper:**Atrial Fibrillation Detection with Arbitrary Leads via a Codebook\-Based Reconstruction\-Classification Framework**\.

The code provides a complete pipeline for atrial fibrillation \(AF\) detection using ECG signals with arbitrary leads, based on a novel codebook\-based learning framework that integrates reconstruction and classification tasks\.

## Dependencies

The following dependencies are required to run the code\. Install them using `pip install \&lt;package\&gt;==\&lt;version\&gt;`:

```plain text
Name: torch
Version: 2.7.1+cu126
Name: pandas
Version: 2.3.3
Name: matplotlib
Version: 3.10.7
Name: numpy
Version: 2.2.6
Name: scikit-learn
Version: 1.7.2
Name: scipy
Version: 1.16.3
```

## Repository Structure \&amp; File/Folder Introduction

This repository is strictly organized according to the functional modules of the AF detection framework, and the detailed role of each file and folder is as follows:

- **checkpoints/**: Used to store the model checkpoints generated during the training process\. Each checkpoint records the model parameters, optimizer state, and training metrics at a specific epoch, which can be used to resume training or directly perform inference\.

- **config/**: Contains all configuration files of the project, including hyperparameter settings \(learning rate, batch size, codebook size, etc\.\), dataset path configuration, model structure parameters, and training/testing related parameters\. Modify the files in this folder to adjust the experimental settings\.

- **dataset/**: Responsible for dataset preparation, including data preprocessing scripts, data loading functions, and training/validation/test set splitting scripts\. It can convert raw ECG data into a format usable by the model and realize data augmentation \(if needed\) to improve model generalization\.

- **model/**: Stores the definition of the proposed codebook\-based reconstruction\-classification framework, including the codebook module, ECG reconstruction sub\-network, AF classification sub\-network, and other core components\. All model\-related code \(network structure, forward propagation logic, etc\.\) is concentrated here\.

- **results/**: Saves all experimental results, including the framework structure diagram \(framework\.png\), AF detection performance metrics \(accuracy, sensitivity, specificity, etc\.\), visualization results of model interpretation \(Grad\-CAM heatmaps, codebook usage analysis\), and test report files\.

- **train\.py**: The main training script of the project\. By calling the configuration files, dataset modules, and model modules, it completes the entire training process \(data loading, model initialization, loss calculation, parameter update, etc\.\), and saves the trained model checkpoints to the checkpoints/ folder\.

- **test\.py**: The inference and test script\. It loads the trained model checkpoint, reads the test set data, performs AF detection inference, calculates the test metrics, and saves the test results \(metrics, visualization figures\) to the results/ folder\.

## Framework Overview

The proposed codebook\-based reconstruction\-classification framework for AF detection is illustrated below:

The proposed codebook\-based reconstruction\-classification framework for AF detection is illustrated below:

> （注：文档部分内容可能由 AI 生成）
