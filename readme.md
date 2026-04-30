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


## Framework Overview

The proposed codebook\-based reconstruction\-classification framework for AF detection is illustrated below:
![framework]((https://github.com/Ou-Young-1999/DCGCNet/results/framework.png?raw=true))

