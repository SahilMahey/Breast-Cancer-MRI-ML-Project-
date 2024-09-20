# Breast Cancer MRI Machine Learning Project

Hi, welcome to this repository!

This project is currently under development, and model details may change as results improve.

Here’s a structured index for your README:

---

# **Index**

1. [Introduction](#introduction)
2. [What is the Breast Cancer MRI Machine Learning Project?](#what-is-the-breast-cancer-mri-machine-learning-project)
3. [What is an MRI?](#what-is-an-mri)
4. [Datasets Used](#datasets-used)  
5. [Data Preprocessing](#data-preprocessing)
6. [Classification Models](#classification-models)  
   6.1 [Convolutional Neural Network (CNN)](#1-convolutional-neural-network-cnn)  
   6.2 [Vision Transformer (ViT)](#2-vision-transformer-vit)
7. [Segmentation](#segmentation)
8. [Current Results](#current-results)

---

This index will help readers quickly navigate to different sections of your README. You can adjust the links to suit the final structure of your document.

## What is the Breast Cancer MRI Machine Learning Project?

I am working on building a Machine Learning (ML) model to predict whether a given MRI image contains cancer or not. This is a **classification** problem in ML.

In the future, I plan to extend the project to develop a model that can **locate** cancerous tumors within MRI images (e.g., by marking a boundary around the tumor). This is known as a **segmentation** problem in ML.

This README provides an overview of the project's current status.

---

## What is an MRI?

Most people have seen MRI machines, but it's important to understand how MRI data is structured.

An MRI file consists of a stack of black-and-white 2D images, called **slices**. Each slice represents a cross-section of the body part being scanned. For example, a breast MRI would contain slices showing different parts of the breast, allowing doctors to examine the tissue layer by layer.

---

## Datasets Used

### 1. **Duke Breast Cancer MRI Dataset**  
This dataset contains MRIs of 922 patients in DICOM format. Each patient has undergone MRI scans at various stages of treatment, and I’ve selected the pre-treatment MRIs for all patients. The dataset also includes an annotation file, which indicates which slices contain visible cancerous tumors. This dataset is used for training and testing.
[Dataset link](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/)

### 2. **MAMA-MIA Dataset**
This dataset combines multiple MRI datasets, including Duke, ISPY2, ISPY1, and TCGA-BRCA. I focused on the ISPY2 dataset, which contains 980 patients. After filtering for bilateral breast MRIs (both breasts visible, no implants, axial position only), I narrowed it down to 166 patients. This dataset is used for validation.
[Dataset link](https://github.com/LidiaGarrucho/MAMA-MIA)




---

## Data Preprocessing

Here’s how I prepared the dataset for machine learning:

1. **Resizing**: I resized each MRI slice to 256x256 pixels and saved them as `.png` images.
2. **Train-Test Split**: I ensured that slices from a single patient were either in the training set or the test set, avoiding data leakage.
3. **Data Augmentation**: Some images in the training, testing, and validation sets were randomly rotated by -90 or +90 degrees.
4. **Conversion**: All images were converted into NumPy arrays, with labels assigned (0 for no cancer, 1 for cancer).
5. **Shuffling**: The data was shuffled to ensure randomness.

Technologies used: `Pandas`, `NumPy`, `PIL`, `matplotlib`, `seaborn`, `pydicom`.

---

## Classification Models

### **1. Convolutional Neural Network (CNN)**

This model is built using a pretrained DenseNet-121 without changing the weights, followed by two fully connected layers with 1024 and 512 units.

- **Framework**: TensorFlow + Keras
- **Activation**: Softmax
- **Learning Rate**: 1e^-3
- **Loss Function**: Categorical Cross-Entropy
- **Features**:
  - Checkpoints saved for the best models (based on loss and accuracy)
  - Learning rate reduced after 3 epochs of no improvement
  - Early stopping after 5 epochs with no improvement

#### Current Results:

![image](https://github.com/user-attachments/assets/ac3fc292-a92b-4410-b068-01adedfe2f5f)

### **2. Vision Transformer (ViT)**

The Vision Transformer (ViT) classifies images using a transformer architecture. The code is based on [this repository](https://github.com/tintn/vision-transformer-from-scratch), with extensions to support data augmentation and validation on independent datasets (e.g., MAMA-MIA).

I am training the model from scratch without using pretrained weights. Here’s the configuration:

```python
config = {
    "patch_size": 16,  
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 256,
    "num_classes": 2,
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}
```

#### Results:
Coming soon...

---

## Segmentation

Coming soon...  
**(A preview: I plan to use a UNet++ pretrained model for segmentation.)**

--- 

Stay tuned for further updates!
