# Project Documentation: Skin Cancer Detection System

## 1. Executive Summary
This project aims to provide an accessible, AI-powered tool for early detection of skin cancer. Using deep learning (Convolutional Neural Networks), the system classifies skin lesion images into 7 diagnostic categories, providing users with instant analysis and confidence scores.

## 2. Technical Architecture

### 2.1 Backend
*   **Framework**: Flask (Python)
*   **ML Engine**: FastAI v1 (built on PyTorch)
*   **Model**: DenseNet169
    *   *Why DenseNet169?* Dense Convolutional Networks connect each layer to every other layer in a feed-forward fashion. It allows for strong gradient flow and feature propagation, making it highly effective for medical image analysis where fine details matter.
*   **Image Processing**: 
    *   Input images are resized to 224x224 pixels.
    *   Standard ImageNet normalization is applied.

### 2.2 Frontend
*   **Design Philosophy**: Modern, clean, and medical-grade aesthetics.
*   **Tech Stack**:
    *   **HTML5/Jinja2**: Dynamic templating.
    *   **CSS3**: Custom styling with gradients, flexbox, and animations.
    *   **Bootstrap 4**: Responsive grid system.
*   **User Flow**:
    1.  Landing Page -> Upload Image / Enter URL.
    2.  Model processing (backend).
    3.  Result Page -> Shows Prediction + Confidence Breakdown + "Detect Another" option.

### 2.3 Data Pipeline (`train_model.py`)
*   **Data Source**: HAM10000 Dataset ("Human Against Machine with 10000 training images").
*   **Preprocessing**:
    *   **Extraction**: Auto-detects and extracts zip files (`archive.zip`).
    *   **Flattening**: Automatically moves images from nested subdirectories to the root data folder.
    *   **Balancing**: The dataset is undersampled to handle class imbalance (limiting dominant classes like 'nv' to avoid bias).
*   **Augmentation**: Random flips and rotations are applied during training to improve model generalization.

## 3. Class Labels
The model predicts one of the following:
1.  **akiec**: Actinic keratoses
2.  **bcc**: Basal cell carcinoma
3.  **bkl**: Benign keratosis-like lesions
4.  **df**: Dermatofibroma
5.  **mel**: Melanoma
6.  **nv**: Melanocytic nevi
7.  **vasc**: Vascular lesions

## 4. Setup & Maintenance
*   **Dependencies**: Managed via `requirements.txt`.
*   **Compatibility**: 
    *   Includes patches for `torch.solve` to ensure compatibility between older FastAI versions and newer PyTorch versions.
    *   Includes specific fixes for Windows file path handling (`pathlib`).

## 5. Future Improvements
*   **Mobile App**: Convert the web app to a native mobile experience (React Native).
*   **More Data**: Integrate ISIC 2019/2020 datasets for better accuracy.
*   **Deployment**: containerize with Docker for cloud deployment (AWS/GCP).
