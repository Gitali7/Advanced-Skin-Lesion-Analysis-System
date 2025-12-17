# Skin Cancer Detection: A Modern Web Development Model

![Skin Cancer Detection](src/static/css/images/logo.png) *<!-- (Optional: Add a logo if you have one) -->*

A state-of-the-art Deep Learning web application designed to classify skin lesions as Benign or Malignant. Built with **FastAI** and **Flask**, featuring a modern, vibrant user interface.

## 🚀 Features

*   **Deep Learning Model**: Powered by a **DenseNet169** architecture trained on the HAM10000 dataset.
*   **7-Class Classification**: Detects 7 distinct types of skin lesions:
    *   Actinic keratoses
    *   Basal cell carcinoma
    *   Benign keratosis
    *   Dermatofibroma
    *   Melanocytic nevi
    *   Melanoma
    *   Vascular lesions
*   **Modern UI/UX**:
    *   Vibrant, gradient-based color scheme.
    *   Responsive card layouts.
    *   "Detect Another" fast-retry workflow.
    *   Visual confidence bars for predictions.
*   **Robust Training**: Includes a custom training script (`train_model.py`) that handles data extraction, balancing, and training automatically.

## 🛠️ Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd SkinCancerDetection-WebApp
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed (3.8+ recommended).
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ How to Run

### 1. Train the Model
If you need to retrain the model (or if running for the first time without a pre-trained model):
1.  Place your dataset (`archive.zip` or `skin-cancer-mnist-ham10000.zip`) in the `data/` folder.
2.  Run the training script:
    ```bash
    python train_model.py
    ```
    *This will extract data, train for 3 epochs (default), and save `models/model_best.pth`.*

### 2. Run the Web Application
Once you have the model:
1.  Start the Flask server:
    ```bash
    python src/app.py
    ```
2.  Open your browser and navigate to:
    ```
    http://localhost:8008
    ```

## 📂 Project Structure

*   **`src/`**: Contains the Flask application.
    *   `app.py`: Main server file.
    *   `templates/`: HTML files (`index.html`, `result.html`).
    *   `static/`: CSS and styling.
*   **`train_model.py`**: Script to train the Neural Network.
*   **`data/`**: Folder for dataset storage.
*   **`models/`**: Stores the trained model (`model_best.pth`).
*   **`requirements.txt`**: Python dependencies.

## 📝 Technologies Used
*   **Python 3**
*   **FastAI (v1)** & **PyTorch**
*   **Flask** (Web Framework)
*   **HTML5, CSS3, Bootstrap 4**
*   **Google Fonts & FontAwesome**

---
*Created for the Skin Cancer Detection Project.*
