# Skin Lesion Classification with Deep Learning (ISIC 2019)

This project aims to classify skin diseases by analyzing dermoscopic images using Deep Learning techniques. It represents a **progressive R&D process**, evolving from basic binary classification to complex multi-class architectures for skin cancer diagnosis.

## 🧬 Dataset

The project utilizes the **ISIC 2019 (International Skin Imaging Collaboration)** dataset, a global standard for skin cancer diagnosis.

* **Source:** ISIC 2019 Challenge
* **Content:** 25,331 dermoscopic images and metadata.
* **Classes:** Melanoma, Melanocytic nevus, Basal cell carcinoma, Actinic keratosis, Benign keratosis, Dermatofibroma, Vascular lesion, Squamous cell carcinoma.
* **Challenges:** Addressed class imbalance and high visual similarity between lesion types.

---

## 📈 Project Evolution & Model Development

The work in this repository follows a **top-down hierarchy**, moving from simple approaches to advanced solutions. The following stages represent how the model evolved over time, highlighting different experiments and performance improvements.

### ⬇️ Phase 1: The Baseline Approach (Binary Classification)
**File:** `2li (1).ipynb`
**Model:** `MobileNetV2`

In the initial phase, the problem was simplified to the core question: "Is there a disease or not?"
* **Approach:** The dataset was reduced to two main classes: **Benign** and **Malignant**.
* **Goal:** To create a rapid prototype and test the model's general learning capacity.
* **Techniques:** Transfer Learning was applied by freezing the weights of the `MobileNetV2` architecture for feature extraction.

### ⬇️ Phase 2: Advanced Multi-Class Classification (Final Model)
**File:** `7li (1).ipynb`
**Model:** `NASNetMobile`

Following the success of Phase 1, the project was expanded to predict the specific type of lesion. This stage represents the **state-of-the-art** version of the project.
* **Approach:** A more challenging problem was tackled by separating the dataset into 7 distinct disease categories (Multi-class Classification).
* **Architecture:** `NASNetMobile`, developed by Google and optimized for mobile efficiency, was selected.
* **Key Improvements:**
    * **Fine-Tuning:** The top layers of the model were unfrozen to learn dataset-specific features.
    * **Callbacks:** `EarlyStopping` and `ReduceLROnPlateau` were implemented to prevent overfitting and dynamically optimize the learning rate.
    * **Data Augmentation:** Techniques such as rotation and zooming were applied to increase data diversity.

---

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Frameworks:** TensorFlow & Keras
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Environment:** Google Colab (Tesla T4 GPU)

## 📊 Results
Throughout the training process, `val_loss` and `val_accuracy` metrics were closely monitored. The **Phase 2 (NASNetMobile)** model, despite handling a more complex classification task, produced more specific and clinically meaningful results compared to the baseline model.

## 🚀 Installation & Usage

To run this project on your local machine:

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install tensorflow pandas numpy matplotlib
    ```
3.  Open the notebooks (`.ipynb`) using Jupyter Notebook or Google Colab.

---
**Developer:** Halil Ayar
