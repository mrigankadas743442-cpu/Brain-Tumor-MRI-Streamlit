# 🧠 Brain Tumor MRI Image Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Processing-blue?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![CNN](https://img.shields.io/badge/CNN-Computer%20Vision-green)
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-Deep%20Learning-purple)
![Medical Imaging](https://img.shields.io/badge/Medical%20Imaging-MRI-red)
![Image Classification](https://img.shields.io/badge/Image%20Classification-Computer%20Vision-blue)
![Data Preprocessing](https://img.shields.io/badge/Data%20Preprocessing-ML-yellow)
![Data Augmentation](https://img.shields.io/badge/Data%20Augmentation-Deep%20Learning-orange)
![Model Evaluation](https://img.shields.io/badge/Model%20Evaluation-ML-success)
![Model Comparison](https://img.shields.io/badge/Model%20Comparison-Analysis-informational)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)

---

## 🚀 Live Demo

**Streamlit Application:**  
https://brain-tumor-mri-image-analysis.streamlit.app/

**GitHub Repository:**  
https://github.com/mrigankadas743442-cpu/Brain-Tumor-MRI-Streamlit

---

## 📌 Project Overview

This project develops a deep learning-based **Brain Tumor MRI Image Classification system** capable of classifying brain MRI images into four categories:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

The project follows a complete deep learning workflow, including:

- Dataset exploration
- Exploratory data analysis
- Data preprocessing
- Data augmentation
- Custom CNN development
- Transfer learning
- Model training
- Model evaluation
- Model comparison
- Streamlit deployment

A custom Convolutional Neural Network (CNN) was developed from scratch and compared with four pretrained ImageNet models:

- ResNet50
- MobileNetV2
- InceptionV3
- EfficientNetB0

Among all evaluated models, **EfficientNetB0 achieved the best overall performance** on the test dataset and was selected for the final Streamlit application.

> **Important:** This project is intended for **AI-assisted research and educational purposes**. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 🎯 Objectives

The main objectives of this project are:

1. Explore and understand the Brain Tumor MRI image dataset.
2. Analyze the distribution of tumor categories.
3. Check image properties and dataset consistency.
4. Preprocess MRI images for deep learning.
5. Resize images to **224 × 224** pixels.
6. Normalize image pixel values where appropriate.
7. Apply image augmentation techniques.
8. Build a custom CNN model from scratch.
9. Apply transfer learning using pretrained ImageNet models.
10. Train and validate multiple deep learning models.
11. Evaluate models using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
12. Compare the performance of all models.
13. Select the best-performing model.
14. Deploy the selected model through a Streamlit application.
15. Provide an interface for uploading MRI images and obtaining an AI-assisted prediction.

---

# 📂 Dataset

The dataset contains a total of **2,450 brain MRI images** divided into four classes.

### Classes

| Class | Description |
|---|---|
| `glioma` | Glioma tumor MRI images |
| `meningioma` | Meningioma tumor MRI images |
| `no_tumor` | MRI images without tumor |
| `pituitary` | Pituitary tumor MRI images |

### Dataset Split

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---:|---:|---:|---:|---:|
| Train | 568 | 358 | 335 | 438 | 1,699 |
| Validation | 161 | 124 | 99 | 121 | 505 |
| Test | 80 | 63 | 49 | 54 | 246 |
| **Total** | **809** | **545** | **483** | **613** | **2,450** |

The dataset contains four image categories with some class imbalance. Glioma represents the largest class, while the no-tumor category contains the fewest images.

---

# 🔎 Dataset Exploration

The exploratory data analysis included:

- Dataset shape and structure
- Column inspection
- Missing-value analysis
- Duplicate-value analysis
- Class distribution
- Train/validation/test distribution
- Sample MRI image visualization
- Multiple image samples from each class
- Image property inspection

The dataset metadata contains:

```text
Split
Class
Filename
File_Path
```

## Dataset Quality Checks

The dataset was checked for:

- Missing values
- Duplicate rows
- Duplicate filenames
- Duplicate file paths
- Class consistency
- Split consistency

The dataset contained:

- **0 missing values**
- **0 duplicate rows**
- **0 duplicate filenames**
- **0 duplicate file paths**

---

# 🖼️ Image Properties

A representative sample of **504 images** was inspected to understand the image properties.

The inspected sample showed:

- Image dimensions: **640 × 640**
- Color format: **RGB**
- File format: **JPEG**

The images were subsequently resized to **224 × 224** pixels for model training.

---

# 📊 Class Distribution

The overall class distribution is approximately:

| Class | Images | Percentage |
|---|---:|---:|
| Glioma | 809 | 33.02% |
| Pituitary | 613 | 25.02% |
| Meningioma | 545 | 22.24% |
| No Tumor | 483 | 19.71% |

The distribution shows moderate class imbalance, which was considered during model development and evaluation.

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Dataset cleaning
- Class-name standardization
- Split-name standardization
- Image resizing
- Pixel normalization
- Batch generation
- Categorical label encoding

### Image Size

All images used for model training were resized to:

```text
224 × 224 × 3
```

### Pixel Normalization

For the standard CNN, ResNet50, MobileNetV2, and InceptionV3 pipelines, image pixel values were scaled to the range:

```text
0 – 1
```

The EfficientNetB0 pipeline used the model's internal input rescaling, so external pixel rescaling was not applied to that pipeline.

---

# 🔄 Data Augmentation

Data augmentation was applied to the training images to improve model generalization.

The augmentation pipeline included:

- Rotation
- Horizontal flipping
- Vertical flipping
- Zoom
- Brightness adjustment
- Width shifting
- Height shifting

Example augmentation configuration:

```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2]
)
```

Validation and test images were not randomly augmented during evaluation.

---

# 🧠 Model Development

Two main deep learning approaches were used:

## 1. Custom CNN

A CNN was developed from scratch using TensorFlow/Keras.

The architecture contains:

- Convolutional layers
- Batch Normalization
- Max Pooling
- Dense layers
- Dropout
- Softmax output layer

### Custom CNN Architecture

```text
Input: 224 × 224 × 3

Conv2D - 32 filters
Batch Normalization
Max Pooling

Conv2D - 64 filters
Batch Normalization
Max Pooling

Conv2D - 128 filters
Batch Normalization
Max Pooling

Conv2D - 256 filters
Batch Normalization
Max Pooling

Flatten

Dense - 256
Batch Normalization
Dropout - 0.5

Output - 4 classes
Softmax
```

---

# 🔬 Transfer Learning

Four pretrained ImageNet architectures were evaluated.

### ResNet50

ResNet50 was loaded with ImageNet pretrained weights and its base layers were frozen while a new classification head was added.

### MobileNetV2

MobileNetV2 was used as a lightweight pretrained feature extractor with a custom classification head.

### InceptionV3

InceptionV3 was used with ImageNet pretrained weights and a custom classification head.

### EfficientNetB0

EfficientNetB0 was used with ImageNet pretrained weights and a custom classification head.

EfficientNetB0 produced the best overall test performance and was selected as the final model.

---

# ⚙️ Model Training

The models were trained using TensorFlow/Keras.

### Optimizer

```text
Adam
```

### Loss Function

```text
Categorical Crossentropy
```

### Output Activation

```text
Softmax
```

### Number of Classes

```text
4
```

### Input Image Size

```text
224 × 224 × 3
```

### Training Controls

The training process used:

- EarlyStopping
- ModelCheckpoint
- Validation loss monitoring
- Best-model restoration

EarlyStopping was used to reduce unnecessary training once validation performance stopped improving.

ModelCheckpoint was used to save the best-performing model based on validation loss.

---

# 📈 Model Evaluation

The trained models were evaluated on the independent test dataset.

The following metrics were calculated:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Training and validation accuracy/loss histories were also visualized to understand model learning behavior.

---

# 🏆 Model Comparison

The final test-set performance is shown below.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Custom CNN | 81.30% | 83.49% | 81.30% | 81.46% |
| ResNet50 | 78.05% | 77.81% | 78.05% | 77.44% |
| MobileNetV2 | 84.15% | 84.72% | 84.15% | 83.61% |
| InceptionV3 | 86.18% | 86.39% | 86.18% | 86.14% |
| **EfficientNetB0** | **88.21%** | **88.23%** | **88.21%** | **88.07%** |

---

# 🥇 Best Performing Model

**EfficientNetB0** achieved the highest overall performance among the evaluated models.

### EfficientNetB0 Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **88.21%** |
| Precision | **88.23%** |
| Recall | **88.21%** |
| F1-Score | **88.07%** |

EfficientNetB0 correctly classified:

```text
217 out of 246 test images
```

This corresponds to an overall test accuracy of:

```text
88.21%
```

---

# 📋 EfficientNetB0 Class-wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 89.16% | 92.50% | 90.80% |
| Meningioma | 80.65% | 79.37% | 80.00% |
| No Tumor | 92.86% | 79.59% | 85.71% |
| Pituitary | 91.53% | 100.00% | 95.58% |

---

# 🔲 EfficientNetB0 Confusion Matrix

The EfficientNetB0 model produced the following test-set classification results:

| Actual Class | Correctly Classified |
|---|---:|
| Glioma | 74 / 80 |
| Meningioma | 50 / 63 |
| No Tumor | 39 / 49 |
| Pituitary | 54 / 54 |

Total correctly classified test images:

```text
217 / 246
```

---

# 💾 Trained Models

The trained models were saved in `.h5` format as required for the project deliverables.

The model files include:

```text
custom_cnn_best.h5
resnet50_best.h5
mobilenetv2_best.h5
inceptionv3_best.h5
efficientnetb0_best.h5
```

The final Streamlit application uses:

```text
efficientnetb0_best.h5
```

because EfficientNetB0 achieved the best overall test performance.

---

# 🌐 Streamlit Application

A Streamlit application was developed to provide an interactive interface for the final model.

The application allows users to:

1. Upload a brain MRI image.
2. Display the uploaded image.
3. Process the image.
4. Generate a prediction using EfficientNetB0.
5. Display the predicted tumor category.
6. Display the prediction confidence.
7. Display probabilities for all four classes.

### Supported Image Formats

```text
JPG
JPEG
PNG
```

### Application Classes

```text
Glioma
Meningioma
No Tumor
Pituitary
```

---

# 🚀 Deployment

The Streamlit application was deployed using **Streamlit Community Cloud**.

### Live Application

https://brain-tumor-mri-image-analysis.streamlit.app/

### GitHub Repository

https://github.com/mrigankadas743442-cpu/Brain-Tumor-MRI-Streamlit

---

# 🧪 Deployment Testing

The deployed application was tested using MRI images representing all four classes.

| Test Class | Predicted Class | Result |
|---|---|---|
| Meningioma | Meningioma | ✅ Correct |
| Glioma | Glioma | ✅ Correct |
| Pituitary | Pituitary | ✅ Correct |
| No Tumor | No Tumor | ✅ Correct |

The application successfully generated predictions for all four tested categories.

The displayed confidence represents the model's output probability for the predicted class and should not be interpreted as medical certainty.

---

# 🏗️ Project Workflow

```text
Dataset
   │
   ▼
Data Understanding
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Cleaning & Wrangling
   │
   ▼
Image Preprocessing
   │
   ├── Resize to 224 × 224
   └── Pixel normalization
   │
   ▼
Data Augmentation
   │
   ├── Rotation
   ├── Flipping
   ├── Zoom
   ├── Brightness
   └── Shifting
   │
   ▼
Model Development
   │
   ├── Custom CNN
   ├── ResNet50
   ├── MobileNetV2
   ├── InceptionV3
   └── EfficientNetB0
   │
   ▼
Model Training
   │
   ├── EarlyStopping
   └── ModelCheckpoint
   │
   ▼
Model Evaluation
   │
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1-Score
   └── Confusion Matrix
   │
   ▼
Model Comparison
   │
   ▼
EfficientNetB0 Selected
   │
   ▼
Streamlit Application
   │
   ▼
Streamlit Community Cloud
```

---

# 📁 Streamlit Repository Structure

```text
Brain-Tumor-MRI-Streamlit/
│
├── app.py
├── efficientnetb0_best.h5
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit application and model inference code |
| `efficientnetb0_best.h5` | Best-performing EfficientNetB0 trained model |
| `requirements.txt` | Python dependencies required for deployment |
| `README.md` | Project documentation |

---

# 🛠️ Technology Stack

## Programming Language

- Python

## Deep Learning

- TensorFlow
- Keras
- Convolutional Neural Networks
- Transfer Learning

## Pretrained Models

- ResNet50
- MobileNetV2
- InceptionV3
- EfficientNetB0

## Data Processing

- NumPy
- Pandas
- Pillow

## Data Visualization

- Matplotlib
- Seaborn

## Machine Learning Evaluation

- Scikit-learn

## Deployment

- Streamlit
- Streamlit Community Cloud

## Version Control

- Git
- GitHub

---

# 🧰 Requirements

The Streamlit application uses the following dependencies:

```text
streamlit==1.65.0
tensorflow==2.21.0
numpy
pillow
```

---

# ▶️ Running the Streamlit Application Locally

Clone the repository:

```bash
git clone https://github.com/mrigankadas743442-cpu/Brain-Tumor-MRI-Streamlit.git
```

Move into the project directory:

```bash
cd Brain-Tumor-MRI-Streamlit
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit URL shown in the terminal.

---

# 📊 Key Findings

The main findings from the project are:

1. The dataset contains **2,450 MRI images** across four classes.
2. The dataset contains moderate class imbalance.
3. Image preprocessing was required to provide a consistent input size for the deep learning models.
4. Data augmentation was used to improve training diversity.
5. The custom CNN achieved **81.30% test accuracy**.
6. Transfer learning improved performance for several pretrained architectures.
7. ResNet50 achieved **78.05% test accuracy**.
8. MobileNetV2 achieved **84.15% test accuracy**.
9. InceptionV3 achieved **86.18% test accuracy**.
10. EfficientNetB0 achieved the highest test accuracy of **88.21%**.
11. EfficientNetB0 also achieved the highest precision, recall, and F1-score among the evaluated models.
12. EfficientNetB0 was therefore selected for the final Streamlit application.
13. The deployed Streamlit application successfully produced predictions for all four tested classes.

---

# 📌 Model Selection

The final model was selected based on test-set performance.

EfficientNetB0 achieved:

```text
Accuracy  : 88.21%
Precision : 88.23%
Recall    : 88.21%
F1-Score  : 88.07%
```

Since it achieved the strongest overall results among the evaluated models, EfficientNetB0 was selected for deployment.

---

# ⚠️ Limitations

This project has several limitations:

- The dataset contains a relatively limited number of images compared with large-scale medical imaging datasets.
- There is some class imbalance between the four categories.
- Model performance depends on the characteristics and distribution of the available dataset.
- The model has not been clinically validated.
- The Streamlit application is not intended to replace professional medical examination.
- Prediction confidence represents model probability and does not represent clinical certainty.
- External clinical validation would be required before considering real-world medical use.

---

# 🔮 Future Improvements

Potential future improvements include:

- Increasing the size and diversity of the dataset.
- Using additional MRI datasets for external validation.
- Performing controlled fine-tuning of pretrained models.
- Applying more advanced hyperparameter optimization.
- Investigating class-balancing strategies.
- Improving model interpretability.
- Adding explainable AI techniques such as Grad-CAM.
- Performing external validation on unseen datasets.
- Improving the Streamlit user interface.
- Adding additional model monitoring and validation capabilities.

---

# 📦 Project Deliverables

The project includes the following deliverables:

### 1. Trained Models

Custom CNN and pretrained transfer-learning models saved in `.h5` format.

### 2. Streamlit Application

An interactive application for uploading MRI images and obtaining model predictions.

### 3. Training and Evaluation Notebook

The notebook contains:

- Dataset understanding
- Exploratory data analysis
- Data preprocessing
- Data augmentation
- Model building
- Model training
- Model evaluation
- Confusion matrices
- Training history visualization
- Model comparison
- Deployment documentation

### 4. Model Comparison

Five models were trained and compared:

```text
Custom CNN
ResNet50
MobileNetV2
InceptionV3
EfficientNetB0
```

### 5. GitHub Repository

The project code and Streamlit deployment files are maintained in a public GitHub repository.

---

# 🔗 Important Links

### Live Streamlit Application

https://brain-tumor-mri-image-analysis.streamlit.app/

### GitHub Repository

https://github.com/mrigankadas743442-cpu/Brain-Tumor-MRI-Streamlit

---

# 🏷️ Technical Tags

```text
Deep Learning
Python
TensorFlow
Keras
CNN
Convolutional Neural Network
Transfer Learning
Computer Vision
Medical Imaging
MRI Image Classification
Brain Tumor Classification
Image Classification
Data Preprocessing
Data Augmentation
Image Processing
ResNet50
MobileNetV2
InceptionV3
EfficientNetB0
Model Training
Model Evaluation
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Model Comparison
Streamlit
Streamlit Community Cloud
GitHub
Healthcare AI
```

---

# 👨‍💻 Project

**Brain Tumor MRI Image Classification**

Developed as a deep learning and computer vision project demonstrating the application of CNNs and transfer learning to MRI image classification.

# 👨‍💻 Author
Mriganka Das

MCA Graduate | AI/ML & Data Science Enthusiast

GitHub:

https://github.com/mrigankadas743442-cpu
---

# ⚕️ Disclaimer

This project is developed for **educational and AI-assisted research purposes only**.

The predictions generated by this application are produced by a machine learning model and should **not be considered a medical diagnosis, medical advice, or a replacement for evaluation by a qualified healthcare professional**.

Any real-world medical application would require appropriate clinical validation, regulatory review, expert oversight, and testing on representative clinical datasets.
