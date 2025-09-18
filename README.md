# 🌾 Rice Leaf Disease Detection using CNN  

## 📌 Project Overview  
This project is a **Convolutional Neural Network (CNN)** based deep learning model to classify rice leaf images into different categories such as **Healthy** or **Diseased**.  
It helps farmers and researchers to detect rice plant diseases early and take preventive measures.  

✅ Achieved **~99% training accuracy** and **100% validation accuracy** on the dataset of rice leaf images.  

---

## 📊 Dataset  
- Total images: **120** (split into train & validation sets)  
- Classes:  
  - 🌱 Healthy  
  - 🍂 Disease1  
  - 🍂 Disease2  
- Train/Validation Split: **75% / 25%**  
- Images organized in folder structure:  

Data/
├── train/
│ ├── healthy/
│ ├── disease1/
│ └── disease2/
└── val/
├── healthy/
├── disease1/
└── disease2/


---

## 🏗️ Model Architecture  
- Base Model: **Transfer Learning (e.g., VGG16 / ResNet / MobileNet)**  
- Layers Added:  
  - Flatten  
  - Dense (ReLU)  
  - Dropout (to reduce overfitting)  
  - Dense (Softmax for classification)  

---

## ⚙️ Training Details  
- Optimizer: **Adam** (Learning rate tuned → best = `0.001`)  
- Batch Size: **64**  
- Data Augmentation applied (rotation, zoom, flip)  
- EarlyStopping & ModelCheckpoint used  

---

## 📈 Results  
- **Training Accuracy:** 99.25%  
- **Training Loss:** 0.0362  
- **Validation Accuracy:** 100%  
- **Validation Loss:** 0.0441  

📊 Evaluation:  
- Confusion Matrix & Classification Report used for detailed analysis  
- Model performs well on unseen test images  

---

## 🚀 Deployment  
The model can be deployed using:  
- **Gradio**: Interactive web interface to upload images & get predictions.  
- **Streamlit** (future scope): For building a professional web app.  
- **TensorFlow Lite**: To deploy on mobile apps.  

Example (Gradio interface):  

```python
!pip install gradio
import gradio as gr
...
interface.launch(share=True)


🔮 Future Improvements
-Add more rice leaf disease categories
-Train on a larger dataset for better generalization
-Deploy as a mobile app using TensorFlow Lite
-Build a Streamlit dashboard for farmers

# Project Structure

Rice-Leaf-Disease-Detection/
 ├── Data/                         # Dataset (train/val folders)
 ├── Rice_leaf_detection.ipynb     # Jupyter Notebook with code
 ├── rice_leaf_disease_model.h5    # Saved trained model
 ├── requirements.txt              # Dependencies
 └── README.md            # Project documentation

🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn (metrics)
- Gradio / Streamlit (deployment)
