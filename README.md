# CSE366-Streamlit-App-
#  CSE366 — Artificial Intelligence  
### Streamlit App with Model Selection & XAI Visualizations  

##  Project Overview
This project is a **Streamlit web application** that allows users to:  
- Select one of multiple trained deep learning models.  
- Upload an image (or use a sample image).  
- Run prediction on the selected model.  
- Visualize **five Explainable AI (XAI) methods** for the same prediction.  

This app demonstrates how model predictions can be made more transparent using visualization techniques.

---

##  Repository Structure

CSE366/
│── app.py # Main Streamlit app
│── requirements.txt # Python dependencies
│── custom_cnn_model.pth # Custom CNN weights
│── transfer_learning_resnet50.pth # ResNet50 weights
│── transfer_learning_vgg16.pth # VGG16 weights
│── transfer_learning_densenet121.pth # DenseNet121 weights
│── transfer_learning_efficientnet_b0.pth # EfficientNet-B0 weights
└── README.md # Documentation

Models Included:

1. Custom CNN (baseline)

2. ResNet50 (Transfer Learning)

3. VGG16 (Transfer Learning)

4. DenseNet121 (Transfer Learning)

5. EfficientNet-B0 (Transfer Learning)

Explanation Methods (XAI)

The app provides five explanation methods to interpret predictions:

i. Grad-CAM

ii. Grad-CAM++

iii. Eigen-CAM

iv. Ablation-CAM

v. LIME

Contribution Note:
This was originally a group project, but I completed individually.
Name: Mahfuz Uddin Ahmed
ID: 2023-1-60-207
Role: Implemented all models, integrated XAI visualizations, developed Streamlit app, prepared documentation, and completed demo.
