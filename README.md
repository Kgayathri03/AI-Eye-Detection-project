# AI-Based Eye Disease Detection Using Deep Learning

This project uses deep learning to detect eye diseases such as diabetic retinopathy and glaucoma from retinal images. The model is built using Convolutional Neural Networks (CNNs) in Python and trained on a dataset of fundus images.

## 🔍 Project Overview

Eye diseases, if left undetected, can lead to severe vision loss. Early diagnosis through retinal imaging can help in timely treatment. This project aims to assist ophthalmologists by automating the classification of eye diseases using deep learning.

## 🚀 Tech Stack

- **Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib, scikit-learn  
- **Model Type:** Convolutional Neural Network (CNN)  
- **Dataset:** Publicly available retinal image dataset (e.g., Kaggle - APTOS / EyePACS)  

## 🧠 Features

- Image preprocessing: resizing, normalization, and augmentation  
- Binary and multiclass classification (e.g., No Disease vs Diabetic Retinopathy / Glaucoma)  
- Model training and evaluation with accuracy, precision, recall, and confusion matrix  
- Visualization of training loss and accuracy over epochs  
- Predictive interface for testing custom retinal images  

## 📁 Project Structure
├── dataset/
│ ├── train/
│ └── test/
├── model/
│ └── cnn_model.h5
├── notebooks/
│ └── training_notebook.ipynb
├── app/
│ └── predict.py
├── requirements.txt

## 📊 Results

- Achieved ~90% accuracy on validation data  
- High recall for diabetic retinopathy detection  
- Effective image classification with low false negatives  

## 📸 Sample Output

> ![Sample Output](path_to_image_output.png)

## 🧪 How to Run

1. Clone the repository:
```bash
git clone https://github.com/Kgayathri03/eye-disease-detection.git
cd eye-disease-detection
2.Install dependencies:
pip install -r requirements.txt
3.Train the model:
python train.py
4.Run prediction:
python app/predict.py --image path_to_retina_image.jpg

✅ Future Improvements
 - Build a web interface using Flask/Streamlit
 - Expand to more eye diseases
 - Integrate with real-time camera input

🙋‍♀️ Author
Kavalakuntla Gayathri
📧 kavalakuntlagayathri@gmail.com
🔗 GitHub | LinkedIn

