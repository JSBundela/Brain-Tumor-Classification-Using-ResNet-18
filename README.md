# 🧠 Brain Tumor Classification using ResNet-18

## 📌 Project Overview
This project focuses on **classifying brain tumors** using **deep learning and transfer learning** with a **ResNet-18** model. The dataset consists of MRI images categorized into **Glioma, Meningioma, Pituitary Tumor, and No Tumor** classes. The model is trained using **PyTorch** and optimized with **data augmentation, stratified sampling, hyperparameter tuning, and regularization** to enhance performance.

## 🚀 Key Features
- ✅ **Deep Learning with Transfer Learning:** Used **pretrained ResNet-18** on ImageNet and fine-tuned it for MRI tumor classification.
- ✅ **Class Imbalance Handling:** Implemented **stratified sampling** to ensure balanced class representation.
- ✅ **Performance Optimization:** Used **hyperparameter tuning, regularization, and early stopping** to enhance model generalization.
- ✅ **Comprehensive Evaluation:** Evaluated performance using **confusion matrix, classification report, and accuracy metrics**.

## 📂 Dataset
The dataset used for this project is the **Brain Tumor MRI Dataset**, which contains MRI images classified into four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

**Dataset Link:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

## 🏗 Model Architecture
We leveraged **ResNet-18**, a powerful CNN architecture, with the following modifications:
1. **Pretrained Model Loading:**
   ```python
   model = models.resnet18(pretrained=True)  # ✅ Transfer Learning from ImageNet
   num_features = model.fc.in_features
   model.fc = nn.Linear(num_features, len(CATEGORIES))  # ✅ Custom classifier
   model = model.to(device)
   ```
2. **Data Preprocessing:**
   - **Normalization & Resizing:** Images are resized to 224x224 and normalized.
   - **Data Augmentation:** Applied **random rotations, flips, and brightness adjustments** to improve generalization.

3. **Training & Evaluation:**
   - **Loss Function:** CrossEntropyLoss
   - **Optimizer:** Adam
   - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and Confusion Matrix

## 📊 Results
| Class       | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| Glioma     | 0.95      | 0.98   | 0.96     | 264     |
| Meningioma | 0.98      | 0.97   | 0.98     | 268     |
| Pituitary  | 0.99      | 0.96   | 0.98     | 292     |
| No Tumor   | 0.99      | 0.99   | 0.99     | 319     |
| **Overall Accuracy** | **98%** |

## 🛠 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the training script:**
   ```bash
   python train.py
   ```
4. **Test the model on an image:**
   ```bash
   python predict.py --image path/to/image.jpg
   ```

## 📝 Future Improvements
- Implementing **ResNet-50 or EfficientNet** for better accuracy.
- Developing a **web-based interface using Streamlit** for easy usability.
- Extending the dataset for better generalization.

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## 📜 License
This project is licensed under the **MIT License**.



