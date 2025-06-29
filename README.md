# 🧬 Breast Cancer Classification using Neural Networks

This project develops a **Neural Network-based classifier** to predict whether a tumor is **benign** or **malignant** based on diagnostic features from a breast cancer dataset.

---

## 📌 Objective

To build a **binary classification model** using a neural network to assist in early detection of breast cancer based on real-world medical diagnostic data.

---

## 📁 Project Structure

```
Breast_Cancer_Classification_using_NN/
├── Breast_Cancer_Classification_using_NN.ipynb
├── README.md
└── data.csv (assumed from context, typically the Wisconsin Breast Cancer Dataset)
```

---

## 📊 Dataset Features

- **Mean Radius**
- **Mean Texture**
- **Mean Perimeter**
- **Mean Area**
- **Mean Smoothness**
- ... *(plus more: standard error and worst-case metrics)*  
- **Diagnosis** (Target: `M = Malignant`, `B = Benign`)

Dataset commonly used: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## 🧠 Model Pipeline

1. **Data Preprocessing**
   - Handling missing values
   - Encoding labels (M → 1, B → 0)
   - Feature scaling using MinMaxScaler

2. **Model Building**
   - Using `Sequential` model from TensorFlow/Keras
   - Layers: Dense → ReLU → Dropout → Sigmoid
   - Binary classification output

3. **Training & Evaluation**
   - Loss: Binary Crossentropy
   - Metrics: Accuracy
   - Evaluated using confusion matrix and classification report

---

## ✅ Model Accuracy

- Training Accuracy: ~99%
- Test Accuracy: ~97% (depending on random state)
- Precision, Recall, F1 Score: High for both classes

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## ▶️ How to Run

1. Install the dependencies:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook Breast_Cancer_Classification_using_NN.ipynb
   ```

3. Run all cells to train the model and see the results.

---

## 📈 Visualizations

- Feature distributions (histograms)
- Heatmap for correlation
- Loss vs. Epoch and Accuracy vs. Epoch graphs

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV or Keras Tuner
- Try other ML models (SVM, Random Forest) for comparison
- Use real clinical datasets for external validation
- Build a Streamlit web app for diagnosis support

---

## ⚠️ Disclaimer

This model is for **educational purposes only** and is **not intended for clinical use** without validation by certified medical professionals.

---
