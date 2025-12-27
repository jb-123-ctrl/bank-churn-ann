
# 🏦 Bank Customer Churn Prediction using ANN

This project is an end-to-end **Deep Learning classification application** that predicts whether a bank customer is likely to leave the bank (churn) or stay.

The model is built using **Artificial Neural Networks (ANN)** with **TensorFlow and Keras**, and deployed as an interactive **Streamlit web application**.

---

## 🚀 Live Demo
👉 https://bank-churn-ann-ynb8a6we2qtgvg7muudzbf.streamlit.app/

---

## 📊 Dataset
- **Source:** Kaggle – Churn Modelling Dataset
- **File:** `Churn_Modelling.csv`
- **Records:** 10,000 customers
- **Target Variable:** `Exited`
  - `0` → Customer stays
  - `1` → Customer leaves

### Features Used:
- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Has Credit Card  
- Is Active Member  
- Estimated Salary  

---

## 🧠 Model Architecture
- Input Layer: 11 features
- Hidden Layer 1: Dense (16 neurons, ReLU)
- Dropout: 0.3
- Hidden Layer 2: Dense (8 neurons, ReLU)
- Dropout: 0.3
- Output Layer: Dense (1 neuron, Sigmoid)

---

## ⚙️ Technologies Used
- Python 3
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Streamlit

---

## 📈 Model Performance
- **Test Accuracy:** ~86%
- Optimizer: Adam
- Loss Function: Binary Crossentropy

---

## 🌐 Streamlit Web App
The Streamlit app allows users to:
- Enter customer details
- Predict churn probability
- View real-time results

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
