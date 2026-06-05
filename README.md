# 🚗 Car Price Prediction Using CNN-BiLSTM Hybrid Deep Learning Model

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-green)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## 📌 Project Overview

This project presents a **Deep Learning-Based Used Car Price Prediction System** that estimates the market value of a vehicle using various attributes such as manufacturer, model, production year, mileage, engine volume, fuel type, accident history, and engineered features.

The system utilizes a **CNN-BiLSTM Hybrid Deep Learning Architecture** to capture complex feature relationships and generate accurate price predictions. A **Streamlit Web Application** is integrated to provide a simple and interactive interface for real-time predictions.

---

## 🎯 Objectives

- Predict used car prices with high accuracy.
- Apply feature engineering techniques to improve model performance.
- Compare deep learning approaches with traditional machine learning models.
- Provide a user-friendly web interface for real-time predictions.
- Assist buyers, sellers, and dealerships in making informed pricing decisions.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Pickle

### Development Tools
- Jupyter Notebook
- VS Code
- Anaconda

---

## 📂 Project Structure

```text
Car-Prediction-System/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── model/
│   ├── car_price_model.h5
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── model_improved/
├── model_tabular/
│
├── app.py
├── model_training.ipynb
├── README.md
└── requirements.txt
```

---

## 📊 Dataset Features

The model uses multiple vehicle attributes including:

- Manufacturer
- Model
- Production Year
- Category
- Fuel Type
- Engine Volume
- Mileage
- Gear Box Type
- Drive Wheels
- Doors
- Wheel Position
- Color
- Airbags
- Accident History
- Brand Value Index
- Safety Features Score
- Comfort Features Score
- Modification Penalty

---

## ⚙️ Data Preprocessing

The following preprocessing techniques are applied:

- Missing Value Handling
- Label Encoding
- Feature Scaling using StandardScaler
- Feature Engineering
- Data Cleaning
- Train-Test Splitting
- Model Input Reshaping

---

## 🧠 Deep Learning Architecture

### CNN-BiLSTM Hybrid Model

The proposed model combines:

### 🔹 CNN Layer
- Extracts meaningful feature patterns
- Learns local feature relationships
- Reduces noise in input data

### 🔹 BiLSTM Layer
- Captures bidirectional dependencies
- Improves contextual understanding
- Enhances feature representation

### 🔹 Dense Layers
- Perform final regression
- Generate accurate price predictions

---

### Model Architecture

```text
Input Features
      │
      ▼
   Conv1D
      │
      ▼
 MaxPooling1D
      │
      ▼
   BiLSTM
      │
      ▼
   Dense
      │
      ▼
 Price Prediction
```

---

## 📈 Model Evaluation

The model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Comparison Models

The CNN-BiLSTM model is compared against:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Results demonstrate improved prediction accuracy and generalization performance.

---

## 🌐 Streamlit Web Application

The project includes a Streamlit-based interactive web application for real-time price prediction.

### Features

✔ Real-Time Car Price Prediction  
✔ User-Friendly Interface  
✔ Automated Data Preprocessing  
✔ Instant Prediction Results  
✔ Input Validation and Error Handling  

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/mohammedifteqhar/Car-Prediction-System.git
cd Car-Prediction-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
jupyter notebook
```

Open:

```text
model_training.ipynb
```

Run all cells.

### Launch the Streamlit Application

```bash
streamlit run app.py
```

---

## 📸 Application Screenshots

### Home Page

_Add screenshot here_

```markdown
![Home Page](assets/homepage.png)
```

### Prediction Result

_Add screenshot here_

```markdown
![Prediction Result](assets/prediction.png)
```

### Model Architecture

_Add screenshot here_

```markdown
![Model Architecture](assets/model_architecture.png)
```

---

## 📋 Sample Workflow

1. Enter vehicle details.
2. Click **Predict Price**.
3. Input data is automatically preprocessed.
4. CNN-BiLSTM model generates a prediction.
5. Predicted market value is displayed instantly.

---

## 🔮 Future Enhancements

- Integration with live automobile marketplaces
- Mobile application support
- Image-based car price estimation
- NLP analysis of vehicle descriptions
- Cloud deployment
- Explainable AI (XAI)
- Real-time market trend analysis

---

## 💡 Key Learning Outcomes

- Deep Learning Model Development
- CNN and BiLSTM Architectures
- Data Preprocessing and Feature Engineering
- Model Evaluation Techniques
- Streamlit Application Development
- Machine Learning Deployment

---

## 👨‍💻 Author

### Mohammed Yaseer

Bachelor of Engineering (Computer Science & Engineering)

**Areas of Interest**

- Deep Learning
- Machine Learning
- Data Science
- Artificial Intelligence

### Connect with Me

- GitHub: https://github.com/mohammedifteqhar
- LinkedIn: *(Add your LinkedIn profile link here)*

---

## 📜 License

This project is developed for **educational and research purposes**.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
