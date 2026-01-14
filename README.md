📉 Customer Churn Prediction with Explainable ML & Web App
📌 Overview

This project predicts customer churn using machine learning models and delivers insights through an interactive Streamlit web application.
In addition to predictions, the app provides model explanations and personalized retention emails generated using an LLM.

The goal is to help businesses identify at-risk customers and take action before churn happens.

🧠 Key Features
- End-to-end ML pipeline (EDA → Modeling → Deployment)
- Multiple ML models with performance comparison
- Feature engineering & SMOTE for imbalance handling
- Explainable churn prediction
- LLM-generated explanations & customer emails
- Interactive Streamlit dashboard

🗂️ Project Structure
```
Customer_Churn_Prediction/
│
├── streamlit_app.py # Main Streamlit application
├── utils.py # Visualization & helper functions
├── requirements.txt # Dependencies
│
├── churn.csv # Dataset 1 (raw customer data)
├── churn_modeling.csv # Dataset 2 (used for training models)
│
├── *.pkl # Saved ML models
├── *.ipynb # EDA, preprocessing, training & experiments
└── README.md
```

📊 Data Analysis & Modeling
**Exploratory Data Analysis**
- Churn distribution
- Age distribution
- Credit score vs age
- Balance vs churn
- Feature impact analysis

**Preprocessing**
- Missing value handling
- Categorical encoding
- Feature scaling
- Train-test split

🤖 Machine Learning Models Used
- Logistic Regression
- Random Forest
- Decision Tree
- KNN
- Naive Bayes
- SVM
- Gradient Boosting
- XGBoost
- Stacking Classifier
- Voting Classifier
**Performance Improvements**
- Feature engineering (CLV, Age Groups, Ratios)
- SMOTE for class imbalance
- Ensemble learning

📈 Accuracy improved from ~74% to ~85%

🌐 Streamlit Web Application
- streamlit_app.py
- User selects a customer
- Inputs auto-filled from dataset
- Predicts churn probability
- Shows:
  - Gauge chart
  - Model-wise probabilities
  - Explanation of churn risk
  - Personalized retention email
- utils.py
- Gauge chart visualization
- Model probability bar chart

🧠 Explainable AI & LLM Integration
- Uses Groq LLM
- Generates:
  - Human-readable explanation of churn
  - Customer-friendly retention email
- No ML jargon exposed to end users

🚀 How to Run Locally
pip install -r requirements.txt
streamlit run streamlit_app.py

🧪 Skills Demonstrated
- Python
- Machine Learning
- Feature Engineering
- SMOTE & Ensembling
- Streamlit
- Model Deployment
- Explainable AI
- LLM Integration
  
🔮 Future Enhancements
- SHAP-based explanations
- Cloud deployment (AWS / GCP)
- Model comparison dashboard
- Real-time data ingestion
