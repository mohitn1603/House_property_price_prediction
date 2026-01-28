# 🏠 House Price Prediction using Machine Learning

This project focuses on predicting **median house prices** using machine learning techniques.  
A **Random Forest Regressor** is trained with proper preprocessing, hyperparameter tuning, evaluation, and an interactive user input system for real-time predictions.

---

##  Project Overview

The notebook performs the complete **end-to-end machine learning pipeline**, including:

- Data loading and inspection
- Exploratory Data Analysis (EDA)
- Data preprocessing (numerical + categorical)
- Model building using Random Forest
- Hyperparameter tuning with RandomizedSearchCV
- Model evaluation using RMSE and R² Score
- Feature importance visualization
- Cross-validation performance analysis
- Saving the trained model
- Predicting house prices from user input

---

##  Dataset

- **Source**: `Data_file.xlsx`
- **Target Variable**: `median_house_value`
- **Features include**:
  - Longitude
  - Latitude
  - Housing median age
  - Total rooms
  - Total bedrooms
  - Population
  - Households
  - Median income
  - Ocean proximity (categorical)

---

##  Technologies & Libraries Used

- **Python**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **Joblib**

---

##  Machine Learning Pipeline

### 🔹 Data Preprocessing
- Missing values handled using `SimpleImputer`
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- Combined using `ColumnTransformer`

###  Model
- **RandomForestRegressor**
- Hyperparameter tuning using **RandomizedSearchCV**
- 5-fold Cross Validation

---

##  Model Evaluation Metrics

- **RMSE (Root Mean Squared Error)**
- **R² Score**
- Cross-validation performance comparison
- Visualization of:
  - Actual vs Predicted prices
  - Top 15 important features

---

##  Model Saving

The trained model is saved using Joblib:

```bash
property_price_model.pkl
```
---

##  User Input Price Prediction

The notebook supports **interactive house price prediction**.

###  How it works:
- The user enters property details such as:
  - Location (latitude & longitude)
  - Population
  - Median income
  - Ocean proximity
  - Housing and household details
- The trained machine learning model processes the input
- The model predicts the **median house value**

###  Sample Output
```
Predicted Median House Value: $408,247.27
```

---

##  How to Run the Project

1. Clone this repository
2. Open `HousePrediction.ipynb` in **Jupyter Notebook** or **Google Colab**
3. Ensure the dataset `Data_file.xlsx` is present and correctly linked
4. Run all the notebook cells
5. Enter property details when prompted to get predictions

---

##  Future Improvements

- Deploy the model as a **web application** using Flask or Streamlit
- Experiment with advanced models like **XGBoost** or **Gradient Boosting**
- Add more **geographic and economic features**
- Enhance **feature engineering** for better accuracy

---

##  Author

**Mohit Nath**  
Machine Learning Enthusiast | CSE Student



