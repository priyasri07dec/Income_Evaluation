# Income_Evaluation

### Project Overview

This project focuses on predicting whether an individual's annual income greater 50K or less than or equals to 50K based on demographic, educational and employment related attributes. The project involves complete end to end Machine Learning workflow including:

* Data Cleaning
* Exploratory Data Analysis(EDA)
* Feature Engineering
* Skewness & Outlier Treatment
* Feature Scaling
* Model Building
* Overfitting Analysis
* Model Comparison
* Streamlit Deployment

### Dataset

Dataset has been taken from Kaggle

The dataset contains information such as:

* Age
* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain/Loss
* Hours per Week
* Native country
* Income Category

### Target Variable

* <=50K
* >50K
  
### Technologies Used
#### Programming Language
* Python

#### Libraries
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

### Data Preprocessing

#### Missing Value Handling
* Replaced "? " , "?" values with NaN
* Filled categorical missing values using mode

#### Feature Engineering
* One hot encoding applied on categorical features
* Label encoding applied on target variable

#### Skewness Treatment
* Log transformation applied on highly skewed numerical features using np.log1p()

#### Outlier Treatment
* IQR method used for outlier handling
* capital-gain and capital-loss excluded due to zero inflated distribution

#### Feature Scaling
* StandardScaler used for normalization

### Exploratory Data Analysis(EDA)
Performed multiple visualization including

* Income distribution analysis
* Correlation heatmap
* Feature importance analysis
* Skewness analysis

### Machine Learning Models Used
1. Logistic Regression
2. Random Forest Classifier
3. Decision Tree Classifier
4. Support Vector Machine(SVM)
5. XGBoost Classifier

### Model Evaluation
Evaluation metrics used:
* Accuracy Score
* Classification Report
* Confusion Matrix
* Overfitting Analysis

### Best Performing Model
XGBoost Classifier

### 🚀 Streamlit Web Application
An interactive Streamlit web app was developed users can"
* Enter demographic and employment details
* Predict income category instantly

### Features
* Interactive UI
* Responsive layout
* Dropdown based user inputs

### 📂Model Deployment Files

* xgb_model.pkl
* scaler.pkl
* model_columns.pkl
* label_encoder.pkl

### Project Screenshot

![App Screenshot]()


