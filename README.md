# prediction_credit
## Dataset

### train.csv

The `train.csv` file is used as the training data for this project. You can download it from Google Drive using the link below:

[Download train.csv](https://drive.google.com/file/d/1u2fbeAC4wgT1uu4_gnrBABU8v6GcV1bw/view?usp=sharing)
# Credit Risk Analysis using XGBoost

## Overview

This project performs credit risk analysis using the XGBoost classifier. The goal is to predict the credit score category of a customer based on their financial data. The project includes data preprocessing, model training, evaluation, and a web app for deployment using Streamlit.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Streamlit Deployment](#streamlit-deployment)
- [Results](#results)


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/credit-risk-analysis.git
2. **Navigate to the project directory:**
    ```bash
    cd credit-risk-analysis
3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
4.  **Ensure your requirements.txt includes the following dependencies:**
    ```bash
    joblib
    matplotlib
    pandas
    seaborn
    scikit-learn
    xgboost
    streamlit

Dataset
The dataset used in this project is train.csv, which contains financial data with a target variable Credit_Score having three classes: Good, Standard, and Poor.

File Path: C:/Users/Dell/New folder/credit/train.csv
Ensure the dataset is placed in the correct directory as specified or update the file path in the code.
Usage
Prepare the Dataset:

Ensure your dataset is clean, with all missing values handled.
The target variable Credit_Score is encoded using LabelEncoder.
 **Train the Model:**
```bash
    python train_model.py


The model's accuracy, confusion matrix, and ROC AUC score will be displayed. The ROC curves for each class will be plotted.

Save the Model:

The trained model and scaler are saved as xgb_model.pkl and scaler.pkl, respectively.

Code Explanation
**Imports:**

Necessary libraries like pandas, seaborn, matplotlib, scikit-learn, and xgboost are imported.
**Loading the Dataset:**

The dataset is loaded using pandas.read_csv().
**Data Preprocessing:**

Missing values are dropped.
The target variable Credit_Score is encoded using LabelEncoder.
Features are converted to dummy variables to handle categorical data.
The features are then scaled using StandardScaler.
**Model Training:**

The XGBClassifier is trained using the processed data.
**Model Evaluation:**

The model's performance is evaluated using accuracy, confusion matrix, and ROC AUC score.
The ROC curve is plotted for each class.
**Model Saving:**

The trained model and scaler are saved for future use.
**Streamlit Deployment**
This project has been deployed using Streamlit, allowing for an interactive web interface where users can input their data and receive predictions.

**Run the Streamlit App:**

To start the Streamlit app, run the following command in your terminal:

```bash
streamlit run app.py
Make sure that your app.py script is correctly set up to load the saved model (xgb_model.pkl) and scaler (scaler.pkl).

Using the Web App:

Open your web browser and go to http://localhost:8501.
Input the necessary features for prediction.
View the predicted credit score category and the model's evaluation metrics.
Results
Accuracy: The model's accuracy is displayed in the console.
Confusion Matrix: The confusion matrix is provided for further analysis.
ROC AUC Score: The ROC AUC score is calculated for multi-class classification using the One-vs-Rest (OvR) strategy.
ROC Curve: The ROC curve is plotted for each class

