# assignment_2
# Car Dheko - Used Car Price Prediction
Using CarDedko Datasets estimating used car price.

# Overview
In the dynamic used car market, accurately predicting vehicle prices is crucial for customer satisfaction and sales optimization. This repository presents a machine learning project that forecasts used car prices based on attributes such as make, model, year, fuel type, transmission, mileage, and city. Utilizing historical data from CarDekho, the project aims to develop a user-friendly tool. The final output is a deployed Streamlit application that allows users to input car details and receive instant price predictions.

# Project Structure

The repository is organized as follows :

Files: cardekho project all cities data cleaning.ipynb,Exploratory Data Analysis for Car price prediction.ipynb,Analysis cars.ipynb,carprice_prediction_streamlit.py

Description: These Python files encompass the entire workflow, including data cleaning, feature engineering, model selection, training, and evaluation. They feature exploratory data analysis (EDA) to identify key factors influencing car prices and implement various machine learning models such as Linear Regression, Decision Trees, Random Forest, and Gradient Boosting, complete with hyperparameter tuning for optimal performance.

Streamlit Application File: carprice_prediction_streamlit.py

Description: A Streamlit-based web application providing an interactive interface for users to input car specifications and obtain predicted prices. The app is designed for ease of use, making it accessible to both technical and non-technical users.

Project Report File: finalreport.pdf

Description: A detailed report documenting the entire project lifecycle, from the initial problem statement and data preprocessing to model evaluation and deployment. It includes the rationale for the chosen methodologies, a summary of results, and insights derived from the analysis.

Contents: These ZIP files include XLSX and CSV datasets.

Getting Started
Prerequisites
To run the Jupyter Notebook and Streamlit application, the following software and libraries are required:

Python 3.7+
Required Libraries:
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
Running the Streamlit Application
Input Fields: The application allows users to input various car attributes, including make, model, year, fuel type, transmission type, and mileage.
Price Prediction: After entering the details, click the 'Predict' button to receive the estimated price of the car.
User Interface: The interface is designed to be intuitive, ensuring a smooth experience for both tech-savvy and non-technical users.
Model Training and Evaluation
Data Cleaning: The dataset undergoes thorough cleaning to address missing values, encode categorical variables, and scale numerical features appropriately.
Model Selection: Multiple models, including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting, are trained and evaluated. Hyperparameter tuning is conducted to optimize model performance.
Evaluation Metrics: Models are evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) values to determine the best model for deployment.
Results
Best Model: The model with the highest R² score and the lowest MSE and MAE was selected for deployment.
Predictive Accuracy: The final model demonstrated strong predictive accuracy, making it reliable for practical applications in estimating used car prices.
Acknowledgements
This project utilizes data from CarDekho and leverages several open-source libraries, including Scikit-learn for machine learning and Streamlit for application deployment.

# References
CarDekho
Scikit-learn Documentation
Streamlit Documentation
