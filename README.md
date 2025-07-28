# Demand_forecasting
Python ML project for demand forecasting, predicting units_sold using a Random Forest Regressor. Addresses a regression problem. Key pre-processing (outlier handling, one-hot encoding store_ID/SKU_ID)  significantly improved RMSE from 29.9 to 17.7, optimising inventory &amp; sales



Demand Forecasting Project
Project Overview
This project focuses on building a demand forecasting model using machine learning in Python to predict the units_sold for various items in different stores. The primary goal is to estimate a numerical value (the number of units sold) rather than classifying data into categories, making it a regression problem. This solution can be valuable for optimising inventory, sales strategies, and overall business operations.


Data Description
The project utilises a Kaggle dataset called "Demand Forecasting", specifically the train.csv file, as the test.csv file does not contain the target variable (units_sold).
The initial raw dataset train.csv contains the following nine columns:
• record_ID
• week
• store_ID
• SKU_ID
• total_price
• base_price
• is_featured.
• is_displayed
• units_sold


Data Pre-processing Steps
Significant pre-processing was performed to enhance the model's performance:
1. Feature Extraction from week: The week feature, initially a string, was split into three new numerical columns: day, month, and year. 
2. Dropping record_ID
3. Outlier Handling for units_sold
4. One-Hot Encoding for Identifiers:
    ◦ store_ID: Although numerically represented, store_ID is an identifier and treated as a categorical feature. There are 76 unique store_ID values. One-hot encoding was applied using pd.get_dummies(), transforming the single store_ID column into 76 new binary columns, each representing a unique store (e.g., store_1234). This approach gives more relevance and information to the feature, as stores with numerically close IDs are not necessarily similar.
    ◦ SKU_ID: Similarly, SKU_ID is an identifier rather than a numerical feature. There are 28 unique SKU_ID values. One-hot encoding was applied, creating 28 new binary columns for each unique item (e.g., item_SKU_ID_ABC).
   
After these transformations, the dataset grew from the initial 9 columns to a total of 112 columns, which is considered manageable for the given number of unique stores and items.
Machine Learning Model and Evaluation

1. Model Choice: A RandomForestRegressor from sklearn.ensemble was chosen for this regression task.
3. Training and Prediction: The RandomForestRegressor learns by constructing multiple decision trees . Each decision tree internally learns a series of conditional rules (e.g., "if total_price > X AND is_featured is True, then predict Y units sold") to partition the data .
   
5. Evaluation Metrics: Model performance was assessed using:
    ◦ R-squared (R²): A statistical measure representing the proportion of variance in the dependent variable that the independent variables can explain.
    ◦ Root Mean Squared Error (RMSE): This metric is in the same units as the units_sold and provides a more intuitive understanding of the average prediction error.
Results and Improvements

Initially, a baseline model without extensive pre-processing yielded an RMSE of approximately 29.9 units.
After implementing the data pre-processing steps, particularly the outlier handling and one-hot encoding of store_ID and SKU_ID, the model's performance significantly improved, achieving an RMSE of approximately 17.7 units. This considerable reduction in RMSE indicates a much more accurate prediction of units_sold. Visualisations confirmed that the predictions were much closer to the actual values after pre-processing.

Hyperparameter Tuning
Further efforts were made to improve the model through hyperparameter tuning using GridSearchCV from sklearn.model_selection. This process involves defining a parameter grid (a dictionary of hyperparameter names and their values to be tested) and allowing GridSearchCV to try all possible combinations of these parameters, often with cross-validation (e.g., 3-fold cross-validation). Common hyperparameters tuned for RandomForestRegressor include n_estimators (number of trees) and min_samples_split (minimum samples required to split a node). While a comprehensive tuning was not performed in the tutorial due to time constraints, the methodology for finding the best estimator and parameters was demonstrated.
Technologies Used
• Python 3.x
• Google Collab
• NumPy: For numerical operations.
• Pandas: For data manipulation and analysis, especially DataFrames.
• Matplotlib: For data visualisation.
• Scikit-learn: For machine learning model implementation, including RandomForestRegressor, train_test_split, root_mean_squared_error, and GridSearchCV.
