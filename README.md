### **Corporation Favorita Time Series Forecasting**
This project involves developing a time series forecasting solution for Corporation Favorita, a large Ecuadorian-based grocery retailer. The goal is to predict future sales using historical sales data, following the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. We employ several forecasting models, including Linear Regression, ARIMA, SARIMA, Random Forest, XGBoost, and Prophet, with an emphasis on hyperparameter tuning, model evaluation, and model storage.

#### **Table of Contents**
- Project Overview
- CRISP-DM Methodology
- Data Description
- Exploratory Data Analysis
- Feature Engineering
- Modeling
    1. Linear Regression
    2. ARIMA
    3. SARIMA
    4. Random Forest
    5. XGBoost
    6. Prophet
- Hyperparameter Tuning
- Model Evaluation
- Model Storage
- Conclusion

### **Business Understanding**

#### **Description**: 
Corporation Favorita is a chain of Ecuadorian-owned grocery stores, offering a wide range of products to customers across various locations.

The company wish to build accurate Machine Learning models for forcasting unit sales of various products in its different stores across the nation. The company aims to improve its sales forecasting accuracy to ensure optimal inventory management and minimize operational costs and increased customer satisifcation. To achieve this goal, she will be leveraging on historical sales data, product information, promotion details, and other external factors like oil prices and holidays to build time series regression model using the CRISP-DM framework.

#### **Problem Statement**

As a data scientist at Corporation Favorita, I am tasked with developing machine learning models to improve the accuracy of unit sales forecasts for thousands of products across various store locations. Inaccurate forecasts lead to overstocking or understocking, which can result in lost sales, wasted resources, and customer dissatisfaction. By leveraging historical sales data, product information, promotion details, and external factors, I aim to build robust time series regression models that can predict future sales with greater accuracy.

#### **Goals, Objectives and Methodology**

##### Goals
1. Reduce inventory management costs by optimizing stock levels based on forecasted demand.

2. Enhance customer satisfaction by ensuring product availability and minimizing stockouts.

3. Improve decision-making for promotional activities by identifying products with high sales potential during promotions.

##### Objectives
1. Understand the data: The first objective is to gain insights into the store sales data, including store-specific information, product families, promotions, and sales numbers. This understanding will enable the company to make informed business decisions.

2. Predict store sales: Develop a reliable time series forecasting model that accurately predicts the unit sales for different product families at various Favorita stores. This will help the company optimize inventory management, plan promotions, and improve overall sales performance.

##### Methodology
1. Data Exploration: Thoroughly explore the provided datasets to understand the available features, their distributions, and relationships. This step will provide initial insights into the store sales data and help identify any data quality issues.

2. Data Preparation: Handle missing values, perform feature engineering, and encode categorical variables as necessary. This step may involve techniques like imputation, scaling, and one-hot encoding.

3. Time Series Analysis: Perform exploratory data analysis and time series decomposition to understand the underlying patterns and trends in the store sales data. This step will help identify any seasonality, trends, and cyclical components in the data.

4. Feature Selection: Identify the most relevant features for forecasting store sales using techniques like correlation analysis, statistical tests, or machine learning algorithms. This step will help the company select the most informative predictors for building the time series regression model.

5. Model Selection: Evaluate different time series regression models, such as ARIMA, SARIMA, Prophet, and LSTM, using appropriate evaluation metrics like Mean Absolute Error (MAE), root mean squared error (RMSE), or mean absolute percentage error (MAPE) and cross-validation techniques. Choose 2 models that performs best on the test data.

6. Model Evaluation: Evaluate the performance of the selected time series regression model using appropriate evaluation metrics like MAE and cross-validation techniques. This step will help the company assess the model's accuracy and determine if further improvements are needed.

7. Deployment: Once the best-performing time series regression models are selected, deploy it as a web service or API for real-time sales forecasting. This step will enable the company to make informed business decisions and provide accurate sales forecasts to.

#### **Stakeholders**

1. *Inventory Management Team*: Benefits from optimized stock levels, leading to reduced costs and improved efficiency.

2. *Sales and Marketing Team*: Gains insights into product demand and can leverage forecasts for targeted promotions and marketing campaigns.

3. *Store Operations Team*: Can utilize forecasts for scheduling staff and allocating resources based on anticipated demand.

4. *Senior Management*: Has access to data-driven insights for strategic decision-making concerning inventory management, pricing, and promotions.

5. *Finance Team*: Can utilize forecasts for budgeting and forecasting expenses related to inventory management.

6. *Customer Service Team*: Ensures product availability and minimizes stockouts, leading to increased customer satisfaction.

#### **Key Metrics and Success Criteria**

##### **Key Metrics**
1. *Mean Absolute Error (MAE)*: Measures the average magnitude of the errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

2. *Mean Squared Error (MSE)*: Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and what is estimated.

3. *Root Mean Squared Error (RMSE)*: Measures the square root of the average squared differences between predicted and actual values.

4. *Root Mean Squared Logarithmic Error (RMSLE)*: Measures the ratio between predicted and actual values by applying a logarithmic transformation. It is particularly useful when the target values span several orders of magnitude.

5. *R-squared (R²)*: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

##### **Success Criteria**

1. `Excellent Model Performance`

    - RMSLE: < 0.2
    - MAE: Low value relative to the average sales (e.g., less than 10% of the average sales value)
    - R²: > 0.85

2. `Good Model Performance`

    - RMSLE: < 0.3
    - MAE: Moderate value relative to the average sales (e.g., less than 15% of the average sales value)
    - R²: > 0.75

3. `Fair Model Performance`

    - RMSLE: < 0.4
    - MAE: Higher value relative to the average sales (e.g., less than 20% of the average sales value)
    - R²: > 0.65

4. `Business Impact and Utility`

    - *Stock Management*: The model should significantly improve stock management by reducing overstock and stockouts, leading to better inventory turnover ratios.

    - *Sales Forecast Accuracy*: The model should provide accurate sales forecasts to enable effective decision-making for promotions, pricing strategies, and demand planning.

    - *Operational Efficiency*: Improved forecasting should lead to operational efficiencies in supply chain management and logistics.

    - *Customer Satisfaction*: Better stock availability should enhance customer satisfaction and reduce the likelihood of lost sales due to stockouts.

#### **Hypothesis**

##### **Null Hypothesis (H0)**
Promotions do not affect the sales significantly.

##### **Alternative Hypothesis (H1)**
Promotions affect the sales significantly.

#### **Analytical Questions**

1.	Is the train dataset complete (has all the required dates)?

2.	Which dates have the lowest and highest sales for each year (excluding days the store was closed)?

3.	Compare the sales for each month across the years and determine which month of which year had the highest sales.

4.	Did the earthquake impact sales?

5.	Are certain stores or groups of stores selling more products? (Cluster, city, state, type)

6.	Are sales affected by promotions, oil prices and holidays?

7.	What analysis can we get from the date and its extractable features?

8.	Which product family and stores did the promotions affect?

9.	What is the difference between RMSLE, RMSE, MSE (or why is the MAE greater than all of them?)

10.	Does the payment of wages in the public sector on the 15th and last days of the month influence the store sales?

#### **Column Descriptions**

1. **id**: Unique identifier for each record
2. **date**: Date of the sales record
3. **store_nbr**: Unique store number
4. **family**: Product category
5. **sales**: Number of units sold
6. **onpromotion**: Number of items on promotion
7. **year**: Year of the sales record
8. **month**: Month of the sales record
9. **week**: Week of the sales record
10. **day_name**: Day of the week
11. **city**: City where the store is located
12. **state**: State where the store is located
13. **type_x**: Type of the store
14. **cluster**: Cluster/Group of the store (Similar store are in the same cluster)
15. **transactions**: Number of transactions recorded in the store on the given date
16. **dcoilwtico**: Daily oil price in Ecuador
17. **type_y**: Type of holiday or event
18. **locale**: Scope of the holiday or event (e.g. national, regional)
19. **locale_name**: Specific location name of the holiday or event
20. **description**: Description of the holiday or event
21. **transferred**: Indicator if the holiday or event was transferred to another date.

### **Data Understanding**
The Data Understanding is divided into three stages which are:

1. *Project Initialization*

2. *Data Collection*

3. *EDA & Data Cleaning*

#### **Project Initialization**

This stage involved the loading of all relevant libraries for the project.

#### **Data Collection and Loading**

This involves accessing different datasets from 3 different sources: 

- A Database 
- OneDrive
- A GitHub repository 

Each of these datasets has a specific mode of accessing it, which are: 

- Querying a database with login credentials
- Downloading CSV files from the OneDrive and uploading them using the downloaded libraries
- Downloading CVS files from the provided GitHub repository and uploading using the expected libraries.

#### **EDA & Data Cleaning**
1. *Data Quality Assessment & Data exploration*
    
    This stage includes:
    - Convert the 'date' columns to datetime format
    - Categorization of Products in the Family Field
    - Handling missing values.
    - Checking and handling duplicates
    - Creating new time-related features (e.g., year, month, day).
    - Merging all datasets using JOINS (LEFT & INNER) using the train_data_cpy as priority
    - Spliting the holidays in the holiday dataframe to either National, Regional, and Local holidays
    - Renaming columns to city and state names in the specific holiday dataframes
    - Decomposing the time series to identify trends, seasonality, and residuals.
    - Stationarity tests, such as the Augmented Dickey-Fuller (ADF) test, to ensure the series is ready for time series modeling.
