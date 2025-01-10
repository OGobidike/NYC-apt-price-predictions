# 🚀 Predicting NYC Apartment Sales Prices: A Machine Learning Pipeline Experiment 🏙️
Hello, World:), and welcome to my project where I dive deep into predicting the sales prices of NYC apartments using a Machine Learning pipeline! This experiment covers the entire process—from cleaning and preprocessing the data to training a Random Forest model and analyzing its performance using Shapley values. It's a fun and informative journey to building a predictive model that can make sense of New York's real estate market.
---

🔗 Data Source: NYC Citywide Rolling Calendar Sales Dataset
Data Timeframe: July 2nd, 2019 - January 11th, 2024
---

🧐 Project Overview
This project is all about predicting the "SALE PRICE" of NYC apartments using various features like:

Location 📍
Property size 🏠
Neighborhood 🌆
The project follows these steps:

1. Problem Definition: What are we trying to predict?
2. Data Exploration: Let's get to know our dataset!
3. Feature Engineering: Cleaning up and creating useful features.
4. Model Design: Building the model and making predictions.
5. Model Evaluation: How well did our model perform?
6. Insights & Interpretation: Understanding which features are most important.

---

💻 Getting Started
Install the Essentials
First, let's make sure you have the necessary libraries installed. Run the following in your terminal:
```python
import pandas as pd
from ydata_profiling import ProfileReport
!pip install category_encoders
```
---

🛠️ Let's Dive Into the Steps
Step 1: Problem Definition 🤔
The mission? Predict apartment sales prices! It's all about making sense of the price of real estate in NYC.

Step 2: Data Exploration 🔍
We'll kick things off by loading the dataset and generating an Exploratory Data Analysis (EDA) report using ydata-profiling. This gives us a snapshot of our data and helps us spot any issues early on.
```python
import pandas as pd
from ydata_profiling import ProfileReport

# Load the dataset
data = pd.read_csv('NYC_Citywide_Rolling_Calendar_Sales_20250106.csv')

# Generate an EDA report
profile = ProfileReport(data, title="NYC Citywide Rolling Calendar Sales Report")
profile
```
---

Step 3: Feature Engineering 🛠️
Now that we know what we're working with, it's time to clean up the data and create some exciting new features. Here's what we do:

Drop irrelevant columns (because no one likes clutter!).
Handle missing values (we don't want missing data ruining our model).
Remove rows with extreme "SALE PRICE" values.
Create new features like average square footage per residential unit (pretty cool, right?).
Step 4: Model Design 🧠
With our data cleaned and ready to go, we design a pipeline that:

Preprocesses numeric and categorical data.
Trains a Random Forest Regressor to predict the apartment sales prices.
Check out the code snippet for how we handle preprocessing:
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define the preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))  # Impute missing numeric values with median
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # OneHot encode categorical features
])

# Combine preprocessing steps for the final pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])
```
---
Step 5: Model Evaluation 📊
After training our model, it's time to evaluate its performance using various metrics like Mean Squared Error (MSE), R², and Mean Absolute Error (MAE). We also perform residual analysis to ensure our model is not making systematic errors.

Here’s a sneak peek at how we evaluate the model:
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Predict and transform back to original scale
y_pred = pipeline.predict(X_test)
y_pred_original = np.expm1(y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_original)
r2 = r2_score(y_test, y_pred_original)
mae = mean_absolute_error(y_test, y_pred_original)
```
---
We also plot the Actual vs. Predicted prices and the Residuals for a visual understanding.
---
Step 6: Feature Importance and Interpretation 🌟
Finally, we take a deep dive into the model’s inner workings using Shapley values. These values tell us how important each feature is in predicting apartment sales prices. We visualize it in a bar plot that gives us insights into which features matter the most.

Here’s how we calculate Shapley values:
```python
import shap

# Generate Shapley values
explainer = shap.Explainer(pipeline.named_steps['regressor'])
shap_values = explainer.shap_values(X_test_preprocessed)

# Create a summary plot of feature importance
shap.summary_plot(shap_values, X_test_preprocessed, plot_type="bar")
```
🔑 Key Insights
* Random Forest works great for predicting apartment sales prices.
* Feature engineering was key to improving model accuracy.
* Shapley values provide valuable insights into the most impactful features, like property size and location.
---
🎉 Conclusion
In this project, we tackled the complex problem of predicting NYC apartment sales prices using a structured and powerful ML pipeline. By combining data preprocessing, feature engineering, and model evaluation, we were able to build a predictive model that performs well on real estate data. Plus, Shapley values helped us interpret which features are the real stars of the show.

Thanks for checking out the project! I hope you found it insightful, and maybe even a little fun. 😄

Feel free to clone this repo and give it a try! 🚀


