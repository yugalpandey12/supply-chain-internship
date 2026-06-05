# Supply Chain Optimization with Demand Forecasting

## Project Overview
This project aims to optimize the supply chain by predicting weekly sales for various stores using a machine learning approach. We used RandomForestRegressor to model the sales data based on features such as store characteristics, promotions, and external factors. This project provides insights into the drivers of sales and helps in making data-driven decisions for optimizing inventory and supply chain operations.

## Objective
The objective of this project is to predict the `Weekly_Sales` for retail stores using historical data, including features such as promotions, store type, holidays, and seasonal factors. By developing a machine learning model, we aim to:

1. Provide accurate sales predictions for inventory management.
2. Analyze the importance of different features in predicting sales.
3. Visualize sales trends and key metrics to derive insights.

## Dataset
The project uses four datasets:

1. **Features.csv**: Contains additional information about the stores, such as promotional events and weather conditions.
2. **Stores.csv**: Contains store-related information, including store type and assortment type.
3. **Train.csv**: Historical sales data for training the model, including `Weekly_Sales` for each store.
4. **Test.csv**: Test data for evaluating the model.

## Step-by-Step Process

### Step 1: Data Loading
Loaded the datasets using Pandas and merged them to create a unified dataset containing all the relevant information for each store and date.

### Step 2: Data Preprocessing
- **Merging Datasets**: Merged the `features`, `stores`, and `train` datasets.
- **Handling Missing Values**: Filled missing values for numeric columns with their median and for categorical columns with their mode.
- **Feature Engineering**: Extracted features such as `Year`, `Month`, `Week`, `Day`, and `DayOfWeek` from the `Date` column for better analysis.
- **Encoding Categorical Variables**: One-hot encoded categorical variables to make them suitable for machine learning models.

### Step 3: Model Training
Used the **RandomForestRegressor** from scikit-learn to train the model on the processed data. The model was trained to predict the `Weekly_Sales` based on various features.

### Step 4: Model Evaluation
The model was evaluated using:
- **Mean Squared Error (MSE)**: To measure the average squared difference between actual and predicted sales.
- **R-squared (R²)**: To measure the proportion of variance explained by the model.

### Step 5: Feature Importance Analysis
We plotted the top 10 features that contributed most to the model's predictions, providing insights into the factors driving sales.

### Step 6: Visualizations
Used **Plotly** to enhance visualizations for deeper insights into sales trends and feature impacts. The visualizations include:
- **Sales Trend Over Time**: A line plot showing `Weekly_Sales` over the entire timeframe, which helps understand seasonal patterns and trends.
- **Sales Distribution by Store Type**: A boxplot depicting the distribution of sales by different store types to identify performance variations.

## Tools and Technologies
- **Python**: Core language used for analysis and modeling.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-Learn**: For machine learning modeling and evaluation.
- **Seaborn and Plotly**: For data visualization.
- **Matplotlib**: For plotting basic visualizations.

## Installation
To run this project, you need to install the following dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

## Usage
1. Clone the repository.
2. Ensure you have the required datasets in the appropriate directory.
3. Run the Python script:

```bash
python Supply_Chain_Opt.py
```

## Results
- **Mean Squared Error (MSE)**: The error metric used to measure the average squared difference between predicted and actual sales.
- **R-squared (R²)**: Indicates how well the features explain the variance in sales.

The model achieved reasonable accuracy in predicting weekly sales, with the feature importance analysis highlighting key drivers of sales such as promotions, store type, and seasonal factors.

## Future Work
- **Hyperparameter Tuning**: Improve the model by tuning hyperparameters of the RandomForestRegressor.
- **Additional Features**: Incorporate external data like economic indicators or regional events to improve model accuracy.
- **Optimization**: Use optimization techniques to enhance supply chain management based on sales predictions.

