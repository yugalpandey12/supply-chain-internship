import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the datasets
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
train = train.sample(n=5000, random_state=42)

#%% Step 2: Data Preprocessing
# Merging datasets
data = train.merge(stores, how='left', on='Store')
data = data.merge(features, how='left', on=['Store', 'Date'])

# Converting Date to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sorting data by Date
data = data.sort_values(by='Date')

# Step 3: Handling Missing Values
# Identify numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns

# Fill missing values for numeric columns with median
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Fill missing values for categorical columns with the mode
data = data.fillna(data.mode().iloc[0])

#%% Step 4: Feature Engineering
# Extracting additional features from the Date column
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.isocalendar().week
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek

# Check if categorical columns exist before encoding
categorical_columns = data.select_dtypes(include=['object']).columns

# One-Hot Encoding for categorical features
if len(categorical_columns) > 0:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

#%% Step 5: Preparing Data for Modeling
# Defining features (X) and target (y)
X = data.drop(columns=['Weekly_Sales', 'Date'])
y = data['Weekly_Sales']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Step 6: Model Training
# Training a RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#%% Step 7: Model Evaluation
# Making predictions
y_pred = model.predict(X_test)

# Calculating metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

#%% Step 8: Feature Importance
# Plotting feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 8))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

#%% Step 9: Visualizations
# Visualizing sales trends using Plotly
fig = px.line(data, x='Date', y='Weekly_Sales', title='Sales Trend Over Time', labels={'Date': 'Date', 'Weekly_Sales': 'Weekly Sales'})
fig.update_layout(xaxis_title='Date', yaxis_title='Weekly Sales')
fig.show()

# Visualizing sales by store type if available
store_type_columns = [col for col in data.columns if 'StoreType' in col]
if len(store_type_columns) > 0:
    fig = px.box(data, x=store_type_columns[0], y='Weekly_Sales', title='Sales Distribution by Store Type', labels={'Weekly_Sales': 'Weekly Sales'})
    fig.update_layout(xaxis_title='Store Type', yaxis_title='Weekly Sales')
    fig.show()

# Visualizing weekly sales by different features
fig = px.box(data, x='Month', y='Weekly_Sales', title='Sales Distribution by Month', labels={'Month': 'Month', 'Weekly_Sales': 'Weekly Sales'})
fig.update_layout(xaxis_title='Month', yaxis_title='Weekly Sales')
fig.show()

fig = px.box(data, x='DayOfWeek', y='Weekly_Sales', title='Sales Distribution by Day of the Week', labels={'DayOfWeek': 'Day of Week', 'Weekly_Sales': 'Weekly Sales'})
fig.update_layout(xaxis_title='Day of the Week', yaxis_title='Weekly Sales')
fig.show()

