# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Boston Housing dataset
df = pd.read_csv("boston_housing.csv")

print("First Five Records:")
print(df.head())

# Separate input features and target variable
X = df.drop("medv", axis=1)
y = df["medv"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

print("\nActual House Prices:")
print(y_test.head())

print("\nPredicted House Prices:")
print(y_pred[:5])

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)