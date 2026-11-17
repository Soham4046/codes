# Import required libraries
import pandas as pd
import seaborn as sns

# Load an open-source dataset
# Tips dataset contains information about restaurant bills and tips
df = sns.load_dataset("tips")

# Display first five records
print("First Five Records:")
print(df.head())

# Display dataset description
print("\nDataset Description:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display dimensions of dataset
print("\nRows and Columns:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display data types
print("\nData Types:")
print(df.dtypes)

# Convert categorical columns into category data type
df["sex"] = df["sex"].astype("category")
df["smoker"] = df["smoker"].astype("category")
df["day"] = df["day"].astype("category")
df["time"] = df["time"].astype("category")

# Convert categorical variables into numerical variables
df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "day", "time"],
    dtype=int
)

# Display converted dataset
print("\nDataset After Categorical Encoding:")
print(df_encoded.head())