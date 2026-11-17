# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load built-in Titanic dataset
df = sns.load_dataset("titanic")

# Display first five records
print("First Five Records:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Plot histogram of ticket fare
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"].dropna(), bins=30, kde=True)

plt.title("Distribution of Titanic Ticket Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()