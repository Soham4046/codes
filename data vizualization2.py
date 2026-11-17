# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Remove rows where age is missing
df = df.dropna(subset=["age"])

# Create box plot of age by gender and survival
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="sex",
    y="age",
    hue="survived"
)

plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# Display average age by gender and survival
print("Average Age:")
print(df.groupby(["sex", "survived"], observed=True)["age"].mean())

# Display median age by gender and survival
print("\nMedian Age:")
print(df.groupby(["sex", "survived"], observed=True)["age"].median())