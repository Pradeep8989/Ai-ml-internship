import pandas as pd

# Load dataset
df = pd.read_csv("D:\\Ai-ml-internship\\task1\\Data\\titanic.csv")

# Basic info
print("First 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())
# Handling missing values


df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop('Cabin', axis=1)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
# Encoding categorical variables

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print("\nData after encoding:")
print(df.head())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\nAfter scaling:")
print(df[['Age', 'Fare']].head())
# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize outliers
sns.boxplot(x=df['Fare'])
plt.title("Boxplot for Fare")
plt.show()
plt.savefig("output/boxplot.png")
# Remove outliers using IQR

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_bound) & (df['Fare'] <= upper_bound)]

print("\nAfter removing outliers:")
print(df.shape)