import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Phase 1: Data Collection and Preparation
df = pd.read_csv("steam_store_data_2024.csv")
df.drop(columns=['description'], inplace=True)
df['price'] = df['price'].str.replace('$', '').astype(float)
df['salePercentage'] = df['salePercentage'].str.replace('%', '').astype(float)
df.dropna(inplace=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df)


# Phase 2: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=20, kde=True)
plt.title('Distribution of Game Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['salePercentage'], bins=20, kde=True)
plt.title('Distribution of Sale Percentages')
plt.xlabel('Sale Percentage (%)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='recentReviews', data=df)
plt.title('Price vs. Recent Reviews')
plt.xlabel('Price ($)')
plt.ylabel('Recent Reviews')
plt.show()

# Phase 3: Customer Segmentation
features = ['price', 'salePercentage']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='salePercentage', hue='cluster', data=df, palette='viridis')
plt.title('Customer Segmentation based on Price and Sale Percentage')
plt.xlabel('Price ($)')
plt.ylabel('Sale Percentage (%)')
plt.show()

# Phase 4: Predictive Modeling

X = df[['price', 'salePercentage']]
y = df['recentReviews'] == 'Very Positive' 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

