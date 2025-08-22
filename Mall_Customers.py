# Mall Customer Segmentation using KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv(r"D:\Saksham\Desktop\Gen Ai\Data FIles(Practice)\Mall_Customers.csv")

# Basic EDA
print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe().T, "\n")
df.rename(columns={'Genre': 'Gender'}, inplace=True)
print("Unique values:\n", df.nunique(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")
print("Columns:", df.columns.tolist(), "\n")

# Feature scaling
X = df.drop(['CustomerID', 'Gender'], axis=1)
X_scaled = StandardScaler().fit_transform(X)

# KMeans clustering with evaluation
inertias, sils, Ks = [], [], range(2, 11)
for k in Ks:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, labels))

# Plot Elbow Method & Silhouette Score
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(Ks, inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(Ks, sils, marker='o')
plt.title('Silhouette Score')
plt.xlabel('K')
plt.ylabel('Score')
plt.grid()
plt.tight_layout()
plt.show()

# Best K according to silhouette score
best_k = Ks[np.argmax(sils)]
print(f'Best K according to Silhouette Score: {best_k}')

# Final KMeans model
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualization of Clusters
plt.figure(figsize=(7, 5))
sns.scatterplot(
    x='Annual Income (k$)', y='Spending Score (1-100)',
    hue='Cluster', data=df, palette='Set1', s=100
)

# Plot centroids
centers = scaler = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0], centers[:, 1],
    s=300, c='yellow', label='Centroids', edgecolors='black'
)

plt.title('Mall Customers - KMeans Clustering')
plt.legend()
plt.show()
