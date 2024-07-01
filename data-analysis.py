#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:10:57 2024

@author: isa

Citation: The seeds dataset used in this exercise was originally published by the Institute of Agrophysics of the Polish Academy of Sciences in Lublin by Dua, D. and Graff, C. (2019). and can be downloaded from the UCI Machine Learning Repository, University of California at Irvine, School of Information and Computer Science.

"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import requests
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Descargar el dataset
url = 'https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/seeds.csv'
response = requests.get(url)
with open('seeds.csv', 'wb') as f:
    f.write(response.content)

# Cargar el dataset
data = pd.read_csv('seeds.csv')

# Mostrar una muestra aleatoria de 10 observaciones (solo las características)
features = data[data.columns[0:6]]
print(features.sample(10))

# Normalizar las características numéricas para que estén en la misma escala
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Obtener dos componentes principales
pca = PCA(n_components=2)
features_2d = pca.fit_transform(scaled_features)
print(features_2d[0:10])

# Graficar los componentes principales
plt.scatter(features_2d[:, 0], features_2d[:, 1])
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.title('Datos')
plt.show()

# Crear 10 modelos con 1 a 10 clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    # Ajustar los puntos de datos
    kmeans.fit(scaled_features)
    # Obtener el valor de WCSS (inercia)
    wcss.append(kmeans.inertia_)

# Graficar los valores de WCSS en un gráfico de líneas
plt.plot(range(1, 11), wcss)
plt.title('WCSS por número de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

# Crear un modelo basado en 3 centroides
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Ajustar a los datos y predecir las asignaciones de clusters para cada punto de datos
km_clusters = model.fit_predict(features.values)
# Ver las asignaciones de clusters
print(km_clusters)

# Función para graficar clusters
def plot_clusters(samples, clusters):
    col_dic = {0: 'blue', 1: 'green', 2: 'orange'}
    mrk_dic = {0: '*', 1: 'x', 2: '+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color=colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.title('Asignaciones')
    plt.show()

plot_clusters(features_2d, km_clusters)

# Graficar clusters basados en especies de semillas
seed_species = data[data.columns[7]]
plot_clusters(features_2d, seed_species.values)

# Crear un modelo de clustering aglomerativo
agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(features.values)
print(agg_clusters)

# Graficar clusters aglomerativos
plot_clusters(features_2d, agg_clusters)

# Calcular y mostrar el coeficiente de silueta para KMeans
silhouette_kmeans = silhouette_score(scaled_features, km_clusters)
print(f'Coeficiente de silueta para KMeans: {silhouette_kmeans}')

# Calcular y mostrar el coeficiente de silueta para Agglomerative Clustering
silhouette_agg = silhouette_score(scaled_features, agg_clusters)
print(f'Coeficiente de silueta para Agglomerative Clustering: {silhouette_agg}')

# Obtener tres componentes principales
pca_3d = PCA(n_components=3)
features_3d = pca_3d.fit_transform(scaled_features)

# Graficar los clusters en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=km_clusters, cmap='viridis', s=50)
ax.set_xlabel('Dimensión 1')
ax.set_ylabel('Dimensión 2')
ax.set_zlabel('Dimensión 3')
ax.set_title('Clusters KMeans en 3D')
plt.show()

# Análisis de la varianza explicada por PCA
explained_variance = pca.explained_variance_ratio_
print(f'Varianza explicada por cada componente principal: {explained_variance}')
print(f'Varianza total explicada por los dos primeros componentes: {np.sum(explained_variance)}')

# Crear y ajustar el modelo DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan_model.fit_predict(scaled_features)

# Graficar los clusters resultantes de DBSCAN
plot_clusters(features_2d, dbscan_clusters)

# Calcular la matriz de correlación
corr_matrix = features.corr()

# Graficar el heatmap de la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación de las Características')
plt.show()