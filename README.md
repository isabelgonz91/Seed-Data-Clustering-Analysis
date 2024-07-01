# Seed Data Clustering Analysis

## Overview

This project performs clustering analysis on a seed dataset to identify different species of seeds based on their features. The analysis includes data preprocessing, normalization, dimensionality reduction using Principal Component Analysis (PCA), and clustering using KMeans and Agglomerative Clustering algorithms. The project also evaluates the quality of the clustering using the Silhouette Coefficient and visualizes the results in both 2D and 3D.

## Dataset

The dataset used for this analysis is obtained from [Microsoft Docs](https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/seeds.csv). It contains the following features for each seed:

1. Area
2. Perimeter
3. Compactness
4. Kernel Length
5. Kernel Width
6. Asymmetry Coefficient

## Analysis Steps

1. **Data Downloading and Loading**:
   - The dataset is downloaded from the provided URL and loaded into a Pandas DataFrame.
   
2. **Data Preprocessing**:
   - A random sample of 10 observations is displayed to get an overview of the data.
   - Features are normalized using MinMaxScaler to ensure they are on the same scale.

3. **Dimensionality Reduction**:
   - PCA is applied to reduce the dimensionality of the data to 2 and 3 components, capturing 91.78% of the total variance.

4. **Clustering**:
   - KMeans clustering is performed with 1 to 10 clusters to determine the optimal number of clusters using the Within-Cluster Sum of Squares (WCSS) method.
   - A KMeans model with 3 centroids is fitted, and clusters are visualized in both 2D and 3D.
   - Agglomerative Clustering is also applied to the data for comparison.

5. **Evaluation**:
   - The quality of the clustering is evaluated using the Silhouette Coefficient.
   - The Silhouette Coefficient for KMeans is 0.4089, and for Agglomerative Clustering, it is 0.3772.
   - A heatmap of the correlation matrix of the features is generated to understand feature relationships.

## Results

- **Silhouette Coefficient**:
  - KMeans: 0.4089
  - Agglomerative Clustering: 0.3772

- **PCA Explained Variance**:
  - First Component: 79.78%
  - Second Component: 12.01%
  - Total: 91.78%

- **Visualization**:
  - 3D scatter plot of KMeans clusters.
  - 2D scatter plot of cluster assignments.
  - Heatmap of feature correlations.

## Conclusions

- The clustering algorithms do not perfectly separate the clusters, with KMeans performing slightly better than Agglomerative Clustering.
- PCA effectively reduces the dimensionality while retaining most of the variance, facilitating better visualization.
- High correlations between some features suggest possible redundancy, which could be explored further in future analyses.

## Usage

To run the analysis, ensure you have the required libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests


Run the provided script to perform the clustering analysis and generate the visualizations.

```bash
python clustering_analysis.py

## Repository contents

clustering_analysis.py: The main script containing the analysis.
README.md: This file, providing an overview of the project.


## License
This project is licensed under the MIT License - see the LICENSE file for details.
