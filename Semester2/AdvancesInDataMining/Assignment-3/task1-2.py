# No external libraries are allowed to be imported in this file
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans


# Function to generate the dataset
def generate_blobs_dataset(n_samples, centers, n_features, random_state):
    """
    Generates a 3D dataset using make_blobs.

    Parameters:
    n_samples (int): Number of samples.
    centers (int): Number of centers.
    n_features (int): Number of features (3 for 3D data).
    random_state (int): Seed for reproducibility.

    Returns:
    tuple: Generated data (X, y)
    """
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=random_state)
    return X, y



# Function to plot the original 3D data
def plot_3d_data(X, y):

    x_vals = X[:,0]
    y_vals = X[:,1]
    z_vals = X[:,2]
 
    my_cmap = ListedColormap(['cornflowerblue', 'orange']) 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sctt = ax.scatter3D(x_vals, y_vals, z_vals,
                     alpha=0.8,
                     c=y, 
                     cmap=my_cmap, 
                     marker='^')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax.set_title('3D Scatter Plot of Blobs Dataset')

    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='cornflowerblue', markersize=10, label='True'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='False')
    ]

    ax.legend(handles=legend_elements, loc='upper right')
    plt.savefig('3d_data.png')
    plt.show()



# Function to standardize the dataset
def standardize_data(X):
    """
    Standardizes the dataset using StandardScaler.

    Parameters:
    X (array): Original dataset.

    Returns:
    array: Scaled dataset.
    """
    # TO DO: Instantiate StandardScaler and use fit_transform to scale the data

    scaler = StandardScaler()

    X_scaled =  scaler.fit_transform(X)

    return X_scaled

# Function to apply PCA to the dataset
def apply_pca(X_scaled, n_components):
    """
    Applies PCA to reduce dimensionality of the dataset.

    Parameters:
    X_scaled (array): Scaled dataset.
    n_components (int): Number of principal components.

    Returns:
    array: PCA transformed data.
    PCA object: The PCA object used.
    """
    # TO DO: Instantiate PCA with n_components and apply PCA transformation

    pca = PCA(n_components)

    principalComponents = pca.fit_transform(X_scaled)

    X_pca = principalComponents

    return X_pca, pca

# Function to plot the 2D PCA projection
def plot_pca_projection(X_pca, y):

    PC1 = X_pca[:, 0]
    PC2 = X_pca[:, 1]

    kmeans = KMeans(n_clusters=2)  
    labels = kmeans.fit_predict(X_pca) 

    #Define a colormap for the clusters
    my_cmap = ListedColormap(['cornflowerblue', 'orange'])  # One color for each cluster

    #Plot PC1 vs PC2 and color points by their cluster label
    fig, ax = plt.subplots()
    scatter = ax.scatter(PC1, PC2, c=labels, cmap=my_cmap)

    # Add a colorbar to show the cluster colors
    plt.colorbar(scatter, ax=ax, ticks=[0, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection with Colors for Clusters')

    plt.savefig('pc1_vs_pc2.png')

    plt.show()
  
    
if __name__ == "__main__":
    np.random.seed(2024)
    X, y = generate_blobs_dataset(n_samples=300, centers=2, n_features=3, random_state=2024)
    plot_3d_data(X, y)  # Visualize the original 3D dataset
    X_scaled = standardize_data(X)  # Standardize the dataset

    #TO DO: Fill in appropriate value for n_components
    X_pca, pca = apply_pca(X_scaled, n_components=2)  # Apply PCA

    plot_pca_projection(X_pca, y)  # Visualize the 2D PCA projection