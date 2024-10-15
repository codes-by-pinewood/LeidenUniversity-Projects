# No external libraries are allowed to be imported in this file
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

# Function to load the dataset
def load_data():
    """
    Loads the Swiss Roll dataset and corresponding color labels from files.

    Returns:
    tuple: The data (X) and color labels (color)
    """
    # TO DO: Load dataset from files

    X = np.load('swiss_roll.npy')
    color = np.load('color.npy')



    return X, color

# Function to apply t-SNE to the dataset
def apply_tsne(X, n_components, perplexity, max_iter, init, random_state=2024):
    """
    Applies t-SNE to the Swiss Roll dataset after scaling it.

    Parameters:
    X (array): The input dataset.
    perplexity (float): t-SNE perplexity parameter.
    random_state (int): Random seed for reproducibility.

    Returns:
    array: The t-SNE transformed dataset with 2 components.
    """
    # TO DO: Create a pipeline to apply StandardScaler and t-SNE


    pipeline = make_pipeline(StandardScaler(), TSNE(n_components=n_components, perplexity=perplexity, n_iter = max_iter, init=init, random_state=random_state))

    X_tsne_2d = pipeline.fit_transform(X)

    return X_tsne_2d

# Function to plot the 2D t-SNE projection
def plot_tsne_projection(X_tsne_2d, color):
    """
    Plots the 2D projection of the t-SNE transformed Swiss Roll dataset.

    Parameters:
    X_tsne_2d (array): The t-SNE transformed dataset.
    color (array): The color labels for the points.
    """
    # TO DO: Use scatter plot to visualize the 2D projection from t-SNE

    PC1 = X_tsne_2d[:,0]
    PC2 = X_tsne_2d[:,1]

    plt.figure(figsize=(10, 6))
    plt.scatter(PC1, PC2, c=color, alpha=0.8, marker='^')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PC1 versus PC2 with Cluster Colors')
    plt.grid(True)
    plt.show()


# Function to return a recogzinable letter from the plot
def return_identified_letter():
    """
    Returns the letter identified from the t-SNE plot.
    """
    # TO DO: If you succeed in unfolding the dataset with t-SNE, you will see a recognizable letter (between A-Z) in the plot.
    # Identify and return the letter. Example: return 'A'.
    return 'O'.upper()


if __name__ == "__main__":
    X, color = load_data()

    # TO DO: Fill in the appropriate values for n_components, perplexity, max_iter, and init
    X_tsne_2d = apply_tsne(X, n_components=2, perplexity=55, max_iter=350, init='pca', random_state=2024)
    # 5 to 50 perplexity
    # 300 to 500 max_iter
    # 'random' to 'pca'

    plot_tsne_projection(X_tsne_2d, color)
    print(return_identified_letter())