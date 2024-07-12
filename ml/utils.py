import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    X = np.load("data/ex7_X.npy")
    return X

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    cmap = ListedColormap(["red", "green", "blue"])
    colors = cmap(idx % 3)  # idx % 3 to ensure values are within colormap range
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=colors, linewidth=0.1, alpha=0.7)
    plt.show()  # Ensure the plot is displayed

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    plot_data_points(X, idx)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    plt.title("Iteration number %d" % i)
    plt.show()  # Ensure the plot is displayed

def plot_kMeans_RGB(X, centroids, idx, K):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(X[:, 0] * 255, X[:, 1] * 255, X[:, 2] * 255, zdir='z', depthshade=False, s=0.3, c=X)
    ax.scatter(centroids[:, 0] * 255, centroids[:, 1] * 255, centroids[:, 2] * 255, zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3)
    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')
    ax.get_yaxis().set_pane_color((0., 0., 0., .2))  # Corrected attribute reference
    ax.set_title("Original colors and their color clusters' centroids")
    plt.show()  # Ensure the plot is displayed

def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    num = np.arange(0, len(centroids))
    plt.figure(figsize=(16, 16))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)
    plt.show()  # Ensure the plot is displayed
