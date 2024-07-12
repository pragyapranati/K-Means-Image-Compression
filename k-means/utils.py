import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    X = np.load("data/ex7_X.npy")
    return X

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    unique_idx = np.unique(idx)
    colors = plt.cm.get_cmap('hsv', len(unique_idx))
    for i in unique_idx:
        color = colors(i / len(unique_idx))
        plt.scatter(X[idx == i, 0], X[idx == i, 1], s=30, c=[color], edgecolors='k', alpha=0.6)

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    plt.figure()
    plot_data_points(X, idx)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    plt.title(f"Iteration number {i}")
    plt.show()

def plot_kMeans_RGB(X, centroids, idx, K):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(221, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], zdir='z', depthshade=False, s=0.3, c=X / 255)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], zdir='z', depthshade=False, s=500, c='red', marker='x', lw=3)
    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')
    ax.yaxis.set_pane_color((0., 0., 0., .2))
    ax.set_title("Original colors and their color clusters' centroids")
    plt.show()

def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    num = np.arange(0, len(centroids))
    plt.figure(figsize=(16, 2))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette.reshape(1, -1, 3) / 255, aspect='auto')
    plt.show()
