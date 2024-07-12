# main.py (No changes required for import statements)
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distance = [] 
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]  
        centroids[k] = np.mean(points, axis=0)
    return centroids

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))
    for i in range(max_iters):
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
        previous_centroids = centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()  # Ensure the plot is displayed at the end
    return centroids, idx

original_img = plt.imread('bird_small.png')
plt.imshow(original_img)
plt.show()  # Ensure the original image is displayed
print("Shape of original_img is:", original_img.shape)

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
K = 16
max_iters = 10

initial_centroids = kMeans_init_centroids(X_img, K)
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

plot_kMeans_RGB(X_img, centroids, idx, K)
show_centroid_colors(centroids)

idx = find_closest_centroids(X_img, centroids)
X_recovered = centroids[idx, :]
X_recovered = np.reshape(X_recovered, original_img.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 16))
ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours' % K)
ax[1].axis('off')
plt.show()  # Ensure the comparison plot is displayed
