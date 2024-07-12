import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
from PIL import Image
from sklearn.cluster import KMeans

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]
        if points.size != 0:
            centroids[k] = np.mean(points, axis=0)
    return centroids

def kMeans_init_centroids(X, K):
    kmeans = KMeans(n_clusters=K, init='k-means++', random_state=0)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m, dtype=int)
    for i in range(max_iters):
        print(f"K-Means iteration {i}/{max_iters-1}")
        idx = find_closest_centroids(X, centroids)
        previous_centroids = centroids
        centroids = compute_centroids(X, idx, K)
    return centroids, idx

def count_unique_colors(img):
    return len(np.unique(np.reshape(img, (-1, img.shape[2])), axis=0))

image_file = "bird_small.png"  # enter the file name here
if not os.path.exists(image_file):
    print("The file does not exist. Please check the file name and try again.")
    exit()

original_img = np.array(Image.open(image_file).convert('RGB'))
plt.imshow(original_img)
plt.title('Original Image')
plt.show()
print("Shape of original_img is:", original_img.shape)

num_unique_colors = count_unique_colors(original_img)
print(f"The image contains {num_unique_colors} unique colors.")

while True:
    K = int(input(f"Enter the number of colors to compress the image to (must be less than {num_unique_colors}): "))
    if K < num_unique_colors:
        break
    print(f"Please enter a number less than {num_unique_colors}.")

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3)).astype(float)
max_iters = 10

print("Initial centroids:")
initial_centroids = kMeans_init_centroids(X_img, K)
print(initial_centroids)

centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
print("Final centroids:")
print(centroids)
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

ax[1].imshow(X_recovered.astype(np.uint8))
ax[1].set_title(f'Compressed with {K} colours')
ax[1].axis('off')
plt.show()
