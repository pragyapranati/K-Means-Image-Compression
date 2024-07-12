# 🎨 K-Means Image Compression

## 📄 Description
**K-Means Image Compression** is a Python-based project that compresses an image by reducing the number of colors used. This technique is implemented using the K-Means clustering algorithm, making it ideal for those looking to understand and apply machine learning concepts in image processing.

## ✨ Features
- 🖼️ Compress images by reducing the number of unique colors.
- 📊 Visualize the original and compressed images.
- 🛠️ Flexible input for the desired number of colors in the compressed image.
- ⚡ Efficient K-Means clustering implementation with customizable iterations.

## 🚀 How to Run

### Prerequisites
- 🐍 Python 3.x
- 📦 Required libraries: `numpy`, `matplotlib`, `PIL`, `scikit-learn`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/pragyapranatie/k-means-image-compression.git
   cd k-means-image-compression
   ```

2. Install the required libraries:

   ```bash
   pip install numpy matplotlib pillow scikit-learn
   ```

### Usage

1. Place your image file in the repository directory. Update the `image_file` variable in the script with your image filename.

2. Run the script:

   ```bash
   python kmeans_image_compression.py
   ```

3. Follow the prompts to enter the desired number of colors for image compression.

## 🛠️ Tech Stack
- Python
- Numpy
- Matplotlib
- Pillow (PIL)
- Scikit-learn

## 🧠 Logic Explanation

### 📚 K-Means Clustering

K-Means clustering is a popular unsupervised machine learning algorithm used for clustering data points. The algorithm aims to partition the data into K clusters, where each data point belongs to the cluster with the nearest mean.

### 🔄 Steps Involved

1. **Initialization**: Select K initial centroids randomly or using the K-Means++ algorithm.
2. **Assignment**: Assign each data point to the closest centroid.
3. **Update**: Calculate the new centroids as the mean of all points assigned to each cluster.
4. **Repeat**: Repeat the assignment and update steps until the centroids do not change significantly or a maximum number of iterations is reached.

In this project, K-Means clustering is applied to the RGB values of the image pixels to reduce the number of unique colors, effectively compressing the image.

## 🌟 Visual Examples

### Original Image
![image](https://github.com/user-attachments/assets/78692670-f79a-498b-821c-06127e9aead3)

![image](https://github.com/user-attachments/assets/64538ff4-02c4-46cf-ae61-f680db12da72)


### Compressed Image with 20 Colors
![image](https://github.com/user-attachments/assets/4ff820bf-2568-46a3-a535-4c963d5a2ec9)
![image](https://github.com/user-attachments/assets/9b7f1f79-4d29-4952-8dc2-24c5af887faa)


## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any improvements or bug fixes.

## 📜 License
This project is licensed under the Apache License.

