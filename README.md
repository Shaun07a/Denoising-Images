Image Denoising using Encoder–Decoder CNN

Deep Learning Programming Assignment – CSA3007

Overview

This project implements an Encoder–Decoder Convolutional Neural Network (CNN) using TensorFlow/Keras to perform image denoising. The model learns to reconstruct clean images from noisy grayscale inputs by compressing the image into a latent representation and then decoding it back into a denoised output.

Objective

Train a neural network to remove noise from grayscale images.

Learn how Encoder–Decoder architectures work in deep learning.

Visualize the difference between noisy, original, and reconstructed images.

Model Architecture

The network follows a simple Autoencoder-style architecture:

Encoder

Convolution Layer

Max Pooling Layer

Feature Compression

Decoder

Convolution Layer

Upsampling Layer

Image Reconstruction

Loss Function: Mean Squared Error (MSE)

Dataset

The dataset contains paired images:

clean_images/ – original grayscale images

noisy_images/ – corresponding noisy images

These pairs are used to train the network to learn the mapping from noisy to clean images.

Project Structure
Denoising-Images
│
├── clean_images
├── noisy_images
├── image_denoising.py
└── README.md
Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

Output Visualization

After training, the model displays:

Noisy Image

Original Clean Image

Reconstructed (Denoised) Image

This allows direct comparison of the model's performance.

How to Run

Install dependencies:

pip install tensorflow matplotlib scikit-learn pillow

Run the program:

python image_denoising.py
Author

Shaun Joseph
Registration Number: 23BAI10555
