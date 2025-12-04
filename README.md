# Mid-Level Neural Network for Digits Classification
A mid-level neural network implementation from scratch on the MNIST Zip-Digits dataset. Includes forward and backward propagation, adaptive activation function selection, training/validation split and testing, and various gradient descent techniques (VLR-GD, SGD, VLR-GD-ES, SGD-ES, VLR-GD-WD, SGD-WD).

The primary purpose of this project was to develop a strong foundation in the theory and implementation of neural networks and to compare the performance of different gradient descent algorithms on digit classification. Writing the codebase in C++ got me implement all of the math and algorithms manually from scratch. I utilized a modular design with a single layer class, a neural network driver class, and flexible user choice for between-layer activation. The results were plotted using Matplotlib and the Python frontend was called using std::system. All matrix operations were handled using the Eigen library in C++, which provided a comprehensive set of hyper-optimized functionalities for linear algebra purposes.

## Dataset
The US Post Office Zip Code data is used for all digits classification, 0-9. The original dataset consists of 7291 training images and 2007 test images, each 16x16 grayscale, and was developed by Yann LeCun at AT&T Research Labs. This project deals with classifying all images with label = 1 vs. all images with label != 1. The dataset has been pre-processed, with each entry consisting of two extracted features - intensity and symmetry. Following pre-processing, I utilized 300 images for training and 8998 for testing.

We define symmetry and intensity features below.

**Symmetry:**

$$
\text{Symmetry} = -\frac{1}{d} \sum_{i = 1}^d \big(|x_i - F(x)_i| + |x_i - G(x)_i|\big)
$$

where \(d = 256\) pixels, \(F(x)\) is the vertical flip of pixel \(x\), and \(G(x)\) is the horizontal flip of pixel \(x\).

**Intensity:**

$$
\text{Intensity} = \frac{1}{d} \sum_{i = 1}^d \frac{B(x_i)}{d}
$$

where \(d = 256\) and 

$$
B(x) = 
\begin{cases}
1, & \text{if $x$ is black} \\
-1, & \text{if $x$ is not black}
\end{cases}
$$
