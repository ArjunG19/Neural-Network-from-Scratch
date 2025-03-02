# 🧠 Neural Networks from Scratch
A Fully Custom Deep Learning Framework Implemented in Python

## 🚀 Overview

This project implements a fully functional neural network from scratch using NumPy without relying on deep learning libraries like TensorFlow or PyTorch. The framework supports:  

✅ Fully Connected Networks (Dense Layers)

✅ Convolutional Neural Networks (CNNs)

✅ Recurrent Neural Networks (RNNs)

✅ Activation Functions (ReLU, Softmax, Sigmoid, etc.)

✅ Dropout Regularization

✅ L1 & L2 Weight Regularization

✅ Optimizers (SGD, Adam, etc.)

✅ Custom Loss Functions

✅ Forward & Backward Propagation

✅ Model Saving & Loading

This is a great project for understanding how deep learning models work at a fundamental level! 

## 🔧 Installation
__1️⃣ Clone the repository__

```
git clone https://github.com/ArjunG19/Neural-Network-from-Scratch
```

__2️⃣ Install dependencies__

```
pip install -r requirements.txt
```

## 🏗️ Features

__📌 1. Fully Connected Layers__

The network supports multiple dense layers with flexible neuron counts. Example:

```python
from NeuralNetwork import Layer_Dense

layer1 = Layer_Dense(n_inputs=784, n_neurons=128)
layer2 = Layer_Dense(n_inputs=128, n_neurons=10)
```
__📌 2. Activation Functions__

Implemented ReLU, Softmax, and Sigmoid activations. Example:

```python
from NeuralNetwork import Activation_ReLU, Activation_Softmax

activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
```
__📌 3. Dropout Regularization__
Prevent overfitting with dropout layers:

```python
from NeuralNetwork import Layer_Dropout

dropout = Layer_Dropout(rate=0.2)  # Drops 20% of neurons during training
```
__📌 4. Optimizers (SGD, Adam, etc.)__
Train with different optimizers:

```python
from NeuralNetwork import Optimizer_Adam

optimizer = Optimizer_Adam(learning_rate=0.001,decay=1e-3)
```
__📌 5. Training & Evaluation__
```python
model.train(X_train, y_train, epochs=10, batch_size=32)
model.evaluate(X_test, y_test)
```

## 🏆 Results
🎯 Achieved high accuracy on datasets like MNIST, CIFAR-10, and more!

📉 Loss reduced significantly with proper regularization and dropout.

📈 CNN outperformed simple dense networks for image classification.

## 🧠 Future Enhancements

Implementing LSTMs & GRUs for advanced sequential learning.

Expanding support for custom datasets.

Adding hyperparameter tuning utilities.

