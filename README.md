# Neural Networks – XOR

This project implements a neural network **from scratch** in Python to solve the classic **XOR problem** — a fundamental test of nonlinear classification capability in neural networks.

---

## 🧠 What is the XOR Problem?

The XOR (exclusive OR) logic gate returns true if inputs differ (i.e., one is 1, the other is 0).

| Input A | Input B | XOR Output |
|---------|---------|------------|
|    0    |    0    |     0      |
|    0    |    1    |     1      |
|    1    |    0    |     1      |
|    1    |    1    |     0      |

A neural network must have **at least one hidden layer** with a **nonlinear activation function** to solve this.

---

## 🚀 Features

- Fully from scratch — no TensorFlow or PyTorch
- Custom implementation of:
  - Dense layers
  - Activation functions (ReLU, Sigmoid)
  - Loss function (Binary Cross-Entropy / MSE)
  - Backpropagation and weight updates
- Visual or print output to show learning

---

## 📂 Project Structure

```
NN-XOR/
│
├── activation.py # Activation function interface
├── activation_func.py # Sigmoid, ReLU, etc.
├── dense.py # Dense (fully connected) layer
├── layer.py # Layer base class
├── loss.py # Loss function
├── network.py # Neural network class
├── xor.py # Entry point to train XOR
└── README.md # You're reading it!
```

---

## 🛠️ How to Run

```bash
python xor.py
```
## 🧪 Output Example

```
Epoch 10000 - Loss: 0.0023
Predictions:
Input: [0, 0] → 0.01
Input: [0, 1] → 0.98
Input: [1, 0] → 0.97
Input: [1, 1] → 0.02

```
---

## 🧰 Requirements
```
Python 3.x

No external libraries (pure Python)
```

---

## ✏️ Author
```
Created by Prasanth
```

