import numpy as np
from dense import Dense
from activation_func import Tanh, Sigmoid
from network import train, predict
from loss import mse, mse_derivative

import matplotlib.pyplot as plt

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [Dense(2, 3), Tanh(), Dense(3, 1), Tanh()]

# network = [Dense(2, 3), Sigmoid(), Dense(3, 1), Sigmoid()]

epochs = 10000
learning_rate = 0.1

# training

train(network, mse, mse_derivative, X, Y, epochs, learning_rate, False)

# testing

# After training
test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected_outputs = [0, 1, 1, 0]

print("\nTesting XOR Network:\n")

for inp, expected in zip(test_inputs, expected_outputs):
    output = predict(network, inp)
    predicted = 1 if output[0][0] > 0.5 else 0
    print(
        f"Input: {inp} => Predicted: {predicted} | Output Value: {output[0][0]:.4f} | Expected: {expected}"
    )


# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0, 0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()
