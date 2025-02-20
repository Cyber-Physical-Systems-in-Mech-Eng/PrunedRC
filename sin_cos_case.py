import numpy as np
from matplotlib import pyplot as plt
from benchmark_systems import load_data

# Generate a simple case for the sin-cos system: x: sin(t), y: cos(t)


x_train, y_train, x_test, y_test = load_data(name="sincos", n_samples=10)

sample_idx = 0
plt.figure()
plt.subplot(2, 1, 1)
for i in range(x_train.shape[2]):
    plt.plot(x_train[sample_idx, :, i], label="x_train")
plt.legend()
plt.title(r"$\sin(t) \mapsto \cos(t)$")

plt.subplot(2, 1, 2)
for i in range(y_train.shape[2]):
    plt.plot(y_train[sample_idx, :, i], label="y_train")
plt.legend()
plt.xlabel(r"$t$")
plt.show()
