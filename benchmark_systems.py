import numpy as np
from scipy.integrate import odeint


"""
Provides a set of benchmark systems for testing the performance of the classical and pruned reservoir computing models
"""


def load_data(name: str, n_samples: int = 10) -> tuple:
    """
    Loads pre-defined data sets.

    Args:
        name (str): The name of the data set to load.
        num_samples (int): The number of samples to generate.

    Returns:
        tuple: A tuple containing the training and testing data sets.
    """

    if name == "lorenz":
        x, y = generate_lorenz_data(n_samples=n_samples)
    elif name == "vdp":
        x, y = generate_vdp_data(n_samples=n_samples)
    elif name == "narma5":
        x, y = generate_narma_data(n_samples=n_samples, order=5)
    elif name == "narma10":
        x, y = generate_narma_data(n_samples=n_samples, order=10)
    elif name == "narma15":
        x, y = generate_narma_data(n_samples=n_samples, order=15)
    elif name == "sincos":
        x, y = generate_sincos_data(n_samples=n_samples, order=1)
    elif name == "sincos3":
        x, y = generate_sincos_data(n_samples=n_samples, order=3)

    else:
        raise ValueError("Invalid data set name")

    # Split the data into training and testing sets
    x_train, y_train, x_test, y_test = split_data(x, y, train_size=0.8)

    return (x_train, y_train, x_test, y_test)


def split_data(
    x: np.ndarray, y: np.ndarray, train_size: float, shuffle: bool = True
) -> tuple:
    # Splits the data into training and testing sets.

    # Maintains the data shape (3-dimensional) as sklearn's train_test_split can only handle
    # 2-dimensional data.

    # generate splitting indices
    n_samples = x.shape[0]
    idx = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(idx)

    idx_train = idx[: int(train_size * n_samples)]
    idx_test = idx[int(train_size * n_samples) :]

    # split data
    x_train, x_test = x[idx_train], x[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    return (x_train, y_train, x_test, y_test)


def generate_lorenz_data(n_samples: int) -> tuple:
    """
    Generates the Lorenz attractor time series data.
    Args:
        n_samples (int): Number of samples representing different initial conditions for which a fixed number of time steps are simulated.
    Returns:
        tuple: A tuple containing two 3-dimensional numpy arrays:
            - x: Inputs representing the system state at time t with shape [n_samples, n_time_steps, n_features].
            - y: Outputs representing the system state at time t+1 with shape [n_samples, n_time_steps, n_features].
    Notes:
        - The Lorenz system parameters are sigma=10, beta=8/3, and rho=28.
        - The time vector is generated with a time step of 0.02 and spans 3001 steps.
        - The shapes of the return arrays are [n_samples, n_time_steps, n_features], where n_features=3 and n_time_steps=3,000.
    """

    # Splits the data according to the following logic:
    # Inputs x: system state at time t: [x(t), y(t), z(t)]
    # Outputs y: system state at time t+1: [x(t+1), y(t+1), z(t+1)]

    # Number of samples represents different initial conditions for which a fixed number of time steps are simulated

    # Lorenz system parameters
    sigma, beta, rho = 10, 8 / 3, 28

    # time vector
    dt = 0.02
    t = np.arange(0, 3001 * dt, dt)

    x, y = [], []

    for _ in range(n_samples):
        # sample initial condition from random normal distribution
        x0 = np.random.rand(3)

        # solve the system
        sol = odeint(lorenz, x0, t, args=(sigma, beta, rho))

        # extract inputs (state at time t) and outputs (state at time t+1)
        x.append(np.expand_dims(sol[:-1], axis=0))
        y.append(np.expand_dims(sol[1:], axis=0))

    # stack the data into a 3-dimensional array
    x = np.vstack(x)
    y = np.vstack(y)

    return (x, y)


def lorenz(x: np.ndarray, t: np.ndarray, sigma: float, beta: float, rho: float):
    # ODE definition of the Lorenz attractor

    x_dot = sigma * (x[1] - x[0])
    y_dot = x[0] * (rho - x[2]) - x[1]
    z_dot = x[0] * x[1] - beta * x[2]

    return np.array([x_dot, y_dot, z_dot])


def generate_vdp_data(n_samples: int) -> tuple:
    # Generates the van der Pol system time series.

    # Splits the data according to the following logic:
    # Inputs x: system state at time t: [x(t), y(t)]
    # Outputs y: system state at time t+1: [x(t+1), y(t+1)]

    # Number of samples represents different initial conditions for which a fixed number of time steps are simulated

    # Shapes of return arrays is 3-dimensional: [n_samples, n_time_steps, n_features], where n_features = 2 and n_time_steps = 10,000

    # van der Pol system parameters
    mu = 1.0

    # time vector
    dt = 0.1
    t = np.arange(0, 2000 * dt + dt, dt)

    x, y = [], []

    for _ in range(n_samples):
        # sample initial condition from random normal distribution
        x0 = np.random.rand(2)

        # solve the system
        sol = odeint(van_der_pol, x0, t, args=(mu,))

        # extract inputs (state at time t) and outputs (state at time t+1)
        x.append(np.expand_dims(sol[:-1], axis=0))
        y.append(np.expand_dims(sol[1:], axis=0))

    # stack the data into a 3-dimensional array
    x = np.vstack(x)
    y = np.vstack(y)

    return (x, y)


def van_der_pol(x: np.ndarray, t: np.ndarray, mu: float):
    # ODE definition of the Van der Pol oscillator

    x_dot = x[1]
    y_dot = mu * (1 - x[0] ** 2) * x[1] - x[0]

    return np.array([x_dot, y_dot])


def generate_narma_data(n_samples: int, order: int) -> tuple:

    length = 2501

    x, y = [], []

    for _ in range(n_samples):

        # input sequence
        u = np.random.uniform(0, 0.5, size=length)  # Input sequence

        # generate NARMA sequence
        sol = narma(u, order)

        # extract inputs (state at time t) and outputs (state at time t+1)
        x.append(np.expand_dims(sol[:-1], axis=(0, -1)))
        y.append(np.expand_dims(sol[1:], axis=(0, -1)))

    # stack the data into a 3-dimensional array
    x = np.vstack(x)
    y = np.vstack(y)

    return (x, y)


def narma(u: np.ndarray, order: int):
    # u is the input sequence, typically random numbers [0, 0.5]
    # order is the order of the NAR

    # initialize the output array
    y = np.zeros(u.shape[0])

    # initial condition
    y[0] = u[0]

    for i in range(u.shape[0] - 1):
        sumy = 0
        for j in range(1, order + 1):
            if i > j:
                sumy += y[i + 1 - j]

        if order > 10:
            y[i + 1] = np.tanh(
                0.3 * y[i] + 0.05 * y[i] * sumy + 1.5 * u[i] * u[i + 1 - order] + 0.1
            )
        else:
            y[i + 1] = (
                0.3 * y[i] + 0.05 * y[i] * sumy + 1.5 * u[i] * u[i + 1 - order] + 0.1
            )
    return y


def generate_sincos_data(n_samples: int, order: int) -> tuple:
    length = 3000
    dt = 0.1
    t = np.arange(0, length * dt, dt)

    if order == 1:
        omega = 1.0
        a_sin = 1.0
        p_sin = 1.0
        b_cos = 1.0
        p_cos = 1.0
    elif order == 3:
        omega = 1.0
        a_sin = 1.0
        p_sin = 1.0
        b_cos = 0.25
        p_cos = 3.0
    else:
        raise ValueError("Invalid order for sincos data generation")

    x, y = [], []

    for _ in range(n_samples):

        # draw random phase
        phi = np.random.uniform(0, 2 * np.pi)

        # inputs: A*sin(omega*t + phi)**p_sin,
        # outputs: B*cos(omega*t + phi)**p_cos
        x.append(
            np.expand_dims(a_sin * (np.sin(omega * t + phi) ** p_sin), axis=(0, -1))
        )
        y.append(
            np.expand_dims(b_cos * (np.cos(omega * t + phi) ** p_cos), axis=(0, -1))
        )

    # stack the data into a 3-dimensional array
    x = np.vstack(x)
    y = np.vstack(y)

    return (x, y)


if __name__ == "__main__":

    datasets = ["sincos", "sincos3", "lorenz", "vdp", "narma5", "narma10", "narma15"]
    n_samples = 20
    for dataset in datasets:
        x_train, y_train, x_test, y_test = load_data(name=dataset, n_samples=n_samples)
        print(f"shape of {dataset} system input data: {x_train.shape}")
        print(f"shape of {dataset} system output data: {y_train.shape}\n")
