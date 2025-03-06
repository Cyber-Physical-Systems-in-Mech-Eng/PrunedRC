import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmark_systems import load_data


NUM_SAMPLES = 10
cases = ["narma5", "narma10", "narma15", "lorenz", "sincos", "sincos3", "vdp"]

for case in cases:
    print(f"Generating data for {case}...")
    x_train, y_train, x_test, y_test = load_data(name=case, n_samples=NUM_SAMPLES)
    data_train = (x_train, y_train)
    data_test = (x_test, y_test)

    # store training and testing data in pkl file
    path = os.path.join(os.getcwd(), "data", f"{case}_data.pkl")
    with open(path, "wb") as f:
        pickle.dump((x_train, y_train, x_test, y_test), f)
