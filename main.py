import numpy as np
import json
from data_generator import generator
from forward_regressor import forward_regressor


if __name__ == "__main__":
    np.random.seed()

    gen = generator()
    #gen.generate_data(10)

    with open("data/parameters.json", "rb") as f:
        parameters = json.load(f)

    regressor = forward_regressor(2)

    posterior_weights = np.load("data/posterior_weights.npy")
    dataset = np.load("data/covariates_test.npy")

    regressor.compute_posterior_distribution(parameters, posterior_weights[1,:], dataset[:, 1])
    regressor.plot_bivariates(2, 0)



