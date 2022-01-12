import numpy as np
import json
from data_generator import generator
from approximate_likelihood import likelihood_approximation
from forward_regressor import forward_regressor

import config
import healpy as hp

if __name__ == "__main__":
    np.random.seed()

    gen = generator()
    #gen.generate_data(10000)

    #with open("data/parameters.json", "rb") as f:
    #    parameters = json.load(f)

    #regressor = forward_regressor(2)
    like_approx = likelihood_approximation()
    like_approx.generate_data(10)

    #posterior_weights = np.load("data/posterior_weights.npy")
    #dataset = np.load("data/covariates_test.npy")

    #regressor.compute_posterior_distribution(parameters, posterior_weights[1,:], dataset[:, 1])
    #regressor.plot_bivariates(2, 0)







