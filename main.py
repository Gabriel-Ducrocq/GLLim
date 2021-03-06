import numpy as np
import json
from data_generator import generator
from approximate_likelihood import likelihood_approximation
from forward_regressor import forward_regressor
from metropolis_hastings import metropolisHastings
from likelihood_no_noise import likelihood
import config
from time import time
import healpy as hp

if __name__ == "__main__":
    np.random.seed()

    #gen = generator()
    #gen.generate_data(1000)

    #with open("data/parameters.json", "rb") as f:
    #    parameters = json.load(f)

    #regressor = forward_regressor(2)
    like_approx = likelihood()
    like_approx.generate_data(50000)

    #all_cls_tt_hat, all_cls_ee_hat, all_cls_te_hat, all_theta = like_approx.generate_data(10000)
    #np.save("data/cls_tt.npy", all_cls_tt_hat)
    #np.save("data/cls_ee.npy", all_cls_ee_hat)
    #np.save("data/cls_te.npy", all_cls_te_hat)
    #np.save("data/all_theta.npy", all_theta)

    #all_cls_tt_hat = np.load("data_true/cls_tt.npy")
    #all_cls_ee_hat = np.load("data_true/cls_ee.npy")
    #all_cls_te_hat = np.load("data_true/cls_te.npy")
    #all_theta = np.load("data_true/all_theta.npy")

    #like_approx.set_observed_cls(all_cls_tt_hat, all_cls_ee_hat, all_cls_te_hat)
    #mh = metropolisHastings(like_approx, cosmo_proposal_cov=config.proposal_covariance)
    #theta_init = np.random.normal(scale = config.COSMO_PARAMS_SIGMA_PRIOR) + config.COSMO_PARAMS_MEAN_PRIOR
    #N = 50000
    #start_time = time()
    #res = mh.run(theta_init, N)
    #end_time = time()
    #np.save("data/mh_official_run.npy", res)



    #posterior_weights = np.load("data/posterior_weights.npy")
    #dataset = np.load("data/covariates_test.npy")

    #regressor.compute_posterior_distribution(parameters, posterior_weights[1,:], dataset[:, 1])
    #regressor.plot_bivariates(2, 0)







