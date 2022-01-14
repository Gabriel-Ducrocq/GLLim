import numpy as np
import matplotlib.pyplot as plt
import config
from forward_regressor import forward_regressor
import json


chain = np.load("data/mh_official_run.npy")
true_theta = np.load("data_true/all_theta.npy")
regressor = forward_regressor(2)

posterior_weights = np.load("data/posterior_weights.npy")
cls_tt_true = np.load("data_true/cls_tt.npy")
cls_te_true = np.load("data_true/cls_te.npy")
cls_ee_true = np.load("data_true/cls_ee.npy")
dataset = np.concatenate([cls_tt_true, cls_ee_true, cls_te_true])/10
with open("data/parameters.json", "rb") as f:
    parameters = json.load(f)

print(posterior_weights)
regressor.compute_posterior_distribution(parameters, posterior_weights[0, :], dataset)


def compute_normal(x, avg, std):
    return np.exp(-(1/2)*(x-avg)**2/std**2)/np.sqrt(2*np.pi*std**2)


print(np.cov(chain.T))
print("\n")
print(np.corrcoef(chain.T))

print(chain.shape)
#plt.plot(chain[:, 0], chain[:, 1], "o", alpha = 0.5)
#plt.show()
regressor.plot_bivariates(4, 5, chain)

for i in range(6):
    low = np.mean(chain[:, i]) - 4*np.std(chain[:, i])
    high = np.mean(chain[:, i]) + 4 * np.std(chain[:, i])
    x = np.linspace(low, high, 1000)
    y = np.array([compute_normal(xx, config.COSMO_PARAMS_MEAN_PRIOR[i], config.COSMO_PARAMS_SIGMA_PRIOR[i]) for xx in x])
    plt.hist(chain[:, i], bins = 30, density=True, alpha=0.5)
    plt.plot(x, y)
    plt.title(config.COSMO_PARAMS_NAMES[i])
    plt.axvline(x = true_theta[i])
    plt.show()
