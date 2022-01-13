import numpy as np
import matplotlib.pyplot as plt
import config


chain = np.load("data/mh_preliminary_run.npy")
true_theta = np.load("data_true/all_theta.npy")

def compute_normal(x, avg, std):
    return np.exp(-(1/2)*(x-avg)**2/std**2)/np.sqrt(2*np.pi*std**2)


print(np.cov(chain.T))
print("\n")
print(np.corrcoef(chain.T))

for i in range(6):
    low = np.mean(chain[:, i]) - 4*np.std(chain[:, i])
    high = np.mean(chain[:, i]) + 4 * np.std(chain[:, i])
    x = np.linspace(low, high, 1000)
    y = np.array([compute_normal(xx, config.COSMO_PARAMS_MEAN_PRIOR[i], config.COSMO_PARAMS_SIGMA_PRIOR[i]) for xx in x])
    plt.hist(chain[:, i], bins = 15, density=True, alpha=0.5)
    plt.plot(x, y)
    plt.title(config.COSMO_PARAMS_NAMES[i])
    plt.axvline(x = true_theta[i])
    plt.show()