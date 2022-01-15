import numpy as np
import matplotlib.pyplot as plt


class forward_regressor():
    def __init__(self, K = 2):
        self.K = K
        self.all_c_star = []
        self.all_Gamma_star=[]
        self.all_pi_star=[]
        self.all_Sigma_star=[]
        self.all_A_star=[]
        self.all_b_star=[]


        self.all_means = []
        self.all_cov = []
        self.all_weights = []

        self.i = 0
        self.j = 1

    def compute_c_star(self, A, c, b):
        return np.dot(A, c) + b

    def compute_gamma_star(self, Sigma, A, Gamma):
        return Sigma + np.dot(np.dot(A, Gamma), A.T)

    def compute_sigma_star(self, Gamma, A, Sigma):
        return np.linalg.pinv(np.linalg.pinv(Gamma) + np.dot(np.dot(A.T, np.linalg.pinv(Sigma)), A))

    def compute_A_star(self, Sigma_star, A, Sigma):
        return np.dot(np.dot(Sigma_star, A.T), np.linalg.pinv(Sigma))

    def compute_b_star(self, Sigma_star, Sigma, A, b, c, Gamma):
        return np.dot(Sigma_star, np.dot(np.linalg.pinv(Gamma), c) - np.dot(np.dot(A.T, np.linalg.pinv(Sigma)), b))

    def compute_forward_parameters(self, json_data):
        self.all_c_star = []
        self.all_Gamma_star = []
        self.all_pi_star = []
        self.all_Sigma_star = []
        self.all_A_star = []
        self.all_b_star = []

        all_A = np.array(json_data["A"])
        all_c = np.array(json_data["c"])
        all_b = np.array(json_data["b"])
        all_Sigma = np.array(json_data["Sigma"])
        all_Gamma = np.array(json_data["Gamma"])

        for k in range(self.K):
            c_star = self.compute_c_star(all_A[k, :, :], all_c[:, k], all_b[:, k])
            Gamma_star = self.compute_gamma_star(all_Sigma[k, :, :], all_A[k, :, :], all_Gamma[k, :, :])
            pi_star = json_data["pi"][k]
            Sigma_star = self.compute_sigma_star(all_Gamma[k, :, :], all_A[k, :, :], all_Sigma[k, :, :])
            A_star = self.compute_A_star(Sigma_star, all_A[k, :, :], all_Sigma[k, :, :])
            b_star = self.compute_b_star(Sigma_star, all_Sigma[k, :, :], all_A[k, :, :], all_b[:, k],
                                         all_c[:, k], all_Gamma[k, :, :])

            self.all_c_star.append(c_star)
            self.all_Gamma_star.append(Gamma_star)
            self.all_pi_star.append(pi_star)
            self.all_Sigma_star.append(Sigma_star)
            self.all_A_star.append(A_star)
            self.all_b_star.append(b_star)


        params_star = {"c_star":self.all_c_star, "Gamma_star":self.all_Gamma_star, "pi_star":self.all_pi_star,
                       "Sigma_star":self.all_Sigma_star, "A_star":self.all_A_star, "b_star":self.all_b_star}

        np.save("data/forward_parameters.npy", params_star)

    def compute_posterior_distribution(self, json_data, posterior_weights, data):
        self.compute_forward_parameters(json_data)

        self.all_weights = posterior_weights
        for k in range(self.K):
            avg = np.dot(self.all_A_star[k], data) + self.all_b_star[k]
            self.all_means.append(avg)
            self.all_cov.append(self.all_Sigma_star[k])

    def compute_bivariate_log_posterior(self, x, y):
        x = np.array([x, y])
        posterior = 0
        for k in range(self.K):
            posterior += self.all_weights[k]*np.exp(-(1/2)*np.dot((x - self.all_means[k][[self.i, self.j]]),
                            np.linalg.solve(self.all_cov[k][[self.i,self.j]][:,[self.i, self.j]],
                            (x - self.all_means[k][[self.i, self.j]]))))/np.sqrt(np.linalg.det(
                self.all_cov[k][[self.i,self.j]][:,[self.i, self.j]])*(2*np.pi)**2)

        return np.log(posterior)


    def plot_bivariates(self, i, j, mcmc_chain):
        self.i = i
        self.j = j
        compute_log_posterior_vec = np.vectorize(self.compute_bivariate_log_posterior)
        mean = self.all_means[1]

        std = np.sqrt(np.diag(self.all_cov[0]))
        high = mean + 5*std
        low = mean - 5*std
        x = np.linspace(low[i], high[i], 100)
        y = np.linspace(low[j], high[j], 100)
        X, Y = np.meshgrid(x, y)
        Z = compute_log_posterior_vec(X, Y)
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        CS = ax.contour(X, Y, Z)
        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('Contour plot of the log posterior')
        plt.plot(mcmc_chain[:, i], mcmc_chain[:, j], "o", alpha= 0.3)
        plt.show()

        print(np.corrcoef(mcmc_chain[:, i], mcmc_chain[:, j]))
        print(self.all_cov[1][i, j]/np.sqrt(self.all_cov[1][i, i]*self.all_cov[1][j, j]))


