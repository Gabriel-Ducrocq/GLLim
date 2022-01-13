import numpy as np
import config


class metropolisHastings:
    def __init__(self, likelihood, lmax=config.lmax, cosmo_names=config.COSMO_PARAMS_NAMES, cosmo_prior_mean = config.COSMO_PARAMS_MEAN_PRIOR,
                 cosmo_prior_std = config.COSMO_PARAMS_SIGMA_PRIOR, cosmo_proposal_std = config.COSMO_PARAMS_SIGMA_PRIOR):
        self.lmax = lmax
        self.cosmo_prior_std = cosmo_prior_std
        self.cosmo_prior_mean = cosmo_prior_mean
        self.cosmo_proposal_std = cosmo_proposal_std
        self.cosmo_names = cosmo_names
        self.likelihood = likelihood
        self.old_log_lik = None

    def compute_log_prior(self, theta):
        return -(1/2)*np.sum((theta - self.cosmo_prior_mean)**2/self.cosmo_prior_std**2)


    def compute_log_ratio(self, theta_proposed, theta_current):
        log_lik_proposed = self.likelihood.compute_log_likelihood(theta_proposed)
        if self.old_log_lik is None:
            self.old_log_lik = self.likelihood.compute_log_likelihood(theta_current)

        proposed_log_prior = self.compute_log_prior(theta_proposed)
        current_log_prior = self.compute_log_prior(theta_current)

        log_ratio = log_lik_proposed + proposed_log_prior - self.old_log_lik - current_log_prior
        return log_ratio, log_lik_proposed

    def propose_theta(self, theta_current):
        theta_proposed = np.random.normal(scale = self.cosmo_proposal_std) + theta_current
        return theta_proposed

    def run(self, theta, N):
        all_theta = np.zeros((N+1, len(self.cosmo_names)))
        all_theta[0, :] = theta[:]
        accept = 0
        for i in range(1, N+1):
            if i % 10 == 0:
                print(i)

            proposed_theta = self.propose_theta(theta)
            log_r, log_lik_proposed = self.compute_log_ratio(proposed_theta, theta)
            if np.log(np.random.uniform()) < log_r:
                theta = proposed_theta[:]
                accept += 1
                self.old_log_lik = log_lik_proposed

            all_theta[i, :] = theta[:]

        print("Acceptance ratio:", accept/N)
        return all_theta
