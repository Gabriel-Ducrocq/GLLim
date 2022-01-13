import numpy as np
import healpy as hp
import config
from time import time
from classy import Class
import matplotlib.pyplot as plt


class likelihood_approximation:
    def __init__(self, lmax=config.lmax, nside=config.nside, noise_covar_temp=config.noise_covar_temp,
                 noise_covar_pol=config.noise_covar_pol, mask_path=config.mask_path, fwhm_radians=config.beam_fwhm_radians,
                 cosmo_means=config.COSMO_PARAMS_MEAN_PRIOR, cosmo_std=config.COSMO_PARAMS_SIGMA_PRIOR,
                 cosmo_names=config.COSMO_PARAMS_NAMES, l_cut = 52):
        self.lmax = lmax
        self.fwhm = fwhm_radians
        self.noise_covar_temp = noise_covar_temp
        self.noise_covar_pol = noise_covar_pol
        self.noise_covar = np.array([noise_covar_temp, noise_covar_pol])
        self.mask_path = mask_path
        self.nside = nside
        self.fsky = 1
        self.cosmo_params_means = cosmo_means
        self.cosmo_params_std = cosmo_std
        self.cosmo_params_names = cosmo_names
        self.l_cut = l_cut
        self.bl_gauss = 1/hp.gauss_beam(fwhm=fwhm_radians, lmax=lmax)**2
        self.cls_tt_observed = None
        self.cls_ee_observed = None
        self.cls_te_observed = None

        self.cosmo = Class()
        if self.mask_path is not None:
            self.mask = hp.ud_grade(hp.read_map(self.mask_path, 0), self.nside)
            self.fsky = np.mean(self.mask)
            self.fsky=1

        self.rescale = np.array([np.round((2*l+1)*self.fsky) for l in range(2, self.lmax+1)])

    def generate_cls(self, theta):
        params = {'output': config.OUTPUT_CLASS,
                  "modes": "s,t",
                  "r": 0.001,
                  'l_max_scalars': config.lmax,
                  'lensing': config.LENSING}
        d = {name: val for name, val in zip(config.COSMO_PARAMS_NAMES, theta)}
        params.update(d)
        self.cosmo.set(params)
        self.cosmo.compute()
        cls = self.cosmo.lensed_cl(config.lmax)
        # 10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
        cls_tt = cls["tt"] * 2.7255e6 ** 2
        cls_ee = cls["ee"] * 2.7255e6 ** 2
        cls_bb = cls["bb"] * 2.7255e6 ** 2
        cls_te = cls["te"] * 2.7255e6 ** 2
        self.cosmo.struct_cleanup()
        self.cosmo.empty()
        return cls_tt, cls_ee, cls_te

    def generate_theta(self, N):
        return np.random.normal(loc = self.cosmo_params_means, scale= self.cosmo_params_std,
                                size = (N, len(self.cosmo_params_names)))

    def generate_data(self, N):
        all_cls_tt_hat = np.zeros((N, self.lmax-1))
        all_cls_ee_hat = np.zeros((N, self.lmax-1))
        all_cls_te_hat = np.zeros((N, self.lmax-1))

        all_theta = self.generate_theta(N)
        for i, theta in enumerate(all_theta):
            if i % 10 == 0:
                print(i)

            start = time()
            cls_tt, cls_ee, cls_te = self.generate_cls(theta)
            end = time()
            print("Cls gen time:", end - start)
            cls_tt_bar = cls_tt + self.bl_gauss*self.noise_covar_temp
            cls_ee_bar = cls_ee + self.bl_gauss*self.noise_covar_pol
            cls_te_bar = cls_te
            for l in range(2, self.l_cut+1):
                cov = np.array([[cls_tt[l], cls_te[l]], [cls_te[l], cls_ee[l]]])
                sqrt_cov = np.linalg.cholesky(cov)
                number_modes = np.round((2*l+1)*self.fsky)
                #number_modes = l

                slms = np.random.normal(size = (2, number_modes)) \
                + 1j * np.random.normal(size = (2, number_modes))
                slms = np.dot(sqrt_cov/np.sqrt(2), slms)

                slms_0 = np.dot( sqrt_cov, np.random.normal(size = 2))

                nlms = np.random.normal(scale = np.sqrt(1/2), size = (2, number_modes)) \
                + 1j * np.random.normal(scale = np.sqrt(1/2), size = (2, number_modes))
                nlms = np.dot(np.diag(np.sqrt(self.bl_gauss[l]*self.noise_covar)), nlms)

                nlms_0 = np.dot(np.diag(np.sqrt(self.bl_gauss[l]*self.noise_covar)), np.random.normal(size=2))

                #alms_0 = slms_0 + nlms_0
                alms = slms + nlms

                all_cls_tt_hat[i, l-2] = np.mean(np.abs(alms[0, :])**2)
                all_cls_ee_hat[i, l-2] = np.mean(np.abs(alms[1, :])**2)
                all_cls_te_hat[i, l-2] = np.sum((alms[0, :].real*alms[1, :].real + alms[0, :].imag*alms[1, :].imag))/number_modes
                #all_cls_tt_hat[i, l-2] = (alms_0[0]**2 + 2*np.sum(np.abs(alms[0, :])**2))/(2*l+1)
                #all_cls_ee_hat[i, l - 2] = (alms_0[1] ** 2 + 2 * np.sum(np.abs(alms[1, :]) ** 2)) / (2 * l + 1)
                #all_cls_te_hat[i, l-2] = (alms_0[0]*alms_0[1] + 2*np.sum(alms[0, :].real*alms[1, :].real + alms[0, :].imag*alms[1, :].imag))/(2*l+1)

            for l in range(self.l_cut+1, self.lmax+1):
                cov = (2/((2*l+1)*self.fsky))*np.array([[cls_tt_bar[l]**2, cls_tt_bar[l]*cls_te_bar[l], cls_te_bar[l]**2],
                                [cls_tt_bar[l]*cls_te_bar[l], (1/2)*(cls_tt_bar[l]*cls_ee_bar[l] + cls_te_bar[l]**2), cls_te_bar[l]*cls_ee_bar[l]],
                                [cls_te_bar[l]**2, cls_te_bar[l]*cls_ee_bar[l], cls_ee_bar[l]**2]])

                sqrt_cov = np.linalg.cholesky(cov)

                cl_tt_hat, cl_te_hat, cl_ee_hat = np.dot(sqrt_cov, np.random.normal(size = 3))
                all_cls_tt_hat[i, l-2] = cl_tt_hat + cls_tt_bar[l]
                all_cls_te_hat[i, l - 2] = cl_te_hat + cls_te_bar[l]
                all_cls_ee_hat[i, l - 2] = cl_ee_hat  + cls_ee_bar[l]

            end_again = time()
            #rescale = np.array([l*(l+1)/(2*np.pi) for l in range(2, self.lmax+1)])
            #plt.plot(all_cls_te_hat[i, :]*rescale)
            #plt.plot(cls_te[2:]*rescale, color="red")
            ##plt.yscale("log")
            ##plt.xscale("log")
            #plt.show()
            print("Total time:", end_again- start)
        return all_cls_tt_hat, all_cls_ee_hat, all_cls_te_hat, all_theta

    def set_observed_cls(self, cls_tt, cls_ee, cls_te):
        self.cls_tt_observed = cls_tt
        self.cls_ee_observed = cls_ee
        self.cls_te_observed = cls_te

    def compute_log_likelihood(self, theta):
        cls_tt, cls_ee, cls_te = self.generate_cls(theta)
        cls_tt_bar = cls_tt[2:] + self.bl_gauss[2:]*self.noise_covar_temp
        cls_ee_bar = cls_ee[2:] + self.bl_gauss[2:]*self.noise_covar_pol
        cls_te_bar = cls_te[2:]

        det_bar = cls_tt_bar*cls_ee_bar - cls_te_bar**2
        det_observed = self.cls_tt_observed*self.cls_ee_observed - self.cls_te_observed**2
        D = cls_tt_bar * self.cls_ee_observed + self.cls_tt_observed*cls_ee_bar - 2*self.cls_te_observed*cls_te_bar

        individual_lik = (D/det_bar + np.log(det_bar) -np.log(det_observed) -2)*self.rescale
        log_lik = -(1/2)*np.sum(individual_lik[:self.l_cut-1])

        for l in range(self.l_cut-1, self.lmax -1):
            cov = (2 / ((2 * (l+2) + 1) * self.fsky)) * np.array(
                [[cls_tt_bar[l] ** 2, cls_tt_bar[l] * cls_te_bar[l], cls_te_bar[l] ** 2],
                 [cls_tt_bar[l] * cls_te_bar[l], (1 / 2) * (cls_tt_bar[l] * cls_ee_bar[l] + cls_te_bar[l] ** 2),
                  cls_te_bar[l] * cls_ee_bar[l]],
                 [cls_te_bar[l] ** 2, cls_te_bar[l] * cls_ee_bar[l], cls_ee_bar[l] ** 2]])

            term_minus_mean = np.array([self.cls_tt_observed[l]- cls_tt_bar[l], self.cls_te_observed[l]- cls_te_bar[l],
                                        self.cls_ee_observed[l]- cls_ee_bar[l]])

            log_lik += -(1/2)*np.dot(term_minus_mean, np.linalg.solve(cov, term_minus_mean)) \
                       - (1/2)*np.log(np.linalg.det(cov))

        return log_lik