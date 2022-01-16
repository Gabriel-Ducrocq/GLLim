import numpy as np
from classy import Class
import config
import healpy as hp
from scipy import stats



class likelihood:
    def __init__(self, lmax=config.lmax, nside=config.nside, fwhm_radians=config.beam_fwhm_radians,
                 cosmo_means=config.COSMO_PARAMS_MEAN_PRIOR, cosmo_std=config.COSMO_PARAMS_SIGMA_PRIOR,
                 cosmo_names=config.COSMO_PARAMS_NAMES):
        self.lmax = lmax
        self.fwhm = fwhm_radians
        self.nside = nside
        self.fsky = 1
        self.cosmo_params_means = cosmo_means
        self.cosmo_params_std = cosmo_std
        self.cosmo_params_names = cosmo_names
        self.bl_gauss = 1/hp.gauss_beam(fwhm=fwhm_radians, lmax=lmax)**2
        self.cls_tt_observed = None
        self.cls_ee_observed = None
        self.cls_te_observed = None

        self.cosmo = Class()
        self.rescale = np.array([int(np.round((2*l+1)*self.fsky)) for l in range(2, self.lmax+1)])

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

    def save_data(self, cls_tt_hat, cls_ee_hat, cls_te_hat, all_theta):
        np.save("data/cls_tt.npy", cls_tt_hat)
        np.save("data/cls_ee.npy", cls_ee_hat)
        np.save("data/cls_te.npy", cls_te_hat)
        np.save("data/all_theta.npy", all_theta)

    def generate_data(self, N):
        all_cls_tt_hat = np.zeros((N, self.lmax+1))
        all_cls_ee_hat = np.zeros((N, self.lmax+1))
        all_cls_te_hat = np.zeros((N, self.lmax+1))

        all_theta = self.generate_theta(N)
        for i, theta in enumerate(all_theta):
            if i % 10 == 0:
                print(i)

            cls_tt, cls_ee, cls_te = self.generate_cls(theta)

            alms_T, alms_E, _ = hp.synalm([cls_tt, cls_ee, np.zeros(len(cls_te)), cls_te, np.zeros(len(cls_te)),
                                           np.zeros(len(cls_te))], lmax=self.lmax,new = True)
            cls_tt_hat, cls_ee_hat, cls_te_hat = hp.alm2cl([alms_T, alms_E], lmax=self.lmax)

            all_cls_tt_hat[i, :] = cls_tt_hat
            all_cls_ee_hat[i, :] = cls_ee_hat
            all_cls_te_hat[i, :] = cls_te_hat

        self.save_data(all_cls_tt_hat, all_cls_ee_hat, all_cls_te_hat, all_theta)

    def set_observed_cls(self, cls_tt, cls_ee, cls_te):
        self.cls_tt_observed = cls_tt
        self.cls_ee_observed = cls_ee
        self.cls_te_observed = cls_te

    def compute_log_likelihood(self, theta):
        cls_tt, cls_ee, cls_te = self.generate_cls(theta)
        C_mat = np.zeros((self.lmax+1, 2, 2))
        C_mat[:, 0, 0] = cls_tt[:]
        C_mat[:, 1, 1] = cls_ee[:]
        C_mat[:, 1, 0] = cls_te[:]
        C_mat[:, 0, 1] = cls_te[:]

        C_observed_mat = np.zeros((self.lmax+1, 2, 2))
        C_observed_mat[:, 0, 0] = self.cls_tt_observed[:]
        C_observed_mat[:, 1, 1] = self.cls_ee_observed[:]
        C_observed_mat[:, 1, 0] = C_observed_mat[:, 0, 1] = self.cls_te_observed[:]
        log_lik = 0
        for l in range(2, self.lmax+1):
            nu = 2*l+1
            p = 2
            log_lik += ((nu - p - 1)/2)*np.log(np.linalg.det(C_observed_mat[l, :, :])) - (nu/2)* np.log(np.linalg.det(C_mat[l, :, :]/nu))\
             - np.sum(np.diag(np.linalg.solve(C_mat[l, :,:]/nu, C_observed_mat[l, :, :])/2))

        return log_lik

