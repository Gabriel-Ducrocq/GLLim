import numpy as np
import healpy as hp
import config
from classy import Class
from time import time




class generator():
    def __init__(self, nside=config.nside, lmax=config.lmax, npix=config.npix, fwhm=config.beam_fwhm_radians,
                 noise_var_temp=config.noise_covar_temp, noise_var_pol=config.noise_covar_temp,
                 mask_path=config.mask_path, cosmo_params_name=config.COSMO_PARAMS_NAMES,
                 cosmo_params_means=config.COSMO_PARAMS_MEAN_PRIOR, cosmo_params_std=config.COSMO_PARAMS_SIGMA_PRIOR):
        self.nside = nside
        self.lmax = lmax
        self.npix = npix
        self.fwhm = fwhm
        self.noise_var_temp = noise_var_temp
        self.noise_var_pol = noise_var_pol
        self.cosmo_params_names = cosmo_params_name
        self.cosmo_params_means = cosmo_params_means
        self.cosmo_params_std = cosmo_params_std
        self.cosmo = Class()
        self.mask_path = mask_path
        if self.mask_path is not None:
            self.mask = hp.ud_grade(hp.read_map(self.mask_path, 0), self.nside)

    def generate_theta(self, N):
        return np.random.normal(loc = self.cosmo_params_means, scale= self.cosmo_params_std,
                                size = (N, len(self.cosmo_params_names)))

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
        return cls_tt, cls_ee, cls_bb, cls_te

    def generate_noisy_map(self, cls_tt, cls_ee, cls_bb, cls_te, cls_tb, cls_eb):
        I, Q, U =hp.synfast([cls_tt, cls_ee, cls_bb, cls_te, cls_eb, cls_tb], new=True, fwhm = self.fwhm,
                            lmax=self.lmax, nside=self.nside)
        I += np.random.normal(scale=np.sqrt(self.noise_var_temp), size = len(I))
        Q += np.random.normal(scale=np.sqrt(self.noise_var_pol), size = len(Q))
        U += np.random.normal(scale=np.sqrt(self.noise_var_pol), size=len(U))

        if self.mask_path is not None:
            return I*self.mask, Q*self.mask, U*self.mask
        else:
            return I, Q, U

    def save_data(self, cls_tt_hat, cls_ee_hat, cls_bb_hat, cls_te_hat, all_theta):
        np.save("data/cls_tt_pseudo.npy", cls_tt_hat)
        np.save("data/cls_ee_pseudo.npy", cls_ee_hat)
        np.save("data/cls_bb_pseudo.npy", cls_bb_hat)
        np.save("data/cls_te_pseudo.npy", cls_te_hat)
        np.save("data/theta_pseudo.npy", all_theta)

    def generate_data(self, N):
        rescale = np.array([l*(l+1)/(2*np.pi) for l in range(2, self.lmax+1)])
        all_cls_tt_hat = np.zeros((N, self.lmax-1))
        all_cls_ee_hat = np.zeros((N, self.lmax-1))
        all_cls_bb_hat = np.zeros((N, self.lmax-1))
        all_cls_te_hat = np.zeros((N, self.lmax-1))

        all_theta = self.generate_theta(N)
        cls_tb = np.zeros(self.lmax+1)
        cls_eb = np.zeros(self.lmax+1)
        start = time()
        for i, theta in enumerate(all_theta):
            if i % 10 == 0:
                print(i)


            cls_tt, cls_ee, cls_bb, cls_te =self.generate_cls(theta)
            I, Q, U = self.generate_noisy_map(cls_tt, cls_ee, cls_bb, cls_te, cls_tb, cls_eb)

            cls_tt_hat, cls_ee_hat, cls_bb_hat, cls_te_hat, _, _ = hp.anafast([I, Q, U], lmax=self.lmax)

            all_cls_tt_hat[i, :] = cls_tt_hat[2:]*rescale
            all_cls_ee_hat[i, :] = cls_ee_hat[2:]*rescale
            all_cls_bb_hat[i, :] = cls_bb_hat[2:]*rescale
            all_cls_te_hat[i, :] = cls_te_hat[2:]*rescale


        end = time()
        print("Total time:", end-start)
        self.save_data(all_cls_tt_hat, all_cls_ee_hat, all_cls_bb_hat, all_cls_te_hat, all_theta)






