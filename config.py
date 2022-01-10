import numpy as np

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'


COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])

nside = 256
lmax = 2*nside
npix = 12*nside**2

mask_path = None
noise_covar_temp = (0.2/np.sqrt(2))**2
noise_covar_pol = 0.2**2


beam_fwhm = 0.5
beam_fwhm_radians = (np.pi / 180) * beam_fwhm