import numpy as np

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'


COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN_PRIOR = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA_PRIOR = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
proposal_covariance = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])**2
print(proposal_covariance)
nside = 256
lmax = 2*nside
npix = 12*nside**2

mask_path = "data/HFI_Mask_GalPlane-apo0_2048_R2_80%_bis.00.fits"
noise_covar_temp = (0.2/np.sqrt(2))**2
noise_covar_pol = 0.2**2


beam_fwhm = 0.5
beam_fwhm_radians = (np.pi / 180) * beam_fwhm


preliminary_run = False

if not preliminary_run:
    preliminary_chain = np.load("data/mh_preliminary_run.npy")
    proposal_covariance = np.cov(preliminary_chain.T)
    np.fill_diagonal(proposal_covariance, np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])**2)
    print("\n")
    print(np.diag(proposal_covariance))
