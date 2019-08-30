import numpy as np

# a relative-like error
def error_estimation(stf_mat, stf_mat_max):
    """error defined as sum(k_ij_n-k_ij_nmax)/max(k_ij_nmax)"""
    nominator = np.sum(np.abs(stf_mat - stf_mat_max),axis=(0,1))
    denominator = np.max(np.abs(stf_mat_max))
    assert (denominator != 0)
    return nominator / denominator

# Root mean square error
def RMSE(stf_mat, stf_mat_max):
    """error defined as RMSE"""
    size = stf_mat.shape
    err = np.power(np.sum(np.power(stf_mat - stf_mat_max, 2.0))/(size[0]*size[1]), 0.5)
    return err
