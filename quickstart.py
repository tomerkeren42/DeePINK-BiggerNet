import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import knockpy
from knockpy.knockoff_filter import KnockoffFilter

# This setting replicates the DeepPINK paper (Lu et al. 2018)
np.random.seed(123)

# for p in [1500,2000,2500,3000]:
#     n = 1000  # number of data points
#     # p = 400  # number of features, you can vary this as they do in the deeppink paper
#     # Taking the inverse to replicate the deeppink paper
#     Sigma = np.linalg.inv(knockpy.dgp.AR1(p=p, rho=0.5))
#     # Compute S-matrix in advance to save computation
#     S = knockpy.smatrix.compute_smatrix(Sigma)
#     beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=30 / p, coeff_size=1.5)
#     reps = 10
#     powers_newdeeppink = np.zeros(reps)
#     fdps_newdeeppink = np.zeros(reps)
#     for j in range(reps):
#         # Sample X and y
#         X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=(n,))
#         y = np.dot(X, beta) + np.random.randn(n)
#         # Fit deeppink
#         kfilter = KnockoffFilter(
#             ksampler='gaussian',
#             fstat='newdeeppink',
#         )
#         rej = kfilter.forward(X=X, y=y, Sigma=Sigma, knockoff_kwargs={"S":S}, fdr=0.1)
#         power = np.dot(rej, beta != 0) / max(1, (beta != 0).sum())
#         fdp = np.dot(rej, beta == 0) / max(1, rej.sum())
#         fdps_newdeeppink[j] = fdp
#         powers_newdeeppink[j] = power
#         # print(f"NewDeepPink at iteration j={j}, power={power}, fdp={fdp}, number of rej = {rej.sum()}")
#     print(f"NewDeepPINK with n=1000 and p={p} power mean={powers_newdeeppink.mean()}, std={powers_newdeeppink.std() / np.sqrt(reps)}, FDP Mean={fdps_newdeeppink.mean()}")


for p in [200,400,600,800,1000,1500,2000,2500,3000]:
    n = 1000  # number of data points
    # p = 400  # number of features, you can vary this as they do in the deeppink paper
    # Taking the inverse to replicate the deeppink paper
    Sigma = np.linalg.inv(knockpy.dgp.AR1(p=p, rho=0.5))
    # Compute S-matrix in advance to save computation
    S = knockpy.smatrix.compute_smatrix(Sigma)
    beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=30 / p, coeff_size=1.5)
    reps = 10
    powers_deeppink = np.zeros(reps)
    fdps_deeppink = np.zeros(reps)
    for j in range(reps):
        # Sample X and y
        X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=(n,))
        y = np.dot(X, beta) + np.random.randn(n)
        # Fit deeppink
        kfilter = KnockoffFilter(
            ksampler='gaussian',
            fstat='deeppink',
        )
        rej = kfilter.forward(X=X, y=y, Sigma=Sigma, knockoff_kwargs={"S":S}, fdr=0.1)
        power = np.dot(rej, beta != 0) / max(1, (beta != 0).sum())
        fdp = np.dot(rej, beta == 0) / max(1, rej.sum())
        fdps_deeppink[j] = fdp
        powers_deeppink[j] = power
        # print(f"DeepPink at iteration j={j}, power={power}, fdp={fdp}, number of rej = {rej.sum()}")

    print(f"DeepPINK with n=1000 and p={p} power mean={powers_deeppink.mean()}, std={powers_deeppink.std() / np.sqrt(reps)}, FDP Mean={fdps_deeppink.mean()}")

# for p in [50,100,200,400,600,800,1000,1500,2000,2500,3000]:
for p in [2000, 2500, 3000]:
    n = 1000  # number of data points
    # p = 400  # number of features, you can vary this as they do in the deeppink paper
    # Taking the inverse to replicate the deeppink paper
    Sigma = np.linalg.inv(knockpy.dgp.AR1(p=p, rho=0.5))
    # Compute S-matrix in advance to save computation
    S = knockpy.smatrix.compute_smatrix(Sigma)
    beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=30 / p, coeff_size=1.5)
    reps = 10
    powers_lasso = np.zeros(reps)
    fdps_lasso = np.zeros(reps)
    for j in range(reps):
        # Sample X and y
        X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=(n,))
        y = np.dot(X, beta) + np.random.randn(n)
        # Fit deeppink
        kfilter = KnockoffFilter(
            ksampler='gaussian',
            fstat='lasso',
        )
        rej = kfilter.forward(X=X, y=y, Sigma=Sigma, knockoff_kwargs={"S":S}, fdr=0.1)
        power = np.dot(rej, beta != 0) / max(1, (beta != 0).sum())
        fdp = np.dot(rej, beta == 0) / max(1, rej.sum())
        fdps_lasso[j] = fdp
        powers_lasso[j] = power
        # print(f"Lasso at iteration j={j}, power={power}, fdp={fdp}, number of rej = {rej.sum()}")

    print(f"Lasso with n=1000 and p={p} power mean={powers_lasso.mean()}, std={powers_lasso.std() / np.sqrt(reps)}, FDP Mean={fdps_lasso.mean()}")
