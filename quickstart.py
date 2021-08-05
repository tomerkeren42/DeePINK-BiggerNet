import numpy as np
import knockpy
from knockpy.knockoff_filter import KnockoffFilter


# This setting replicates the DeepPINK paper (Lu et al. 2018)
np.random.seed(123)

all_p = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
# all_p = [50, 100]

all_models = ['deeppink', 'newdeeppink', 'lasso']
# all_models = ['newdeeppink']
number_of_data_points = 1000


def get_attributes(p):
    Sigma = np.linalg.inv(knockpy.dgp.AR1(p=p, rho=0.5))
    S = knockpy.smatrix.compute_smatrix(Sigma)
    beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=20 / p, coeff_size=1.5)
    return Sigma, S, beta


def calc_model(evaluate_model, num_of_data_points):
    for p in all_p:
        sigma, s, beta = get_attributes(p)
        n = num_of_data_points  # number of data points
        reps = 10
        powers = np.zeros(reps)
        fdps = np.zeros(reps)
        for j in range(reps):

            # Sample X and y
            X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=(n,))
            y = np.dot(X, beta) + np.random.randn(n)

            # X, y, _ = get_data()
            kfilter = KnockoffFilter(
                ksampler='gaussian',
                fstat=evaluate_model,
            )
            rej = kfilter.forward(X=X, y=y, Sigma=sigma, knockoff_kwargs={"S": s}, fdr=0.1)

            power = np.dot(rej, beta != 0) / max(1, (beta != 0).sum())
            fdp = np.dot(rej, beta == 0) / max(1, rej.sum())
            fdps[j] = fdp
            powers[j] = power
            # print(f"{model} at iteration j={j}, power={power}, fdp={fdp}, number of rej = {rej.sum()}")
        print(f"{evaluate_model} with n={num_of_data_points} and p={p} | power mean={powers.mean()} | std={powers.std() / np.sqrt(reps)} | FDP Mean={fdps.mean()}")


def start_evaluate():
    print(f"Starting evaluating the next models: {all_models}")
    for model in all_models:
        calc_model(evaluate_model=model, num_of_data_points=number_of_data_points)
