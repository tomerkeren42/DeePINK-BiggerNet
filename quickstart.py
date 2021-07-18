import numpy as np
import knockpy
import warnings
from knockpy.knockoff_filter import KnockoffFilter


np.random.seed(123)
n = 300 # number of data points
p = 500  # number of features
Sigma = knockpy.dgp.AR1(p=p, rho=0.5) # Stationary AR1 process with correlation 0.5

# Sample X
X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=(n,))

# Create random sparse coefficients
beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=0.1)
y = np.dot(X, beta) + np.random.randn(n)

kfilter_lasso = KnockoffFilter(
    ksampler='gaussian',
    fstat='lasso',
)

# Flags of whether each feature was rejected
rejections_lasso = kfilter_lasso.forward(
    X=X,
    y=y,
    Sigma=Sigma,
    fdr=0.1 # desired level of false discovery rate control
)
# Check the number of discoveries we made
power_lasso = np.dot(rejections_lasso, beta != 0) / (beta != 0).sum()
fdp_lasso = np.around(100*np.dot(rejections_lasso, beta == 0) / rejections_lasso.sum())
print(f"The lasso knockoff filter has discovered {100*power_lasso}% of the non-nulls with a FDP of {fdp_lasso}%")

kfilter_deeppink = KnockoffFilter(
    ksampler='gaussian',
    fstat='deeppink',
)

# Flags of whether each feature was rejected
rejections_deeppink = kfilter_deeppink.forward(
    X=X,
    y=y,
    Sigma=Sigma,
    fdr=0.1 # desired level of false discovery rate control
)
# Check the number of discoveries we made
power_deeppink = np.dot(rejections_deeppink, beta != 0) / (beta != 0).sum()
fdp_deeppink = np.around(100*np.dot(rejections_deeppink, beta == 0) / rejections_deeppink.sum())
print(f"The deeppink knockoff filter has discovered {100*power_deeppink}% of the non-nulls with a FDP of {fdp_deeppink}%")