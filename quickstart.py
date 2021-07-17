import knockpy as kpy
from knockpy.knockoff_filter import KnockoffFilter

# Generate synthetic data from a Gaussian linear model
data_gen_process = kpy.dgp.DGP()
data_gen_process.sample_data(
        n=1500, # Number of datapoints
        p=500, # Dimensionality
        sparsity=0.1,
        x_dist='gaussian',
)
X = data_gen_process.X
y = data_gen_process.y
Sigma=data_gen_process.Sigma

# Run model-X knockoffs
kfilter = KnockoffFilter(
        fstat='lasso',
        ksampler='gaussian',
)
rejections = kfilter.forward(X=X, y=y, Sigma=Sigma)