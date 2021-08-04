from sklearn.model_selection import KFold
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import pandas as pd
from torch import nn
import torch
import knockpy.utilities

def create_batches(features, y, batchsize):

    # Create random indices to reorder datapoints
    n = features.shape[0]
    p = features.shape[1]
    inds = torch.randperm(n)

    # Iterate through and create batches
    i = 0
    batches = []
    while i < n:
        batches.append([features[inds][i : i + batchsize], y[inds][i : i + batchsize]])
        i += batchsize
    return batches

class TabNetDeepPinkModel(nn.Module):
    def __init__(self, p, hidden_sizes=[64], y_dist="gaussian", normalize_Z=True):
        super().__init__()
        # Initialize weight for first layer
        self.p = p
        self.y_dist = y_dist
        self.Z_weight = nn.Parameter(torch.ones(2 * p))
        self.norm_Z_weight = normalize_Z

        # Save indices/reverse indices to prevent violations of FDR control
        self.inds, self.rev_inds = knockpy.utilities.random_permutation_inds(2 * p)
        self.feature_inds = self.rev_inds[0:self.p]
        self.ko_inds = self.rev_inds[self.p:]

        self.mlp = [nn.Linear(p, hidden_sizes[0])]
        self.relu = (nn.ReLU())
        self.tabnet_model = self.get_tabnet_model()

    def _fetch_Z_weight(self):
        # Possibly don't normalize
        if not self.norm_Z_weight:
            return self.Z_weight
        # Else normalize, first construct denominator
        normalizer = torch.abs(self.Z_weight[self.feature_inds]) + torch.abs(self.Z_weight[self.ko_inds])
        # Normalize
        Z = torch.abs(self.Z_weight[self.feature_inds]) / normalizer
        Ztilde = torch.abs(self.Z_weight[self.ko_inds]) / normalizer
        # Concatenate and reshuffle
        return torch.cat([Z, Ztilde], dim=0)[self.inds]

    def forward(self, features):
        """
        Note: features are now shuffled
        """

        # First layer: pairwise weights (and sum)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features).float()

        features = features[:, self.inds]  # shuffle features to prevent FDR violations
        features = self._fetch_Z_weight().unsqueeze(dim=0) * features
        features = features[:, self.feature_inds] - features[:, self.ko_inds]

        # Apply tabnet
        features = self.mlp(features)
        features = self.relu(features)
        return self.tabnet_model.predict(features)

    def predict(self, features):
        """
        Wraps forward method, for compatibility
        with sklearn classes.
        """
        with torch.no_grad():
            return self.forward(features) # .numpy()

    def l1norm(self):
        out = 0
        for parameter in self.mlp.parameters():
            out += torch.abs(parameter).sum()
        out += torch.abs(self.Z_weight).sum()  # This is just for stability
        return out

    def l2norm(self):
        out = 0
        for parameter in self.mlp.parameters():
            out += (parameter ** 2).sum()
        out += (self.Z_weight ** 2).sum()
        return out

    def feature_importances(self, weight_scores=False):
        # TODO: get W from tabnet model
        with torch.no_grad():
            if weight_scores:
                layers = list(self.mlp.named_children())
                W = layers[0][1].weight.detach().numpy().T
                for layer in layers[1:]:
                    if isinstance(layer[1], nn.ReLU):
                        continue
                    weight = layer[1].weight.detach().numpy().T
                    W = np.dot(W, weight)
                    W = W.squeeze(-1)
            else:
                W = np.ones(self.p)
            # Multiply by Z weights
            Z = self._fetch_Z_weight().detach().numpy()
            feature_imp = Z[self.feature_inds] * W
            knockoff_imp = Z[self.ko_inds] * W
            # print(f"feature_imp {feature_imp}")
            return np.concatenate([feature_imp, knockoff_imp])

    @staticmethod
    def get_tabnet_model(max_epoches=100, patience=150, verbose=0):
        # X, y, X_test = get_data()

        tabnet_regression = None
        # kf = KFold(n_splits=2, random_state=42, shuffle=True)
        # predictions_array = []
        # CV_score_array = []
        # for train_index, test_index in kf.split(X):
        #     X_train, X_valid = X[train_index], X[test_index]
        #     y_train, y_valid = y[train_index], y[test_index]
        #     tabnet_regression = TabNetRegressor(verbose=verbose, seed=42)
        #     tabnet_regression.fit(X_train=X_train, y_train=y_train,
        #                           eval_set=[(X_valid, y_valid)],
        #                           patience=patience, max_epochs=max_epoches,
        #                           eval_metric=['rmse'])
        #     CV_score_array.append(tabnet_regression.best_cost)
            # predictions_array.append(np.expm1(regressor.predict(X_test)))

        # predictions = np.mean(predictions_array, axis=0)

        # print("The CV score is %.5f" % np.mean(CV_score_array, axis=0))

        tabnet_regression = TabNetRegressor(verbose=verbose, seed=42)
        return tabnet_regression

def train_deeppink(
    model,
    features,
    y,
    batchsize=100,
    num_epochs=5,
    lambda1=None,
    lambda2=None,
    verbose=True,
    **kwargs,
):

    # Infer n, p, set default lambda1, lambda2
    n = features.shape[0]
    p = int(features.shape[1] / 2)
    # if lambda1 is None:
    #     lambda1 = 10 * np.sqrt(np.log(p) / n)
    # if lambda2 is None:
    #     lambda2 = 0
    # batchsize = min(features.shape[0], batchsize)
    batchsize = features.shape[0]
    features = torch.tensor(features).detach().float()
    features = features[:, model.inds]  # shuffle features to prevent FDR violations
    features = model._fetch_Z_weight().unsqueeze(dim=0) * features
    features = features[:, model.feature_inds] - features[:, model.ko_inds]
    # features = features.detach().numpy()

    # opt = torch.optim.Adam(model.parameters(), **kwargs)
    # opt = torch.optim.Adam
    #
    # tabnet_regression = TabNetRegressor(verbose=verbose, seed=42, optimizer_fn=opt)
    for j in range(num_epochs):
        # Create batches, loop through
        batches = create_batches(features, y, batchsize=batchsize)
        predictive_loss = 0
        for Xbatch, ybatch in batches:
            Xbatch = Xbatch.detach().numpy()
            model.tabnet.fit(X_train=Xbatch, y_train=ybatch.reshape(-1, 1),
                                      patience=10, max_epochs=20)
    # features = torch.tensor(features).detach().float()
    # features = features[:, model.inds]  # shuffle features to prevent FDR violations
    # features = model._fetch_Z_weight().unsqueeze(dim=0) * features
    # features = features[:, model.feature_inds] - features[:, model.ko_inds]
    features = features.detach().numpy()



    # kf = KFold(n_splits=2, random_state=42, shuffle=True)
    # # features, y = map(lambda x: torch.tensor(x).detach().float(), (features, y))
    # for train_index, test_index in kf.split(features):
    #     X_train, X_valid = features[train_index], features[test_index]
    #     y_train, y_valid = y.reshape(-1,1)[train_index], y.reshape(-1,1)[test_index]
    #     # tabnet_regression = TabNetRegressor(verbose=verbose, seed=42)
    #     model.tabnet_model.fit(X_train=X_train, y_train=y_train,
    #                           patience=150, max_epochs=10,
    #                           eval_metric=['rmse'])

    # tabnet_regression = TabNetRegressor(verbose=verbose, seed=42)
    return  model.tabnet

def get_feature_importance(p, weight_scores=False):
    # Calculate weights from MLP
    # if weight_scores:
        # layers = list(self.mlp.named_children())
        # W = layers[0][1].weight.detach().numpy().T
        # for layer in layers[1:]:
        #     if isinstance(layer[1], nn.ReLU):
        #         continue
        #     weight = layer[1].weight.detach().numpy().T
        #     W = np.dot(W, weight)
        #     W = W.squeeze(-1)
    # else:
    W = np.ones(p)

    # Multiply by Z weights
    Z = fetch_Z_weight().numpy()
    feature_imp = Z[self.feature_inds] * W
    knockoff_imp = Z[self.ko_inds] * W
    return np.concatenate([feature_imp, knockoff_imp])

def get_data():
    # train_data = pd.read_csv('train.csv')
    # test_data = pd.read_csv('test.csv')
    # features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
    #             'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',
    #             '1stFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
    #             'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces',
    #             'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    #             'EnclosedPorch', 'PoolArea', 'YrSold']
    #
    # X = train_data[features]
    # y = np.log1p(train_data["SalePrice"])
    # X_test = test_data[features]
    #
    # X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
    # X_test = X_test.apply(lambda x: x.fillna(x.mean()), axis=0)
    #
    # X = X.to_numpy()
    # y = y.to_numpy().reshape(-1, 1)
    # X_test = X_test.to_numpy()
    #

    p=50
    n=1000
    sigma = np.linalg.inv(knockpy.dgp.AR1(p=p, rho=0.5))
    S = knockpy.smatrix.compute_smatrix(sigma)
    beta = knockpy.dgp.create_sparse_coefficients(p=p, sparsity=20 / p, coeff_size=1.5)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=(n,))
    y = np.dot(X, beta) + np.random.randn(n)
    y = y.reshape(-1, 1)
    X_test = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=(n,))

    return X, y, X_test
