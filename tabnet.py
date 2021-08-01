from sklearn.model_selection import KFold
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np
import pandas as pd
from torch import nn
import torch
import knockpy.utilities


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

        # Apply MLP
        return self.tabnet_model.predict(features)

    def predict(self, features):
        """
        Wraps forward method, for compatibility
        with sklearn classes.
        """
        # with torch.no_grad():
        # TODO: changed this
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

    def feature_importances(self):
        # TODO: get W from tabnet model
        W = np.ones(self.p)
        # Multiply by Z weights
        Z = self._fetch_Z_weight().numpy()
        print(f"only Z: {Z}")
        feature_imp = Z[self.feature_inds] * W
        knockoff_imp = Z[self.ko_inds] * W
        print(f"feature_imp {feature_imp}")
        return np.concatenate([feature_imp, knockoff_imp])

    @staticmethod
    def get_tabnet_model(max_epoches=100, patience=150, verbose=0):
        X, y, X_test = get_data()

        tabnet_regression = None
        kf = KFold(n_splits=2, random_state=42, shuffle=True)
        # predictions_array = []
        CV_score_array = []
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            tabnet_regression = TabNetRegressor(verbose=verbose, seed=42)
            tabnet_regression.fit(X_train=X_train, y_train=y_train,
                                  eval_set=[(X_valid, y_valid)],
                                  patience=patience, max_epochs=max_epoches,
                                  eval_metric=['rmse'])
            CV_score_array.append(tabnet_regression.best_cost)
            # predictions_array.append(np.expm1(regressor.predict(X_test)))

        # predictions = np.mean(predictions_array, axis=0)

        print("The CV score is %.5f" % np.mean(CV_score_array, axis=0))
        return tabnet_regression

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
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF',
                '1stFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces',
                'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', 'PoolArea', 'YrSold']

    X = train_data[features]
    y = np.log1p(train_data["SalePrice"])
    X_test = test_data[features]

    X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
    X_test = X_test.apply(lambda x: x.fillna(x.mean()), axis=0)

    X = X.to_numpy()
    y = y.to_numpy().reshape(-1, 1)
    X_test = X_test.to_numpy()
    return X, y, X_test
