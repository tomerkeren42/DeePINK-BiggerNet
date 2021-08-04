import warnings
import numpy as np
import scipy as sp
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import utilities


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


class NewDeepPinkModel(nn.Module):
    def __init__(self, p, hidden_sizes=[64], y_dist="gaussian", normalize_Z=True):
        """
        Adapted from https://arxiv.org/pdf/1809.01185.pdf.
        The module has two components:
        1. A sparse linear layer with dimension 2*p to p.
        However, there are only 2*p weights (each feature
        and knockoff points only to their own unique node).
        This is (maybe?) followed by a ReLU activation.
        2. A multilayer perceptron (MLP)
        Parameters
        ----------
        p : int
            The dimensionality of the data
        hidden_sizes: list
            A list of hidden sizes for the mlp layer(s).
            Defaults to [64].
        normalize_Z : bool
            If True, the first sparse linear layer is normalized
            so the weights for each feature/knockoff pair have an
            l1 norm of 1. This can modestly improve power in some
            settings.
        """

        super().__init__()

        # Initialize weight for first layer
        self.p = p
        self.y_dist = y_dist
        self.Z_weight = nn.Parameter(torch.ones(2 * p))
        self.norm_Z_weight = normalize_Z

        # Save indices/reverse indices to prevent violations of FDR control
        self.inds, self.rev_inds = utilities.random_permutation_inds(2 * p)
        self.feature_inds = self.rev_inds[0:self.p]
        self.ko_inds = self.rev_inds[self.p:]

        # Prepare for either MSE loss or cross entropy loss

        hidden_size = 32 * 16
        sign_size1 = 32
        sign_size2 = 32 // 2
        output_size = (32 // 4) * 32

        self.hidden_size = hidden_size
        self.cha_input = 16
        self.cha_hidden = 32
        self.K = 2
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = 0.2
        self.dropout_hidden = 0.2
        self.dropout_output = 0.2

        self.first_layer = nn.Linear(p, hidden_sizes[0])
        self.batch_norm1 = nn.BatchNorm1d(hidden_sizes[0])
        self.dropout1 = nn.Dropout(0.2)
        dense1 = nn.Linear(hidden_sizes[0], hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)
        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(self.cha_input)
        conv1 = conv1 = nn.Conv1d(
            self.cha_input,
            self.cha_input * self.K,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.cha_input,
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(self.cha_input * self.K)
        self.dropout_c2 = nn.Dropout(self.dropout_hidden)
        conv2 = nn.Conv1d(
            self.cha_input * self.K,
            self.cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(self.cha_hidden)
        self.dropout_c3 = nn.Dropout(self.dropout_hidden)
        conv3 = nn.Conv1d(
            self.cha_hidden,
            self.cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(self.cha_hidden)
        conv4 = nn.Conv1d(
            self.cha_hidden,
            self.cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=self.cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)
        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        self.flt = nn.Flatten()
        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(self.dropout_output)
        dense2 = nn.Linear(output_size, 1, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)
        self.loss = nn.BCEWithLogitsLoss()

        # k=p

        # Create MLP layers
        # mlp_layers = [nn.Linear(p, hidden_sizes[0])]
        # # for i in range(1):
        # #     mlp_layers.append(nn.ReLU())
        # #     mlp_layers.append(nn.Linear(k, k-100))
        # #     k=k-100
        # # Prepare for either MSE loss or cross entropy loss
        # mlp_layers.append(nn.ReLU())
        # if y_dist == "gaussian":
        #     mlp_layers.append(nn.Linear(k, 1))
        # else:
        #     mlp_layers.append(nn.Linear(hidden_sizes[-1], 2))
        #
        # # Then create MLP
        # self.mlp = nn.Sequential(*mlp_layers)
        # print(self.mlp)

        # from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        # self.mlp = TabNetClassifier()

    def _fetch_Z_weight(self):

        # Possibly don't normalize
        if not self.norm_Z_weight:
            return self.Z_weight

        # Else normalize, first construct denominator
        normalizer = torch.abs(self.Z_weight[self.feature_inds]) + torch.abs(
            self.Z_weight[self.ko_inds]
        )
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
        features = features[:, self.inds] # shuffle features to prevent FDR violations
        features = self._fetch_Z_weight().unsqueeze(dim=0) * features
        features = features[:, self.feature_inds] - features[:, self.ko_inds]

        x = nn.functional.relu(self.first_layer(features))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

    def predict(self, features):
        """
        Wraps forward method, for compatibility
        with sklearn classes.
        """
        with torch.no_grad():
            return self.forward(features).numpy()

    # def l1norm(self):
    #     out = 0
    #     for parameter in self.mlp.parameters():
    #         out += torch.abs(parameter).sum()
    #     out += torch.abs(self.Z_weight).sum()  # This is just for stability
    #     return out
    #
    # def l2norm(self):
    #     out = 0
    #     for parameter in self.mlp.parameters():
    #         out += (parameter ** 2).sum()
    #     out += (self.Z_weight ** 2).sum()
    #     return out

    def feature_importances(self, weight_scores=True):
        with torch.no_grad():
            # # Calculate weights from MLP
            # if weight_scores:
            #     layers = list(self.mlp.named_children())
            #     W = layers[0][1].weight.detach().numpy().T
            #     for layer in layers[1:]:
            #         if isinstance(layer[1], nn.ReLU):
            #             continue
            #         weight = layer[1].weight.detach().numpy().T
            #         W = np.dot(W, weight)
            #         W = W.squeeze()
            #         # print(W.shape)
            # else:
            W = np.ones(self.p)

            # Multiply by Z weights
            Z = self._fetch_Z_weight().numpy()
            feature_imp = Z[self.feature_inds] * W
            knockoff_imp = Z[self.ko_inds] * W
            return np.concatenate([feature_imp, knockoff_imp])


def train_newdeeppink(
    model,
    features,
    y,
    batchsize=100,
    num_epochs=50,
    lambda1=None,
    lambda2=None,
    verbose=True,
    **kwargs,
):

    # Infer n, p, set default lambda1, lambda2
    n = features.shape[0]
    p = int(features.shape[1] / 2)
    if lambda1 is None:
        lambda1 = 10 * np.sqrt(np.log(p) / n)
    if lambda2 is None:
        lambda2 = 0

    # Batchsize can't be bigger than n
    batchsize = min(features.shape[0], batchsize)

    # Create criterion
    features, y = map(lambda x: torch.tensor(x).detach().float(), (features, y))
    if model.y_dist == "gaussian":
        criterion = nn.MSELoss(reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")
        y = y.long()

    # Create optimizer
    opt = torch.optim.Adam(model.parameters(), **kwargs)

    # Loop through epochs
    for j in range(num_epochs):

        # Create batches, loop through
        batches = create_batches(features, y, batchsize=batchsize)
        predictive_loss = 0
        for Xbatch, ybatch in batches:
            # Forward pass and loss
            output = model(Xbatch)
            loss = criterion(output, ybatch.unsqueeze(-1))
            predictive_loss += loss

            # Add l1 and l2 regularization
            # loss += lambda1 * model.l1norm()
            # loss += lambda2 * model.l2norm()

            # Step
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose and j % 10 == 0:
            print(f"At epoch {j}, mean loss is {predictive_loss / n}")

    return model