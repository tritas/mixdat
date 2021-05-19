# coding=utf-8
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from ..features.standardizer import MixedDataStandardizer

logger = logging.getLogger(__name__)
seed = 42
th.manual_seed(seed)
has_cuda = th.cuda.is_available()
device = th.device("cuda" if has_cuda else "cpu")
if has_cuda:
    logging.info("[CUDA] Running on {} devices".format(th.cuda.device_count()))
    th.cuda.manual_seed(seed)


def flatten(x):
    return x.view((x.size(0), -1))


def LOG_EMBEDDING_FUNC(d):
    return int(np.ceil(np.log2(d)))


def _fc(input_dim, output_dim):
    return nn.Linear(input_dim, output_dim)


def _layers(dims):
    return list(map(lambda i: _new_layer(dims[i - 1], dims[i]), range(1, len(dims))))


def fc_layers(layer_dims, *args, **kwargs):
    """Make a sequence of fully-connected layers"""
    layers = [
        _new_layer(layer_dims[i - 1], layer_dims[i], *args, **kwargs)
        for i in range(1, len(layer_dims))
    ]
    return layers


def _new_layer(input_dim, output_dim, activation, dropout=0, noise_scale=0, bn=False):
    ops = []
    if noise_scale:
        ops.append(GaussianNoiseSampler(noise_scale))
    if dropout:
        ops.append(nn.Dropout(dropout, inplace=True))
    ops.append(nn.Linear(input_dim, output_dim))
    if bn:
        ops.append(nn.BatchNorm1d(output_dim))
    ops.append(activation)
    return nn.Sequential(*ops)


class GaussianNoiseSampler(nn.Module):
    def __init__(self, scale=0.01, inplace=False):
        super(GaussianNoiseSampler, self).__init__()
        if scale < 0:
            raise ValueError(
                "noise scale has to be greather than 0, " "but got {}".format(scale)
            )
        self.scale = scale
        self.inplace = inplace

    def forward(self, inputs):
        if self.scale:
            inputs.add_(th.randn_like(inputs) * self.scale)
        return inputs

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "scale={}{}".format(self.scale, inplace_str)


def mixed_to_tensor(df, inplace=False):
    if not inplace:
        df = df.copy()
    categ_cols = df.dtypes[df.dtypes == "category"].index
    df.loc[:, categ_cols] = df.loc[:, categ_cols].apply(lambda s: s.values.codes)
    return th.from_numpy(df.values.astype(float))


class MixedAutoencoder(nn.Module, BaseEstimator, TransformerMixin):
    """Modeling mixed data with piece-wise losses in the output.
    - boolean
    - continuous
    - categorical
    - high-cardinality categorical"""

    def __init__(
        self,
        features_dtypes,
        vocab_sizes,
        hidden_dims,
        embedding_thr,
        embedding_func=LOG_EMBEDDING_FUNC,
        input_dropout=0.2,
        latent_dropout=0.5,
        noise_scale=0.1,
        batch_norm=True,
        tie_weights=False,
        activation=nn.LeakyReLU(negative_slope=0.1, inplace=True),
    ):
        """

        Parameters
        ----------
        features_dtypes: pd.Series of shape [n_features,]
            Should have features names as index and pandas/numpy dtypes as values.
        vocab_sizes: pd.Series of shape [n_categoricals,]
            The cardinality for each categorical variable in the dataset.
            Should be `max(id) + 1` for identifier `id`.
        hidden_dim: list of int
          The layers' number of hidden units.
        """
        super(MixedAutoencoder, self).__init__()
        # Neural network hyper-parameters
        self.latent_dims = hidden_dims
        self.activation = activation
        self.dropout_input = input_dropout
        self.dropout_latent = latent_dropout
        self.noise_scale = noise_scale
        # Masks and data structs to handle mixed data types
        self.vocab_size = vocab_sizes
        self.bool_mask = features_dtypes == bool
        self.continuous_mask = features_dtypes == float
        self.n_features = self.bool_mask.sum() + self.continuous_mask.sum()
        self.n_features = int(self.n_features)
        # Parameters and loss criteria definition
        self.embeds = {}
        self.output_embeds = {}
        self.categorical_col_ind = OrderedDict()
        self.loss_fn = dict(boolean=nn.BCEWithLogitsLoss(), continuous=nn.MSELoss())
        # Add input/output embeddings as modules
        for i, (feature_name, n_classes) in enumerate(vocab_sizes.items()):
            col_indx = features_dtypes.index.searchsorted(feature_name)
            self.categorical_col_ind[col_indx] = feature_name
            if n_classes > embedding_thr:
                embedding_dim = embedding_func(n_classes)
                emb = nn.Embedding(n_classes, embedding_dim, padding_idx=-1)
                self.embeds[col_indx] = emb
                self.add_module("input_{}".format(feature_name), emb)
                self.n_features += embedding_dim
            else:
                self.n_features += n_classes
            emb_out = nn.Linear(hidden_dims[0], n_classes)
            self.add_module("output_{}".format(feature_name), emb_out)
            self.output_embeds[col_indx] = emb_out
            if tie_weights:
                # This won't work usually as
                # embedding_dim << hidden_dims[0]
                self.output_embeds[col_indx].weight = self.embeds[col_indx].weight
            self.loss_fn[col_indx] = nn.CrossEntropyLoss()

        # Build recognition network
        recognition_layers = [
            _new_layer(
                self.n_features, hidden_dims[0], input_dropout, noise_scale, batch_norm
            )
        ]
        recognition_layers.extend(
            fc_layers(hidden_dims[1:], latent_dropout, noise_scale, batch_norm)
        )
        self.recognition_network = nn.Sequential(*recognition_layers)

        # Build generation network
        generation_layers = fc_layers(
            list(reversed(hidden_dims)), latent_dropout, noise_scale, batch_norm
        )
        self.generation_network = nn.Sequential(*generation_layers)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    @classmethod
    def from_df(cls, df, *args, **kwargs):
        mds = MixedDataStandardizer(df)
        return cls(mds.types, mds.sizes, *args, **kwargs)

    def forward(self, input_tensor):
        # Project vocabularies and concatenate with the rest of the inputs
        # to form the initial feature vector.
        embeds = []
        for col_indx in self.categorical_col_ind.keys():
            classes_tensor = input_tensor[:, col_indx].long()
            embeddings = flatten(self.embeds[col_indx](classes_tensor))
            embeds.append(embeddings)
        # Make visible units vector
        x = th.cat(
            [input_tensor[:, self.bool_mask], input_tensor[:, self.continuous_mask]]
            + embeds,
            dim=1,
        )
        x = self.recognition_network(x)
        x = self.generation_network(x)
        # Piece-wise activations
        logits_vec = x[:, self.bool_mask]
        continuous_vec = self.activation(x[:, self.continuous_mask])
        categorical_outputs = {}
        for col_indx in self.categorical_col_ind.keys():
            visible = self.output_embeds[col_indx](x)
            categorical_outputs[col_indx] = flatten(visible)
        return logits_vec, continuous_vec, categorical_outputs

    def fit(self, inputs):
        self.train()
        for batch in inputs:
            batch_metrics = self.partial_fit(batch)

    def partial_fit(self, X):
        self.train()
        self.zero_grad()
        # TODO: Optionally do not evaluate loss on nan inputs
        # nanmask = th.isnan(X)
        if isinstance(X, pd.DataFrame):
            X_ts = mixed_to_tensor(X)
        else:
            X_ts = th.from_numpy(X)
        logit_output, continous_output, categorical_outputs = self(X_ts)
        # Logits
        logit_targets = X_ts[:, self.bool_mask]
        logit_loss = self.loss_fn["boolean"](logit_output, logit_targets)
        logit_loss.backward()
        # Continuous vars
        continous_targets = X_ts[:, self.continuous_mask]
        mse_loss = self.loss_fn["continuous"](continous_output, continous_targets)
        mse_loss.backward()
        metrics = dict(logit=logit_loss, continuous=mse_loss)
        # Categorical vars
        for col_indx in self.categorical_col_ind.keys():
            preds = categorical_outputs[col_indx]
            targets = X_ts[:, col_indx].long()
            nll_loss = self.loss_fn[col_indx](preds, targets)
            nll_loss.backward()
            metrics[col_indx] = nll_loss
        self.optimizer.step()
        return metrics

    def infer(self, func, inputs):
        self.eval()
        with th.no_grad():
            inputs = th.from_numpy(inputs)
            output = func(inputs)
            output = output.detach().numpy()
        return output

    def sample(self, X=None):
        if X is None:
            X = np.random.randn(1, self.latent_dims[-1])
        return self.generate(X)

    def transform(self, X):
        return self.infer(self.recognition_network, X)

    def generate(self, X):
        return self.infer(self.generation_network, X)

    def reconstruct(self, inputs):
        # TODO: This is wrong (only gives output visible units' activations)
        func = lambda x: self.generation_network(self.recognition_network(x))
        return self.infer(func, inputs)
