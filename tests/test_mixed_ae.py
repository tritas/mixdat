# coding=utf-8
import logging
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pytest
from mixdat.models.th_training import train_torch_model

tmpdir = mkdtemp()
logging.basicConfig(level=logging.DEBUG)

vocab_size = 20
target_embedding_dim = 5
batch_size = 1
epochs = 5
hidden = 5
lr = 0.2

# Dummy FFQ dataframe
sample_cols = [
    "user",
    "session",
    "weight",
    "product",
    "marketplace",
    "day",
    "week",
    "month",
]
sizes = [1000, 1000, 1, vocab_size, 10, 31, 52, 12]

# TODO: Define fit & transform data
fit_data = []
transform_data = []


def test_multi_embed_setup():
    assert entity_embedder.emb["product"].embedding_dim == target_proj_size


def test_multi_embed_dataframe_transform():
    X_multi_emb = pd.DataFrame({"product": [0, 1], "household": [0, 1]})
    X_multi_emb_tf = entity_embedder.transform(X_multi_emb)
    assert len(X_multi_emb) == len(X_multi_emb.columns)
    for col, output in X_multi_emb_tf.items():
        dim = entity_embedder.emb[col].embedding_dim
        assert output.shape == (len(X_multi_emb), dim)


@pytest.mark.parametrize("model,X,y", fit_data)
def test_own_fit(model, X, y):
    loss_value = model.fit(X, y)
    # No good generic test here as it depends on the loss function
    mean_sample_loss = loss_value / len(y)
    assert 0 < mean_sample_loss < 10


@pytest.mark.parametrize("model,X,y", fit_data)
def test_fit_loop(model, X, y, jobs=1):
    model, losses = train_torch_model(
        model, X, y, batch_size, epochs, tol=0, n_jobs=jobs
    )
    print(model)
    print(losses)
    assert len(losses) == epochs
    assert np.all(np.less_equal(np.diff(losses, 1), 0))


@pytest.mark.parametrize("model,X,output_dim", transform_data)
def test_inner_transform(model, X, output_dim):
    X_tf = model.transform(X)
    assert X_tf.shape == (len(X), output_dim)
