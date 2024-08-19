# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Functions for 1D regression data."""
import chex
from enn import datasets
from enn import networks
from enn import utils
from enn.supervised import base as supervised_base
import haiku as hk
import jax
import numpy as np
import pandas as pd
import plotnine as gg


def make_regression_df() -> pd.DataFrame:
  """Creates our regression dataset."""
  seed = 0
  n_data = 10
  x = np.concatenate([np.linspace(0, 0.5, n_data), np.linspace(1, 1.5, n_data)])
  w = np.random.RandomState(seed).randn(n_data * 2) * 0.1
  y = x + np.sin(3 * x) + np.sin(12 * x) + w
  return pd.DataFrame({'x': x, 'y': y}).reset_index()


def make_dataset(extra_input_dim: int = 1) -> datasets.ArrayBatchIterator:
  """Factory method to produce an iterator of Batches."""
  df = make_regression_df()
  data = datasets.ArrayBatch(
      x=np.vstack([df['x'].values, np.ones((extra_input_dim, len(df)))]).T,
      y=df['y'].values[:, None],
  )
  chex.assert_shape(data.x, (None, 1 + extra_input_dim))
  return utils.make_batch_iterator(data)


def make_plot(experiment: supervised_base.BaseExperiment,
              num_sample: int = 20,
              extra_input_dim: int = 1) -> gg.ggplot:
    """Generate a regression plot with sampled predictions."""
    plot_df = make_plot_data(
      experiment, num_sample=num_sample, extra_input_dim=extra_input_dim)

    print(f"DEBUG: Printing num_sample: {num_sample}")
    print(f"DEBUG: plot_df.shape = {plot_df.shape}")

    plot_df["k"] = plot_df["k"].astype(
        "category"
    )  # Convert k to a categorical variable

    p = (gg.ggplot()
       + gg.aes('x', 'y')
       + gg.geom_point(data=make_regression_df(), size=3, colour='blue')
       + gg.geom_line(gg.aes(group='k', color='k'), data=plot_df, alpha=0.5)
      )

    return p


def make_plot_data(experiment: supervised_base.BaseExperiment,
                   num_sample: int = 20,
                   extra_input_dim: int = 1) -> pd.DataFrame:
  """Generate a panda dataframe with sampled predictions."""
  preds_x = np.vstack([np.linspace(-1, 2), np.ones((extra_input_dim, 50))]).T

  data = []
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed=0))
  for k in range(num_sample):
    net_out = experiment.predict(preds_x, key=next(rng))
    preds_y = networks.parse_net_output(net_out)
    data.append(pd.DataFrame({'x': preds_x[:, 0], 'y': preds_y[:, 0], 'k': k}))
  plot_df = pd.concat(data)

  return plot_df
