# @title General imports

import warnings

warnings.filterwarnings("ignore")

# @title Development imports
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import plotnine as gg

import dataclasses
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


# @title ENN imports
import enn
from enn import losses
from enn import networks
from enn import supervised
from enn import base
from enn import data_noise
from enn import utils
from enn.loggers import TerminalLogger
from enn.supervised import classification_data
from enn.supervised import regression_data

import importlib

from jaxlib.xla_extension import ArrayImpl


@dataclasses.dataclass
class Config:
    num_batch: int = 100    #changed from 1000 to 100
    index_dim: int = 10
    num_index_samples: int = 10
    seed: int = 0
    prior_scale: float = 5.0
    learning_rate: float = 1e-3
    noise_std: float = 0.1


FLAGS = Config()

# @title Create the regression experiment

# Generate dataset
dataset = regression_data.make_dataset(plot_df_flag=False)

# Logger
logger = TerminalLogger("supervised_regression")

# Create Ensemble ENN with a prior network
enn = networks.MLPEnsembleMatchedPrior(
    output_sizes=[50, 50, 1],
    dummy_input=next(dataset).x,
    num_ensemble=FLAGS.index_dim,
    prior_scale=FLAGS.prior_scale,
    seed=FLAGS.seed,
)

# L2 loss on perturbed outputs
noise_fn = data_noise.GaussianTargetNoise(enn, FLAGS.noise_std, FLAGS.seed)
single_loss = losses.add_data_noise(losses.L2Loss(), noise_fn)
loss_fn = losses.average_single_index_loss(single_loss, FLAGS.num_index_samples)

# Optimizer
optimizer = optax.adam(FLAGS.learning_rate)

# Aggregating different components of the experiment
experiment = supervised.Experiment(
    enn, loss_fn, optimizer, dataset, FLAGS.seed, logger=logger
)

# Train the experiment
# experiment.train(FLAGS.num_batch)

for iteration in range(FLAGS.num_batch):
    experiment.step += 1
    batch = next(experiment.dataset)
    key = next(experiment.rng)

    experiment.state, loss_metrics = experiment._sgd_step(
        training_state=experiment.state,
        batch=batch, 
        key=key
    )

# print(
#     f"\nDEBUG: Step = {experiment.step} - len(jax.random.split(next(self.rng), self._eval_enn_samples)) = {len(jax.random.split(next(experiment.rng), experiment._eval_enn_samples))}\n"
# )

# print(f"\nDEBUG: batch type is {type(batch)}")
print(f"\nDEBUG: batch.x type is {type(batch.x)}")
# print(f"\nDEBUG: batch len is {len(batch)}\n")
print(f"\nDEBUG: batch.x len is {len(batch.x)}\n")

random_keys = jax.random.split(next(experiment.rng), experiment._eval_enn_samples)

print(f"\nDEBUG: random_keys shape = {random_keys.shape}\n")
print(random_keys[:10])

output = experiment._batch_fwd(
    experiment.state.params, experiment.state.network_state, batch.x, random_keys
)

def aggregate_predictions_acc_to_index(
    x_batch: np.ndarray, y_batch: np.ndarray, n_samples: int = 10
    ):
        """Aggregate the predictions according to the index.
        The index can't be shown. """
        pairs = []
        if x_batch.shape[1] == 2:
            x_batch = x_batch[:, 0]
        for i in range(x_batch.shape[0]):
            x_val = x_batch[i]
            y_vals = y_batch[:n_samples, i]

            for y_val in y_vals:
                pairs.append((x_val, y_val.item()))

        df = pd.DataFrame(pairs, columns=['x', 'y'])
        return df


def plot_predictions_with_index(dataframe: pd.DataFrame):
    """Plot all the predictions varying with the index, for the given dataframe."""
    p = (gg.ggplot(dataframe)
        + gg.aes('x', 'y')
        + gg.geom_point()
        + gg.ylim(-9, 1)
        + gg.xlim(-1, 2)
      )
    p.save("predictions_with_index.png", dpi=300)

new_df = aggregate_predictions_acc_to_index(batch.x, output.preds, indices=random_keys)
plot_predictions_with_index(new_df)

print(new_df.tail())
