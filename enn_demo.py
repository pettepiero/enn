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

importlib.reload(regression_data)
importlib.reload(losses)


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

importlib.reload(regression_data)
importlib.reload(losses)


@dataclasses.dataclass
class Config:
    num_batch: int = 1_000
    index_dim: int = 10
    num_index_samples: int = 10
    seed: int = 0
    prior_scale: float = 5.0
    learning_rate: float = 1e-3
    noise_std: float = 0.1


FLAGS = Config()


# @title Create the regression experiment

# Generate dataset
dataset = regression_data.make_dataset()

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
experiment.train(FLAGS.num_batch)


# @title Plot the output
p = regression_data.make_plot(experiment=experiment, num_sample=10)
_ = p.draw()
_.savefig("./regression.png", dpi=300)
