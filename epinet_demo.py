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
import dill
import tensorflow as tf
import tensorflow_datasets as tfds

# @title ENN imports
import enn
from enn import datasets
from enn.checkpoints import base as checkpoint_base
from enn.networks.epinet import base as epinet_base
from enn.checkpoints import utils
from enn.checkpoints import imagenet
from enn.checkpoints import catalog
from enn.loggers import TerminalLogger
from enn import metrics as enn_metrics

with open('./processed_batch.npzs', 'rb') as file:
  batch = dill.load(file)
images, labels = batch['images'], batch['labels']

# Define a dict of metrics including `accuracy`, `marginal nll`, and `joint nll`.
evaluation_metrics = {
    "accuracy": enn_metrics.make_accuracy_calculator(),
    "marginal nll": enn_metrics.make_nll_marginal_calculator(),
    "joint nll": enn_metrics.make_nll_polyadic_calculator(tau=10, kappa=2),
}

# Get the Epinet checkpoint
epinet_resnet50_imagenet_ckpt = catalog.ImagenetModels.RESNET_50_FINAL_EPINET.value


# Set the number of sample logits per input image
num_enn_samples = 100
# Recover the enn sampler
epinet_enn_sampler = utils.make_epinet_sampler_from_checkpoint(
    epinet_resnet50_imagenet_ckpt,
    num_enn_samples=num_enn_samples,
)
# Get the epinet logits
key = jax.random.PRNGKey(seed=0)
print(f"\nDEBUG: key: {key}\n")
epinet_logits = epinet_enn_sampler(images, key)
print(f"\nDEBUG: epinet_logits: {epinet_logits}\n")
# epinet logits has shape [num_enn_sample, eval_batch_size, num_classes]
print(f"\nDEBUG: epinet_logits.shape: {epinet_logits.shape}\n")

# Labels loaded from our dataset has shape [eval_batch_size,]. Our evaluation
# metrics requires labels to have shape [eval_batch_size, 1].
eval_labels = labels[:, None]
# Evaluate
epinet_results = {
    key: float(metric(epinet_logits, eval_labels))
    for key, metric in evaluation_metrics.items()
}
print(f"epinet_results: {epinet_results}")

print(f"\nDEBUG: images[0]: {images[0]}\n")

#######################################################################à
# Load pre-trained ResNet

# Get the ResNet-50 checkpoint
resnet50_imagenet_ckpt = catalog.ImagenetModels.RESNET_50.value

# Set the number of sample logits per input image to 1
num_enn_samples = 1
# Recover the enn sampler
resnet50_enn_sampler = utils.make_enn_sampler_from_checkpoint(
    resnet50_imagenet_ckpt,
    num_enn_samples=num_enn_samples,
)
# Get the epinet logits
key = jax.random.PRNGKey(seed=0)
resnet50_logits = resnet50_enn_sampler(images, key)

# ResNet logits has shape [num_enn_sample, eval_batch_size, num_classes]
print(f"\nDEBUG: resnet50_logits.shape: {resnet50_logits.shape}\n")

# Labels loaded from our dataset has shape [eval_batch_size,]. Our evaluation
# metrics requires labels to have shape [eval_batch_size, 1].
eval_labels = labels[:, None]
# Evaluate
resnet50_results = {
    key: float(metric(resnet50_logits, eval_labels))
    for key, metric in evaluation_metrics.items()
}
print(f"\nresnet50_results: {resnet50_results}\n")


# Make a dataframe of the results
resnet50_results["model"] = "resnet"
epinet_results["model"] = "epinet"
df = pd.DataFrame([resnet50_results, epinet_results])


# Compare the results
plt_df = pd.melt(df, id_vars=["model"], value_vars=evaluation_metrics.keys())
p = (
    gg.ggplot(plt_df)
    + gg.aes(x="model", y="value", fill="model")
    + gg.geom_col()
    + gg.facet_wrap(
        "variable",
        scales="free",
    )
    # + gg.theme(figure_size=(14, 4), panel_spacing=0.7)
)
p.draw()
p.save("epinet_vs_resnet.png", dpi=300)


