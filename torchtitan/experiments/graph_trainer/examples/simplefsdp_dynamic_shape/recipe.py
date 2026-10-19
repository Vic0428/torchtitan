# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SimpleFSDP demo recipes for studying shape inference in GraphTrainer."""

from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.llama3.model import GraphTrainerLlama3Model

from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def simplefsdp_llama3_fsdp4():
    config = to_graph_trainer_config(
        llama3_debugmodel(attn_backend="flex"), GraphTrainerLlama3Model.Config
    )
    config.compile = GraphTrainerCompileConfig()
    config.parallelism.data_parallel_shard_degree = 4
    config.training.steps = 10
    config.debug.seed = 42
    config.metrics.enable_tensorboard = False
    return config
