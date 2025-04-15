# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime
import wandb

# os.environ["PATH"] = "/home/serho/miniconda3/envs/isaac2/bin:/home/serho/miniconda3/condabin:" + os.environ["PATH"]


# os.environ["PATH"] = "/nethome/srho31/flash/miniconda3/envs/isaac2/bin:/nethome/srho31/flash/miniconda3/condabin:" + os.environ["PATH"]
# os.environ["PATH"] = "/home/srho31/miniconda3/envs/isaac/bin:/home/srho31/miniconda3/condabin:" + os.environ["PATH"]

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from legged_gym.debugger import break_into_debugger

def train(args):
    mode = "online"
    if not args.debug:
        wandb.init(project=args.task, name=args.run_name, mode=mode, dir="../../logs", sync_tensorboard=True)
        # if args.run_name=="go1":
        name = args.task.split('_')
        # wandb.save(LEGGED_GYM_ENVS_DIR + "/"+ "a1/" + args.task + "_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + "/"+ name[0]+"/"+ args.task + "_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
        wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
