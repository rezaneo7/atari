import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gym
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from metrics import MetricLogger
from agent import ATARI
from wrappers import ResizeObservation, SkipFrame, make_atari, wrap_deepmind


env = gym.make("BreakoutNoFrameskip-v4")

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")

# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/atari.chkpt')
atari = ATARI(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 100000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()
    state_l = np.transpose((np.array(state)), (2, 0, 1))
    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        

        action = atari.act(state_l)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        next_state_l = np.transpose((np.array(next_state)), (2, 0, 1))


        atari.cache(state_l, next_state_l, action, reward, done)

        # 7. Learn
        q, loss = atari.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state_l = next_state_l

        # 10. Check if end of game
        if done:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=atari.exploration_rate,
            step=atari.curr_step
        )
