import random, datetime
from pathlib import Path

import gym

from metrics import MetricLogger
from agent import ATARI
from wrappers import ResizeObservation, SkipFrame, make_atari, wrap_deepmind
import numpy as np
env = gym.make("BreakoutNoFrameskip-v4")

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")

# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)


env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2022-08-05T20-24-17/atari_net_5.chkpt')
atari = ATARI(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
atari.exploration_rate = atari.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):

    state = env.reset()
    state_l = np.transpose((np.array(state)), (2, 0, 1))
    while True:

        env.render()

        action = atari.act(state_l)

        next_state, reward, done, info = env.step(action)

        next_state_l = np.transpose((np.array(next_state)), (2, 0, 1))
        atari.cache(state_l, next_state_l, action, reward, done)

        logger.log_step(reward, None, None)

        state_l = next_state_l

        if done:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=atari.exploration_rate,
            step=atari.curr_step
        )
