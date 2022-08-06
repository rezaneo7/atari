import random, datetime
from pathlib import Path
import torch

import gym

from metrics import MetricLogger
from agent import ATARI
from wrappers import ResizeObservation, SkipFrame, make_atari, wrap_deepmind,wrap_deepmind_2_
import numpy as np
# visualize game by gif
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation

def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif
    """
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename:
        anim.save(filename, dpi=72, writer='pillow')


# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")

# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)


save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = Path('checkpoints/2022-08-05T23-12-29/atari_net_3.chkpt')
atari = ATARI(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
atari.exploration_rate = atari.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 100

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")

# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(7)


env2 = make_atari("BreakoutNoFrameskip-v4")
env2 = wrap_deepmind_2_(env2, frame_stack=True, scale=True)
env2.seed(7)

frames = []
frames2 = []
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print ('Your system: ' + str(device))

atari.net.eval()
with torch.no_grad():

  for _ in range(5):
      t = 0
      reward_sum = 0.0
      reward_sum2 = 0.0

      state = env.reset()
      state2 = env2.reset()
      frames.append(state)
      frames2.append(state2[:,:,0:3])
      state = torch.tensor(np.transpose((np.array(state) * 255), (2, 0, 1)), dtype=torch.uint8).to(device)


      while True:
          t += 1 
          # ----------------------------------------------------------------------------
          # choose action (greedy)
          # ----------------------------------------------------------------------------
          atari.net.eval()
          with torch.no_grad():
            x = state.to(torch.float32).unsqueeze(0) / 255.0
            outputs = atari.net(x, 'online')
          action = np.argmax(outputs[0].tolist())     
          
          # ----------------------------------------------------------------------------
          # do the step 
          # ----------------------------------------------------------------------------
          next_state, reward, done, info = env.step(action)
          next_state2, reward2, done2, info2 = env2.step(action)

          frames.append(next_state)
          frames2.append(next_state2[:,:,0:3])
          next_state = torch.tensor(np.transpose((np.array(next_state) * 255), (2, 0, 1)), dtype=torch.uint8).to(device)
          reward_sum += reward
          reward_sum2 += reward2

          # ----------------------------------------------------------------------------
          # go to next_state 
          # ----------------------------------------------------------------------------
          state = next_state         
          
          if done:
              print(f"reward_sum -> {reward_sum} | reward_sum2 -> {reward_sum2} | num_step -> {t}")  
              break

env.close()
save_frames_as_gif(frames2, filename='result.gif')

"""
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
"""