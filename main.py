import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random, datetime
from pathlib import Path

import gym
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from metrics import MetricLogger
from agent import ATARI
from wrappers import ResizeObservation, SkipFrame


env = gym.make("BreakoutNoFrameskip-v4")


# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/atari.chkpt')
atari = ATARI(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000

### for Loop that train the model num_episodes times by playing the game
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # 3. Show environment (the visual) [WIP]
        # env.render()

        # 4. Run agent on the state
        action = atari.act(state)

        # 5. Agent performs action
        next_state, reward, done, info = env.step(action)

        # 6. Remember
        atari.cache(state, next_state, action, reward, done)

        # 7. Learn
        q, loss = atari.learn()

        # 8. Logging
        logger.log_step(reward, loss, q)

        # 9. Update state
        state = next_state

        # 10. Check if end of game
        if done or info['flag_get']:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=atari.exploration_rate,
            step=atari.curr_step
        )
