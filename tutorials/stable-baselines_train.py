"""Ring road example.

Trains a single autonomous vehicle to stabilize the flow of 21 human-driven
vehicles in a variable length ring road.
"""

import argparse
import json
import os

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from functools import partial

from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.envs.nemodrive_lab.build_envs import make_lab_env

# time horizon of a single rollout
HORIZON = 128


def update_linear_schedule(frac, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr * frac
    return lr


def run_model(num_cpus=1, rollout_size=50, num_steps=50, num_mini_batch=4, tensorboard_log=None):
    """Run the model for num_steps if provided. The total rollout length is rollout_size."""
    if num_cpus == 1:
        constructor = env_constructor(params=flow_params, version=0)()
        env = DummyVecEnv([lambda: constructor])  # The algorithms require a vectorized environment to run
    else:
        env = SubprocVecEnv([env_constructor(params=flow_params, version=i) for i in range(num_cpus)])

    model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size, cliprange=0.1, vf_coef=0.5, ent_coef=0.01,
                 learning_rate=partial(update_linear_schedule, initial_lr=2.5e-4), nminibatches=num_mini_batch,
                 tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=num_steps)
    return model


if __name__ == "__main__":
    # "--num-cpus 4 --num-steps 2600000 --rollout-size 128 --result-name lab_env2_train --env-name lab_env2 --num-mini-batch 4 --tensorboard-log ./lab_env2"
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-cpus', type=int, default=1, help='How many CPUs to use')
    parser.add_argument('--num-steps', type=int, default=5000, help='How many total steps to perform learning over')
    parser.add_argument('--rollout-size', type=int, default=1000, help='How many steps are in a training batch.')
    parser.add_argument('--result-name', type=str, default='lab_env2', help='Name of saved model')
    parser.add_argument("--env-name", type=str, default='lab_env2')
    parser.add_argument("--num-mini-batch", type=int, default=4)
    parser.add_argument("--tensorboard-log", type=str, default=None)

    args = parser.parse_args()

    gym_name, flow_params, flow_json = make_lab_env(args.env_name)
    model = run_model(args.num_cpus, args.rollout_size, args.num_steps, args.num_mini_batch,
                      tensorboard_log=args.tensorboard_log)
    # Save the model to a desired folder and then delete it to demonstrate loading
    if not os.path.exists(os.path.realpath(os.path.expanduser('~/baseline_results'))):
        os.makedirs(os.path.realpath(os.path.expanduser('~/baseline_results')))
    path = os.path.realpath(os.path.expanduser('~/baseline_results'))
    save_path = os.path.join(path, args.result_name)

    print('Saving the trained model!')
    model.save(save_path)
    # dump the flow params
    with open(os.path.join(path, args.result_name) + '.json', 'w') as outfile:
        json.dump(flow_params, outfile,  cls=FlowParamsEncoder, sort_keys=True, indent=4)
    del model
    del flow_params

    # Replay the result by loading the model
    print('Loading the trained model and testing it out!')
    model = PPO2.load(save_path)
    flow_params = get_flow_params(os.path.join(path, args.result_name) + '.json')
    flow_params['sim'].render = True
    env_constructor = env_constructor(params=flow_params, version=0)()
    env = DummyVecEnv([lambda: env_constructor])  # The algorithms require a vectorized environment to run
    obs = env.reset()
    reward = 0
    for i in range(flow_params['env'].horizon):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        reward += rewards
    print('the final reward is {}'.format(reward))
