"""Environments that can train both lane change and acceleration behaviors."""

from flow.envs.ring.accel import AccelEnv
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.discrete import Discrete

from flow.envs.nemodrive_lab.env1_lab import LaneChangeAccelEnv1

import numpy as np

ADDITIONAL_ENV2_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 0,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False,
    # Amplifier factor for rewarding progress of agent (meters * gain)
    'forward_progress_gain': 0.1,
    # Reward for collision
    'collision_reward': -1,
    # Penalty for changing lane
    'lane_change_reward': -0.1,
    # Safe distances to keep with car in front; and with lateral cars when changing lane
    'frontal_collision_distance': 2.0,  # in meters
    'lateral_collision_distance': 3.0,  # in meters
    # Return shape as box 2 - continuous values
    "action_space_box": False,
    # =====================================================================================
    # ENV 2 Params
    "pos_noise_std": [0.5, 2],  # in meters
    "pos_noise_steps_reset": 100,
    "speed_noise_std": [0.2, 0.8],  # in  m/s
    "acc_noise_std": [0.2, 0.4],  # m/s^2
}


class LaneChangeAccelEnv2(LaneChangeAccelEnv1):
    """
    Modified version of LaneChangeAccelEnv1:

    ENV 1:
    * reward RL agent progress
    * penalty for frontal and lateral collision
    * lane change reward
    + ENV 2:
    * noisy RL agent pos: accumulating noise (gaussian) - reset error periodically
    * noisy RL agent speed: gaussian noise
    * noisy command for acceleration: gaussian noise
    * other agents change lane periodically (each with it's own freq initialized random on reset)
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV2_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        add_param = env_params.additional_params

        self.pos_noise_std = add_param["pos_noise_std"]
        self.pos_noise_steps_reset = add_param["pos_noise_steps_reset"]
        self.speed_noise_std = add_param["speed_noise_std"]
        self.acc_noise_std = add_param["acc_noise_std"]

        super().__init__(env_params, sim_params, network, simulator)

        self._crt_pos_noise_accum = None
        self._crt_pos_noise_std = None
        self._crt_speed_noise_std = None
        self._crt_acc_noise_std = None
        self._reset_noise()
        self._pos_noise = []

    def _reset_noise(self):
        self._crt_pos_noise_accum = 0
        self._crt_pos_noise_std = np.random.uniform(*self.pos_noise_std)
        self._crt_speed_noise_std = np.random.uniform(*self.speed_noise_std)
        self._crt_acc_noise_std = np.random.uniform(*self.acc_noise_std)

    def get_state(self):
        state = super().get_state()
        no_cars = len(self.initial_ids)
        length = self.k.network.length()
        max_speed = self.k.network.max_speed()

        # Add noise to RL agent speed
        act_speed_f = state[0]
        speed_noise = np.random.normal(0, self._crt_speed_noise_std) / max_speed
        state[0] += speed_noise

        # Add noise to RL agent pos
        if self.step_counter % self.pos_noise_steps_reset == 0:
            self._crt_pos_noise_accum = 0
        self._crt_pos_noise_accum += np.random.normal(0, self._crt_pos_noise_std) * act_speed_f
        self._pos_noise.append(self._crt_pos_noise_accum)
        state[no_cars] += (self._crt_pos_noise_accum / length)

        return state

    def reset(self):
        self._reset_noise()

        return super().reset()

    def _apply_rl_actions(self, actions):
        acc_noise = np.random.normal(0, self._crt_acc_noise_std)
        actions[0] += acc_noise
        """See class definition."""
        super()._apply_rl_actions(actions)
