"""Environments that can train both lane change and acceleration behaviors."""

from flow.envs.ring.accel import AccelEnv
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple import Tuple
from gym.spaces.discrete import Discrete

import numpy as np

ADDITIONAL_ENV1_PARAMS = {
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
}


class LaneChangeAccelEnv1(AccelEnv):
    """
    Modified version of LaneChangeAccelEnv:

    * reward RL agent progress
    * penalty for frontal and lateral collision
    * lane change reward
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV1_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))
        self.collision_reward = ADDITIONAL_ENV1_PARAMS["collision_reward"]
        self.frontal_collision_distance = ADDITIONAL_ENV1_PARAMS["frontal_collision_distance"]
        self.lateral_collision_distance = ADDITIONAL_ENV1_PARAMS["lateral_collision_distance"]
        self.forward_progress_gain = ADDITIONAL_ENV1_PARAMS["forward_progress_gain"]
        self.action_space_box = ADDITIONAL_ENV1_PARAMS["action_space_box"]

        super().__init__(env_params, sim_params, network, simulator)

        self.num_lanes = max(self.k.network.num_lanes(edge)
                             for edge in self.k.network.get_edge_list())
        self.prev_lane = self.crt_lane = None, None

    @property
    def action_space(self):
        """See class definition."""
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        num_rl_agents = self.initial_vehicles.num_rl_vehicles

        # return (Box(np.array(lb), np.array(ub), dtype=np.float32),
        if self.action_space_box:
            # Models that cannot interpret Complex actions
            return Box(
                    low=-abs(max_decel),
                    high=max_accel,
                    shape=(num_rl_agents * 2,),
                    dtype=np.float32)
        else:
            return Tuple(
                (Box(
                    low=-abs(max_decel),
                    high=max_accel,
                    shape=(num_rl_agents,),
                    dtype=np.float32), ) +
                (Discrete(3),) * num_rl_agents
            )

    @property
    def observation_space(self):
        """See class definition."""
        return Box(
            low=0,
            high=1,
            shape=(3 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def compute_collision_reward(self, rl_actions):
        collision_distance = self.frontal_collision_distance
        collision_reward = self.collision_reward
        lateral_collision_distance = self.lateral_collision_distance
        vehicles = self.k.vehicle
        num_lanes = self.num_lanes

        reward = 0
        if rl_actions is None:
            return reward

        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in vehicles.get_rl_ids()
        ]

        directions = rl_actions[1::2]

        for i, rel_id in enumerate(sorted_rl_ids):
            lane = vehicles.get_lane(rel_id)
            direction = directions[i]
            headways = vehicles.get_lane_headways(rel_id)

            # Check front collision
            if headways[lane] < collision_distance:
                reward += collision_reward

            # Check lateral collision
            if direction != 0:
                tailways = vehicles.get_lane_tailways(rel_id)

                # Action has already been applied in the env
                # Calculate lane change based on intent
                prev_lane = self.prev_lane[i]
                new_lane = int(prev_lane + direction)

                # Trying to get out of the road
                if new_lane < 0 or new_lane >= num_lanes:
                    reward += collision_reward
                elif np.abs(headways[new_lane]) < lateral_collision_distance or \
                        np.abs(tailways[new_lane]) < lateral_collision_distance:
                    reward += collision_reward

        return reward

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # compute the system-level performance of vehicles from a velocity
        # perspective
        reward = rewards.rl_forward_progress(self, gain=self.forward_progress_gain)

        # Calculate collision reward
        collision_r = self.compute_collision_reward(rl_actions)
        reward += collision_r

        # punish excessive lane changes by reducing the reward by a set value
        # every time an rl car changes lanes (10% of max reward)
        for veh_id in self.k.vehicle.get_rl_ids():
            if self.k.vehicle.get_last_lc(veh_id) == self.time_counter:
                reward -= 0.1

        return reward

    def get_state(self):
        """See class definition."""
        # normalizers
        state_size = len(self.initial_ids)
        no_cars = len(self.sorted_ids)
        empty_cars = state_size - no_cars

        max_speed = self.k.network.max_speed()
        length = self.k.network.length()
        max_lanes = max(
            self.k.network.num_lanes(edge)
            for edge in self.k.network.get_edge_list())

        speed = [self.k.vehicle.get_speed(veh_id) / max_speed
                 for veh_id in self.sorted_ids] + [0.] * empty_cars
        pos = [self.k.vehicle.get_x_by_id(veh_id) / length
               for veh_id in self.sorted_ids] + [0.] * empty_cars
        lane = [self.k.vehicle.get_lane(veh_id) / max_lanes
                for veh_id in self.sorted_ids] + [0.] * empty_cars

        # Save to know previous lane of agents
        sorted_rl_ids = [
            i for i, veh_id in enumerate(self.sorted_ids)
            if veh_id in self.k.vehicle.get_rl_ids()
        ]

        self.prev_lane = self.crt_lane
        self.crt_lane = [lane[x] * max_lanes for x in sorted_rl_ids]

        return np.array(speed + pos + lane)

    def _apply_rl_actions(self, actions):
        """See class definition."""
        acceleration, direction = np.split(actions, 2)

        # re-arrange actions according to mapping in observation space
        if len(self.k.vehicle.get_rl_ids()) <= 0:
            raise ValueError("No RL agent id")

        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]

        # represents vehicles that are allowed to change lanes
        lane_change_duration = self.env_params.additional_params["lane_change_duration"]
        if lane_change_duration > 0:
            non_lane_changing_veh = \
                [self.time_counter <=
                 self.env_params.additional_params["lane_change_duration"]
                 + self.k.vehicle.get_last_lc(veh_id)
                 for veh_id in sorted_rl_ids]

            # vehicle that are not allowed to change have their directions set to 0
            direction[non_lane_changing_veh] = \
                np.array([0] * sum(non_lane_changing_veh))

        self.k.vehicle.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.k.vehicle.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(self):
        """Define which vehicles are observed for visualization purposes."""
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

    def clip_actions(self, rl_actions=None):
        """Clip & redo direction the actions passed from the RL agent.

        Parameters
        ----------
        rl_actions : array_like
            list of actions provided by the RL algorithm

        Returns
        -------
        array_like
            The rl_actions clipped according to the box or boxes
        """
        # ignore if no actions are issued
        if rl_actions is None:
            return

        # clip according to the action space requirements
        if isinstance(self.action_space, Box):
            rl_actions = np.clip(
                rl_actions,
                a_min=self.action_space.low,
                a_max=self.action_space.high)
        elif isinstance(self.action_space, Tuple):
            for idx, action in enumerate(rl_actions):
                subspace = self.action_space[idx]
                if isinstance(subspace, Box):
                    rl_actions[idx] = np.clip(
                        action,
                        a_min=subspace.low,
                        a_max=subspace.high)

        if isinstance(rl_actions[0], np.ndarray):
            acceleration = rl_actions[0]
            direction = np.array(rl_actions[1:])
        else:
            acceleration, direction = np.split(rl_actions, 2)

        # discrete lane change
        direction = direction.clip(0., 2.)
        direction = (direction - 1).astype(np.int)

        return np.concatenate([acceleration, direction])
