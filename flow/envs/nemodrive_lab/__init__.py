"""Empty init file to ensure documentation for the ring env is created."""

from flow.networks.figure_eight import FigureEightNetwork

from flow.envs.nemodrive_lab.networks import ADDITIONAL_NET_PARAMS_ENV1
from flow.envs.nemodrive_lab.vehicles import get_vehicles, get_vehicles_with_lane_change
from flow.envs.nemodrive_lab.env1_lab import LaneChangeAccelEnv1, ADDITIONAL_ENV1_PARAMS
from flow.envs.nemodrive_lab.env2_lab import LaneChangeAccelEnv2, ADDITIONAL_ENV2_PARAMS


ENV1 = {
    "NETWORK": FigureEightNetwork,
    "VEHICLES": get_vehicles,
    "ADDITIONAL_NET_PARAMS": ADDITIONAL_NET_PARAMS_ENV1,
    "INITIAL_CONFIG_PARAMS": {"spacing": "random", "perturbation": 50},
    "ENVIRONMENT": LaneChangeAccelEnv1,
    "ADDITIONAL_ENV_PARAMS": ADDITIONAL_ENV1_PARAMS,
    "HORIZON": 3000,
}

ENV2 = {
    "NETWORK": FigureEightNetwork,
    "VEHICLES": get_vehicles_with_lane_change,
    "ADDITIONAL_NET_PARAMS": ADDITIONAL_NET_PARAMS_ENV1,
    "INITIAL_CONFIG_PARAMS": {"spacing": "random", "perturbation": 50},
    "ENVIRONMENT": LaneChangeAccelEnv2,
    "ADDITIONAL_ENV_PARAMS": ADDITIONAL_ENV2_PARAMS,
    "HORIZON": 3000,
}

LAB_ENVS = {
    "lab_env1": ENV1,
    "lab_env2": ENV2,
}
