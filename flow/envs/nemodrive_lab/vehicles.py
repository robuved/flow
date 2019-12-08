from flow.core.params import VehicleParams
from flow.controllers.car_following_models import IDMController, CFMController
from flow.controllers.routing_controllers import ContinuousRouter, MinicityRouter
import numpy as np
from flow.controllers.rlcontroller import RLController

from flow.envs.nemodrive_lab.lane_change_controller import PeriodicLaneChangeController


def get_vehicles():
    num_vehicles = 22

    # Params: {param_name: (mean, std, min_value, max_value),...}
    params = dict({
        "v0": (30, 10, 0, 100),  # float desirable velocity, in m/s (default: 30)
        "T": (1, 0.5, 0.1, 10),  # float safe time headway, in s (default: 1)
        "a": (1, 0.2, 0.1, 3),  # float max acceleration, in m/s2 (default: 1)
        "b": (1.5, 0.2, 0.1, 3),  # float comfortable deceleration, in m/s2 (default: 1.5)
        "s0": (2, 0.5, 0.5, 10),  # float linear jam distance, in m (default: 2)
        "noise": (2, 1, 0., 10),
        # float std dev of normal perturbation to the acceleration (default: 0)
    })

    vehicles = VehicleParams()

    veh_param = {p: np.clip(np.random.normal(x[0], x[1]), x[2], x[3]) for p, x in
                 params.items()}

    vehicles.add(f"rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=1)

    for i in range(num_vehicles):
        veh_param = {p: np.clip(np.random.normal(x[0], x[1]), x[2], x[3]) for p, x in
                     params.items()}
        vehicles.add(f"human_{i}",
                     acceleration_controller=(IDMController, veh_param),
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=1)

    return vehicles


def get_vehicles_with_lane_change():
    num_vehicles = 22

    # Params: {param_name: (mean, std, min_value, max_value),...}
    params = dict({
        "v0": (30, 10, 0, 100),  # float desirable velocity, in m/s (default: 30)
        "T": (1, 0.5, 0.1, 10),  # float safe time headway, in s (default: 1)
        "a": (1, 0.2, 0.1, 3),  # float max acceleration, in m/s2 (default: 1)
        "b": (1.5, 0.2, 0.1, 3),  # float comfortable deceleration, in m/s2 (default: 1.5)
        "s0": (2, 0.5, 0.5, 10),  # float linear jam distance, in m (default: 2)
        "noise": (2, 1, 0., 10),
        # float std dev of normal perturbation to the acceleration (default: 0)
    })

    lane_change_param = dict({
        "lane_change_params": {"lane_change_freqs": [30, 150]}
    })

    vehicles = VehicleParams()

    veh_param = {p: np.clip(np.random.normal(x[0], x[1]), x[2], x[3]) for p, x in
                 params.items()}

    vehicles.add(f"rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(ContinuousRouter, {}),
                 num_vehicles=1)

    for i in range(num_vehicles):
        veh_param = {p: np.clip(np.random.normal(x[0], x[1]), x[2], x[3]) for p, x in
                     params.items()}
        vehicles.add(f"human_{i}",
                     acceleration_controller=(IDMController, veh_param),
                     lane_change_controller=(PeriodicLaneChangeController, lane_change_param),
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=1)

    return vehicles
