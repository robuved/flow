from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController

import numpy as np


class PeriodicLaneChangeController(BaseLaneChangeController):
    def __init__(self, veh_id, lane_change_params=None):
        """Instantiate the base class for lane-changing controllers."""
        if lane_change_params is None:
            lane_change_params = {}

        self.veh_id = veh_id
        self.lane_change_params = lane_change_params
        self.freq_interval = np.random.randint(*lane_change_params["lane_change_freqs"])

    def get_lane_change_action(self, env):
        change_lane = 0
        if env.step_counter % self.freq_interval == 0:
            change_lane = np.random.randint(0, 2) * 2 - 1
        """See parent class."""

        return change_lane
