from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian
from tutorials.distributions import MixedDistributionModule


class MyPolicy(Policy):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super().__init__(obs_shape, action_space, base, base_kwargs)
        self.dist = MixedDistributionModule(self.base.output_size)