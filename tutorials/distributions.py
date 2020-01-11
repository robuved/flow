import torch
import torch.nn as nn
from a2c_ppo_acktr.distributions import Categorical, FixedCategorical, DiagGaussian

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/211


class MyMixedDistribution:
    def __init__(self, *children):
        self.children = children

    def log_probs(self, actions):
        r = []
        for i, child in enumerate(self.children):
            r.append(child.log_probs(actions[:, i].view(-1, 1)).view(-1, 1))
        r = torch.cat(r, dim=1).sum(dim=1, keepdim=True)
        return r

    def entropy(self):
        e = tuple(map(lambda x: x.entropy().view(-1, 1), self.children))
        r = torch.cat(e, dim=1).sum(dim=1, keepdim=True)
        return r

    def sample(self):
        e = tuple(map(lambda x: x.sample().float(), self.children))
        s = column_to_row(e)
        return s

    def mode(self):
        e = tuple(map(lambda x: x.mode().float(), self.children))
        return column_to_row(e)


class MixedDistributionModule(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.accel_dist = DiagGaussian(num_inputs, 1)
        self.lane_dist = Categorical(num_inputs, 3)
        self.mixed = MyMixedDistribution(self.accel_dist, self.lane_dist)

    def forward(self, x):
        accel_distr = self.accel_dist(x)
        lane_distr = self.lane_dist(x)

        return MyMixedDistribution(accel_distr, lane_distr)


def column_to_row(e):
    return torch.cat(e, dim=1)
    # batch_size = e[0].shape[0]
    # f = [c.view(-1) for c in e]
    # rows = []
    # for i in range(batch_size):
    #     row = []
    #     for c in f:
    #         row.append(c[i])
    #     rows.append(torch.stack(row))
    # r = torch.stack(rows, dim=0)
    # return r


# class MixedDistribution:
#     # a category with mixed episodic sizes
#     def __init__(self, *children):
#         self.children = children
#
#     def log_probs(self, actions):
#         r = []
#         for i, child in enumerate(self.children):
#             r.append(child.log_probs(actions[i].view(-1, 1)).view(-1, 1))
#         r = torch.cat(r, dim=1)
#         return r
#
#     def entropy(self):
#         e = tuple(map(lambda x: x.entropy(), self.children))
#         return column_to_row(e)
#
#     def mode(self):
#         e = tuple(map(lambda x: x.mode(), self.children))
#         return column_to_row(e)
#
#     def sample(self):
#         e = tuple(map(lambda x: x.sample(), self.children))
#         return column_to_row(e)
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.children})"
#
#
# class CatDistribution(MixedDistribution):
#     # batch level mixture of features
#     def log_probs(self, actions):
#         r = []
#         for i, child in enumerate(self.children):
#             r.append(child.log_probs(actions[:, i]).view(-1, 1))
#         r = torch.cat(r, dim=1).sum(dim=1, keepdim=True)
#         return r
#
#
# class NodeObjective(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(NodeObjective, self).__init__()
#         self.actions = Categorical(num_inputs, num_outputs)
#
#     def forward(self, x):
#         a, n, n_batch, q, q_batch = x
#         a_dist = self.actions(a)
#         n_dist = []
#         q_dist = []
#         for i in range(a.shape[0]):
#             # 1 x N
#             ni = (n[n_batch == i]).transpose(0, 1)
#             qi = (q[q_batch == i]).transpose(0, 1)
#             ni_dist = FixedCategorical(logits=ni)
#             qi_dist = FixedCategorical(logits=qi)
#             n_dist.append(ni_dist)
#             q_dist.append(qi_dist)
#         n_dist = MixedDistribution(*n_dist)
#         q_dist = MixedDistribution(*q_dist)
#         return CatDistribution(a_dist, n_dist, q_dist)
