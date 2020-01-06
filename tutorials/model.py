from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian
from tutorials.distributions import MixedDistributionModule
from a2c_ppo_acktr import algo
from a2c_ppo_acktr import storage
from a2c_ppo_acktr import distributions
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import torch.nn as nn
import torch
import torch.nn.functional as F

import argparse

'''
REGRESSION TO DISCRETE DISTRIBUTION
SOMETHING BEHAVIOURS
AUXILIARY LOSS IDEAS
https://www.borealisai.com/en/blog/tutorial-4-auxiliary-tasks-deep-reinforcement-learning/

model that first values refer to current agent
need absolute position of current car AND RELATIVE position of others
link first position to controlled agent

aux_others_coef = model other agents
aux_reward_coef = reward
aux_wrong_lchange_coef = penalize useless lane change - leads to no change change lane only when needed

aux_closest_car_coef = penalize distance to closest car
aux_high_speed_lane = penalize if on low speed lane

?? avoid collision? predict collision? with lookahead



TRAINED LAB1 on ENV1
  ENV1 2314
  ENV2 2288

TRAINED LAB2 on ENV2
FIRST TRIO
  POLICY 2437
  POLICY_NO_NORMALIZE_2515
  MY_POLICY_NO_NORMALIZE 2511


'''


class MyPolicy(Policy):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super().__init__(obs_shape, action_space, base, base_kwargs)
        self.dist = MixedDistributionModule(self.base.output_size)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, actor_features, action_log_probs, dist_entropy, rnn_hxs


class MyRolloutStorage(storage.RolloutStorage):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size):
        super().__init__(num_steps, num_processes, obs_shape, action_space, recurrent_hidden_state_size)

    def feed_forward_generator(self,
                               advantages,
                               aux_data,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            batch_aux_data = {}
            for key, data in aux_data.items():
                data = data[:num_steps]

                batch_aux_data[key] = data.view(-1, data.size(-1))[indices]
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                  value_preds_batch, return_batch, rewards_batch, masks_batch, old_action_log_probs_batch, adv_targ, batch_aux_data


class MyPPO(algo.PPO):
    def __init__(self, actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 aux_coefs=None,
                 device=None):
        super().__init__(actor_critic,
                         clip_param,
                         ppo_epoch,
                         num_mini_batch,
                         value_loss_coef,
                         entropy_coef,
                         lr=lr,
                         eps=eps,
                         max_grad_norm=max_grad_norm,
                         use_clipped_value_loss=use_clipped_value_loss)
        self.aux_coefs = aux_coefs
        self.device = device
        self.init_aux_outputs()

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        aux_data = self.compute_aux_data(rollouts, self.aux_coefs)
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, aux_data, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, reward_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ, aux_data_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, actor_features, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                aux_losses = self.compute_aux_losses(actor_features, reward_batch, aux_data_batch)

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                # print(f"Value {value_loss * self.value_loss_coef}; Action {action_loss}; Entropy { - dist_entropy * self.entropy_coef}; ", end='')
                # for key, coef in self.aux_coefs.items():
                #     if coef > 0:
                #         loss += coef * aux_losses[key]
                #         print(f"{key} {aux_losses[key]}; ", end='')
                # print()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def compute_aux_data(self, rollouts, coefs):
        data = {}
        # aux_others_coef = model other agents
        speed, pos, lane = torch.split(rollouts.obs, 23, dim=2)
        if 'aux_others' in coefs and coefs['aux_others'] > 0:
            # speed + pos + lane in .obs
            # compute acc at each step
            data['acc'] = torch.empty_like(speed)
            data['acc'][:-1] = speed[1:] - speed[:-1]
            # data['bad_acc'] = torch.full(data['acc'].shape, 0)
            # data['bad_acc'][-1] = 1

            data['lane_change'] = torch.empty_like(lane)
            data['lane_change'][:-1] = (lane[:-1] == lane[1:]).float()
            # data['bad_lane_change'] = torch.full(data['lane_change'].shape, 0)
            # data['bad_lane_change'][-1] = 1

        # aux_reward_coef = predict reward
        if 'aux_reward' in coefs and coefs['aux_reward'] > 0:
            # nothing, rollouts already contain rewards
            pass

        # aux_closest_car_coef = penalize distance to closest car
        if 'aux_closest_car' in coefs and coefs['aux_closest_car'] > 0:
            different_lane = lane != lane[:, :, 0:1]
            rel_pos = pos - pos[:, :, 0:1]
            inf_rel_pos = 1.1
            rel_pos[different_lane] = inf_rel_pos

            only_pos_rel_pos = rel_pos.clone()
            only_pos_rel_pos[only_pos_rel_pos <= 0] = inf_rel_pos

            closest_from_only_pos, _ = torch.min(only_pos_rel_pos, dim=2, keepdim=True)
            closest_from_all, _ = torch.min(rel_pos, dim=2, keepdim=True)
            closest_from_all += 1

            closest = closest_from_only_pos
            where_close_from_all = closest_from_only_pos > closest_from_all
            closest[where_close_from_all] = closest_from_all[where_close_from_all]

            data['distance_to_closest'] = closest

        # aux_high_speed_lane = penalize low speed on lane in neighbourhood
        if 'aux_high_speed_lane' in coefs and coefs['aux_high_speed_lane'] > 0:
            same_lane = lane == lane[:, :, 0:1]
            other_lane = lane != lane[:, :, 0:1]

            same_lane_speed = speed.clone()
            same_lane_speed[same_lane] = 0.0
            other_lane_speed = speed.clone()
            other_lane_speed[other_lane] = 0.0

            same_lane_speed = same_lane_speed.sum(dim=2, keepdim=True) / same_lane.sum(dim=2, keepdim=True).float()
            other_lane_speed = other_lane_speed.sum(dim=2, keepdim=True) / other_lane.sum(dim=2, keepdim=True).float()
            same_lane_speed = same_lane_speed[:-1]
            other_lane_speed = other_lane_speed[:-1]

            data['on_higher_mean_speed_lane'] = torch.gt(same_lane_speed, other_lane_speed).float()

        return data

    def init_aux_outputs(self):
        self.aux_module = {}
        coefs = self.aux_coefs
        feature_size = self.actor_critic.base.output_size
        if 'aux_others' in coefs and coefs['aux_others'] > 0:
            self.aux_module['acc_predictor'] = DiagGaussian(feature_size, 23).to(self.device)
            self.aux_module['lane_predictors'] = []
            for i in range(23):
                self.aux_module['lane_predictors'].append(Categorical(feature_size, 2 * 23).to(self.device))

        # aux_reward_coef = predict reward
        if 'aux_reward' in coefs and coefs['aux_reward'] > 0:
            self.aux_module['reward_predictor'] = DiagGaussian(feature_size, 1).to(self.device)

        # aux_closest_car_coef = penalize distance to closest car
        if 'aux_closest_car' in coefs and coefs['aux_closest_car'] > 0:
            self.aux_module['closest_car_predictor'] = DiagGaussian(feature_size, 1).to(self.device)

        # aux_high_speed_lane = penalize if on lower speed lane
        if 'aux_high_speed_lane' in coefs and coefs['aux_high_speed_lane'] > 0:
            self.aux_module['on_high_speed_lane_predictor'] = Categorical(feature_size, 2).to(self.device)

    def compute_aux_losses(self, features, rewards, aux_data):
        coefs = self.aux_coefs
        # aux_others_coef = model other agents
        losses = {}
        if 'aux_others' in coefs and coefs['aux_others'] > 0:
            predicted_acc = self.aux_module['acc_predictor'](features).sample()
            losses['aux_others'] = (predicted_acc - aux_data['acc']).pow(2).mean()

            for car_index, predictor in enumerate(self.aux_module['lane_predictors']):
                s = predictor(features).sample().float()
                losses['aux_others'] += F.binary_cross_entropy_with_logits(s, aux_data['lane_change'][:, car_index:car_index+1], reduction='none').mean()

        # aux_reward_coef = predict reward
        if 'aux_reward' in coefs and coefs['aux_reward'] > 0:
            predicted_rewards = self.aux_module['reward_predictor'](features).sample()
            aux_reward_loss = (predicted_rewards - rewards).pow(2).mean()
            losses['aux_reward'] = aux_reward_loss

        # aux_closest_car_coef = penalize distance to closest car
        if 'aux_closest_car' in coefs and coefs['aux_closest_car'] > 0:
            closest = self.aux_module['closest_car_predictor'](features).sample()
            losses['aux_closest_car'] = (closest - aux_data['distance_to_closest']).pow(2).mean()
                                        # + closest

        # aux_high_speed_lane = penalize if on lower speed lane
        if 'aux_high_speed_lane' in coefs and coefs['aux_high_speed_lane'] > 0:
            loss_on_high_speed_lane = self.aux_module['on_high_speed_lane_predictor'](features).sample().float()
            losses['aux_high_speed_lane'] = F.binary_cross_entropy_with_logits(loss_on_high_speed_lane,
                                                                               aux_data['on_higher_mean_speed_lane'])
        return losses


def get_my_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    # --aux-others-coef  0.01 --aux-reward-coef  0.01 --aux-closest-car-coef  0.01 --aux-high-speed-lane-coef 0.01
    parser.add_argument(
        '--aux-others-coef',
        type=float,
        default=0.0,
        help='aux loss coef (default: 0.0)')
    parser.add_argument(
        '--aux-reward-coef',
        type=float,
        default=0.0,
        help='aux loss coef (default: 0.0)')
    parser.add_argument(
        '--aux-closest-car-coef',
        type=float,
        default=0.0,
        help='aux loss coef (default: 0.0)')
    parser.add_argument(
        '--aux-high-speed-lane-coef',
        type=float,
        default=0.0,
        help='aux loss coef (default: 0.0)')


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
