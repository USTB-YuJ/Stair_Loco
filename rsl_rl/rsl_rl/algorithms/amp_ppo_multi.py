# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticDepth
from rsl_rl.storage import RolloutStorageEX as RolloutStorage
from rsl_rl.storage.replay_buffer_multi import ReplayBufferMulti


class AMPPPOMulti:
    actor_critic: ActorCriticDepth
    def __init__(self,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 disc_learning_rate=2e-5,
                 policy_learning_rate=2e-5,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,
                 amp_loader_type='16dof',
                 num_amp_frames=2,
                 use_amp=False,
                 use_depth=False,
                 default_pos=None,
                 seg_loss_coef=0.1,
                 affordance_loss_coef=0.1,
                 affordance_safety_loss_weight=0.5,
                 affordance_edge_loss_weight=0.25,
                 affordance_target_loss_weight=0.25,
                 bev_safety_loss_coef=0.0,
                 bev_safety_unsafe_weight=4.0,
                 bev_safety_transition_weight=2.0,
                 **kwargs
                 ):
        self.use_amp = use_amp
        self.amp_loader_type = amp_loader_type
        self.num_amp_frames = num_amp_frames
        self.use_depth = use_depth
        self.seg_loss_coef = seg_loss_coef
        self.affordance_loss_coef = affordance_loss_coef
        self.affordance_safety_loss_weight = affordance_safety_loss_weight
        self.affordance_edge_loss_weight = affordance_edge_loss_weight
        self.affordance_target_loss_weight = affordance_target_loss_weight
        self.bev_safety_loss_coef = bev_safety_loss_coef
        self.bev_safety_unsafe_weight = bev_safety_unsafe_weight
        self.bev_safety_transition_weight = bev_safety_transition_weight
        self.device = device
        if default_pos is not None: 
            self.default_pos = torch.tensor(default_pos, device=self.device)

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.disc_learning_rate = disc_learning_rate
        self.policy_learning_rate = policy_learning_rate

        # Discriminator components
        if self.use_amp:
            print("**************** train with AMP ****************") 
            self.discriminator = discriminator
            self.discriminator.to(self.device)
            self.amp_transition = RolloutStorage.Transition()
            self.amp_storage = ReplayBufferMulti(
                discriminator.state_dim, amp_replay_buffer_size, self.num_amp_frames, device)
            self.amp_data = amp_data
            self.amp_normalizer = amp_normalizer
            self.optimizer_disc = optim.AdamW(self.discriminator.parameters(), lr=self.disc_learning_rate, weight_decay=1e-2)
        else:
            self.discriminator = None
            self.amp_normalizer = None
            print("**************** train without AMP ****************") 

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=self.policy_learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_len, history_dim, depth_shape=None, depth_buffer_len=None, foot_affordance_shape=None, bev_safety_shape=None):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, history_len, history_dim,
            depth_shape, depth_buffer_len, store_gt_safety_heatmap=self.seg_loss_coef > 0,
            foot_affordance_shape=foot_affordance_shape, bev_safety_shape=bev_safety_shape)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, history):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        if isinstance(obs, tuple):
            aug_obs, depth_image, aug_critic_obs = obs[0].detach(), obs[1].detach(), critic_obs.detach()
            self.transition.actions = self.actor_critic.act(aug_obs, history, depth_image).detach()
            self.transition.observations = obs[0]
            self.transition.depth_image = obs[1]
        else:
            aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
            self.transition.actions = self.actor_critic.act(aug_obs, history).detach()
            self.transition.observations = obs
        
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs, history=history).detach()
        if len(self.transition.actions.shape) > 2:
            self.transition.actions = self.transition.actions.reshape(self.transition.actions.shape[0], -1)
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.history = history
        self.transition.critic_observations = critic_obs
        return self.transition.actions
        
    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs, amp_obs_frames=None, **kwargs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        self.transition.next_critic_observations = next_critic_obs
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        if amp_obs_frames is not None:
            self.amp_storage.insert(amp_obs_frames)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, history):
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs, history).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc
    
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits, device=self.device))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits, device=self.device))
        return loss

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_agent_acc = 0
        mean_demo_acc = 0
        mean_affordance_loss = 0
        mean_bev_safety_loss = 0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        if self.use_amp:
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.num_learning_epochs * self.num_mini_batches,
                self.storage.num_envs * self.storage.num_transitions_per_env //
                    self.num_mini_batches)
            if self.amp_loader_type == 'lafan_16dof_multi':
                amp_expert_generator = self.amp_data.feed_forward_generator_lafan_16dof_multi(
                    self.num_learning_epochs * self.num_mini_batches,
                    self.storage.num_envs * self.storage.num_transitions_per_env //
                        self.num_mini_batches)
            else:
                NotImplementedError()
                
            for sample_amp_policy, sample_amp_expert in zip(amp_policy_generator, amp_expert_generator):
                    
                expert_states = sample_amp_expert
                policy_states = sample_amp_policy
                # Discriminator loss.
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        expert_states = self.amp_normalizer.normalize_torch(expert_states.to(self.device), self.device)
                        policy_states = self.amp_normalizer.normalize_torch(policy_states, self.device)
                policy_d = self.discriminator(policy_states.flatten(1))
                expert_states = expert_states.to(self.device)
                expert_d = self.discriminator(expert_states.flatten(1))
                agent_acc, demo_acc = self._compute_disc_acc(policy_d, expert_d)
                # prediction loss
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                
                # grad penalty
                grad_pen_loss = self.discriminator.compute_grad_pen(expert_states, lambda_=5)
                
                # logit reg
                logit_weights = self.discriminator.get_disc_logit_weights()
                disc_logit_loss = torch.sum(torch.square(logit_weights))
                disc_logit_loss = 0.01 * disc_logit_loss

                # weight decay
                disc_weights = self.discriminator.get_disc_weights()
                disc_weights = torch.cat(disc_weights, dim=-1)
                disc_weight_decay = torch.sum(torch.square(disc_weights))
                disc_weight_decay = 0.0001 * disc_weight_decay

                disc_loss = amp_loss + grad_pen_loss + disc_logit_loss + disc_weight_decay
                self.optimizer_disc.zero_grad()
                disc_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.optimizer_disc.step()
                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_states.cpu().numpy())
                    self.amp_normalizer.update(expert_states.cpu().numpy())

                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
                mean_agent_acc += agent_acc.mean().item()
                mean_demo_acc += demo_acc.mean().item()
        
        for obs_batch, critic_obs_batch, actions_batch, next_obs_batch, next_critic_observations_batch, history_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, depth_image_batch, gt_safety_heatmap_batch, foot_affordance_labels_batch, bev_safety_labels_batch, *_ in generator:

            aug_obs_batch, history_batch = obs_batch.detach(), history_batch.detach()
            need_seg_masks = self.use_depth and gt_safety_heatmap_batch is not None and self.seg_loss_coef > 0
            need_affordance_loss = (self.use_depth and foot_affordance_labels_batch is not None and
                                    self.affordance_loss_coef > 0 and
                                    getattr(self.actor_critic, "enable_foot_affordance", False))
            need_bev_safety_loss = (self.use_depth and bev_safety_labels_batch is not None and
                                    self.bev_safety_loss_coef > 0 and
                                    getattr(self.actor_critic, "enable_bev_safety", False))
            if self.use_depth:
                aug_depth_image_batch = depth_image_batch.detach()
                self.actor_critic.act(aug_obs_batch, history_batch, aug_depth_image_batch, return_masks=need_seg_masks)
            else:
                self.actor_critic.act(obs_batch, history_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, history=history_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.policy_learning_rate = max(1e-5, self.policy_learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.policy_learning_rate = min(1e-2, self.policy_learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.policy_learning_rate
                
            # Bound loss
            soft_bound = 1.0
            mu_loss_high = torch.maximum(mu_batch - soft_bound,
                                        torch.tensor(0, device=self.device)) ** 2
            mu_loss_low = torch.minimum(mu_batch + soft_bound,
                                        torch.tensor(0, device=self.device)) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
            b_loss = b_loss.mean()

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            # Compute total loss.
            loss = (
                0 * b_loss +
                surrogate_loss +
                self.value_loss_coef * value_loss -
                self.entropy_coef * entropy_batch.mean())

            if self.use_depth and gt_safety_heatmap_batch is not None and self.seg_loss_coef > 0:
                if not hasattr(self, '_seg_loss_printed'):
                    self._seg_loss_printed = True
                    print(f"[SEG_DEBUG] seg_loss enabled: coef={self.seg_loss_coef}, use_depth={self.use_depth}")
                import torch.nn.functional as _F
                pred_masks = self.actor_critic._last_pred_masks
                pred_masks_flat = pred_masks.flatten(0, 1)
                gt_masks_flat = gt_safety_heatmap_batch.flatten(0, 1).to(pred_masks.device)
                seg_loss = _F.mse_loss(pred_masks_flat, gt_masks_flat, reduction='mean')
                if not hasattr(self, '_seg_raw_printed'):
                    self._seg_raw_printed = True
                    print(f"[SEG_RAW] mse_mean={seg_loss.item():.6f} shape={pred_masks_flat.shape}")
                # store only a tiny sample for tensorboard visualization; keeping the full mini-batch costs GBs.
                self.last_seg_loss = seg_loss.item()
                self._last_pred_heatmap = pred_masks[:1].detach().cpu()
                self._last_gt_heatmap = gt_safety_heatmap_batch[:1].detach().cpu()
                if not hasattr(self, '_seg_diag_printed'):
                    self._seg_diag_printed = True
                    print(f"[SEG_DIAG] pred_mean={pred_masks.mean().item():.4f} gt_mean={gt_safety_heatmap_batch.mean().item():.4f} seg_loss={seg_loss.item():.6f}")
                loss = loss + self.seg_loss_coef * seg_loss

            if need_bev_safety_loss:
                pred_bev = self.actor_critic._last_bev_safety_pred
                labels = bev_safety_labels_batch.to(pred_bev.device)
                target = labels[:, 0]
                valid = labels[:, 1]
                unsafe = (target < 0.5) & (valid > 0.5)
                transition = (target > 0.05) & (target < 0.95)
                weight = (1.0 +
                          self.bev_safety_unsafe_weight * unsafe.float() +
                          self.bev_safety_transition_weight * transition.float())
                weighted_valid = weight * valid
                bev_safety_loss = (torch.square(pred_bev - target) * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
                if unsafe.any():
                    unsafe_recall = ((pred_bev < 0.5) & unsafe).float().sum() / unsafe.float().sum().clamp_min(1.0)
                else:
                    unsafe_recall = torch.ones((), device=pred_bev.device)
                if not hasattr(self, '_bev_safety_loss_printed'):
                    self._bev_safety_loss_printed = True
                    print(f"[BEV_DEBUG] bev_safety_loss enabled: coef={self.bev_safety_loss_coef}, shape={pred_bev.shape}")
                self.last_bev_safety_loss = bev_safety_loss.item()
                self.last_bev_pred_mean = pred_bev.mean().item()
                self.last_bev_gt_mean = target.mean().item()
                self.last_bev_pred_std = pred_bev.std().item()
                self.last_bev_gt_std = target.std().item()
                self.last_bev_unsafe_recall = unsafe_recall.item()
                self._last_pred_bev_safety = pred_bev[:1].detach().cpu()
                self._last_gt_bev_safety = labels[:1].detach().cpu()
                loss = loss + self.bev_safety_loss_coef * bev_safety_loss
            else:
                bev_safety_loss = None

            if need_affordance_loss:
                import torch.nn.functional as _F
                pred_affordance = self.actor_critic._last_affordance_pred
                labels = foot_affordance_labels_batch.to(pred_affordance.device)
                target = labels[..., :3]
                target_mask = labels[..., 3]
                safety_loss = _F.mse_loss(pred_affordance[..., 0], target[..., 0], reduction='mean')
                if self.affordance_edge_loss_weight > 0:
                    edge_loss = _F.mse_loss(pred_affordance[..., 1], target[..., 1], reduction='mean')
                else:
                    edge_loss = torch.zeros((), device=pred_affordance.device)
                if self.affordance_target_loss_weight > 0:
                    target_sq = torch.square(pred_affordance[..., 2] - target[..., 2]) * target_mask
                    denom = target_mask.sum().clamp_min(1.0)
                    target_loss = target_sq.sum() / denom
                else:
                    target_loss = torch.zeros((), device=pred_affordance.device)
                total_affordance_weight = (
                    self.affordance_safety_loss_weight +
                    self.affordance_edge_loss_weight +
                    self.affordance_target_loss_weight)
                if total_affordance_weight > 0:
                    affordance_loss = (
                        self.affordance_safety_loss_weight * safety_loss +
                        self.affordance_edge_loss_weight * edge_loss +
                        self.affordance_target_loss_weight * target_loss) / total_affordance_weight
                else:
                    affordance_loss = torch.zeros((), device=pred_affordance.device)
                if not hasattr(self, '_affordance_loss_printed'):
                    self._affordance_loss_printed = True
                    print(f"[AFF_DEBUG] affordance_loss enabled: coef={self.affordance_loss_coef}, weights=({self.affordance_safety_loss_weight}, {self.affordance_edge_loss_weight}, {self.affordance_target_loss_weight}), shape={pred_affordance.shape}")
                self.last_affordance_loss = affordance_loss.item()
                self.last_affordance_safety_loss = safety_loss.item()
                self.last_affordance_edge_loss = edge_loss.item()
                self.last_affordance_target_loss = target_loss.item()
                self._last_pred_affordance = pred_affordance[:1].detach().cpu()
                self._last_gt_affordance = labels[:1].detach().cpu()
                loss = loss + self.affordance_loss_coef * affordance_loss
            else:
                affordance_loss = None

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

            if self.device != 'cuda:0':
                for param in self.optimizer.state.values():
                    if isinstance(param, torch.Tensor):
                        pass
                        # param.data = param.data.to(self.device)
                    elif isinstance(param, dict):
                        for k, v in param.items():
                            if isinstance(v, torch.Tensor):
                                if k == "step":
                                    param[k] = v.to('cpu')
            self.optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if affordance_loss is not None:
                mean_affordance_loss += affordance_loss.item()
            if bev_safety_loss is not None:
                mean_bev_safety_loss += bev_safety_loss.item()
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_agent_acc /= num_updates
        mean_demo_acc /= num_updates
        mean_affordance_loss /= num_updates
        mean_bev_safety_loss /= num_updates
        self.last_mean_affordance_loss = mean_affordance_loss
        self.last_mean_bev_safety_loss = mean_bev_safety_loss
        
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred,  \
                mean_agent_acc, mean_demo_acc
