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

import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, output_dim, output_activation=None, in_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.output_dim = output_dim
        activation = nn.ELU()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            activation,
            nn.Linear(128, output_dim),
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        x = self.conv1(images.unsqueeze(1))
        conv_feat = self.conv2(x)  # [B, 64, 5, 5]
        latent = self.output_activation(self.fc(conv_feat))
        return latent, conv_feat

class SegDecoder(nn.Module):
    def __init__(self, in_channels=64, out_h=64, out_w=64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Upsample(size=(out_h, out_w), mode='bilinear', align_corners=False),
            nn.Sigmoid(),
        )
    def forward(self, conv_feat):
        return self.decoder(conv_feat).squeeze(1)  # [B, H, W]


class FootAffordanceHead(nn.Module):
    def __init__(self, num_actor_obs, his_latent_dim, depth_dim=128, token_dim=32, hidden_dim=64,
                 candidate_x=(0.10, 0.25, 0.40, 0.55, 0.70),
                 candidate_y_offsets=(-0.08, 0.0, 0.08),
                 foot_lateral_offset=0.11):
        super().__init__()
        self.token_dim = token_dim
        activation = nn.ELU()

        self.body_encoder = nn.Sequential(
            nn.Linear(num_actor_obs + his_latent_dim, 128),
            activation,
            nn.Linear(128, 64),
            activation,
        )
        self.scorer = nn.Sequential(
            nn.Linear(depth_dim + 64 + 3, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
        )
        self.pred_head = nn.Linear(hidden_dim, 3)
        self.token_head = nn.Sequential(nn.Linear(hidden_dim, token_dim), activation)

        candidate_x = [float(x) for x in candidate_x]
        candidate_y_offsets = [float(y) for y in candidate_y_offsets]
        candidates = []
        for side in (1.0, -1.0):
            side_candidates = []
            for x in candidate_x:
                for y_off in candidate_y_offsets:
                    side_candidates.append([x, side * float(foot_lateral_offset) + y_off, side])
            candidates.append(side_candidates)
        self.register_buffer("candidate_geometry", torch.tensor(candidates, dtype=torch.float32))

    def forward(self, observations: torch.Tensor, his_feature: torch.Tensor, depth_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        body_feature = self.body_encoder(torch.cat((observations, his_feature), dim=-1))
        B = observations.shape[0]
        candidate_geometry = self.candidate_geometry.to(observations.device)
        sides = candidate_geometry.shape[0]
        num_candidates = candidate_geometry.shape[1]

        depth_expand = depth_feature[:, None, None, :].expand(B, sides, num_candidates, depth_feature.shape[-1])
        body_expand = body_feature[:, None, None, :].expand(B, sides, num_candidates, body_feature.shape[-1])
        geom_expand = candidate_geometry[None, :, :, :].expand(B, sides, num_candidates, candidate_geometry.shape[-1])
        features = torch.cat((depth_expand, body_expand, geom_expand), dim=-1)

        hidden = self.scorer(features.reshape(B * sides * num_candidates, -1))
        pred = torch.sigmoid(self.pred_head(hidden)).reshape(B, sides, num_candidates, 3)
        token_values = self.token_head(hidden).reshape(B, sides, num_candidates, self.token_dim)

        # Safety-only pooling keeps the token focused on deployable local stepability.
        weights = torch.softmax(pred[..., 0], dim=-1)
        tokens = torch.sum(weights.unsqueeze(-1) * token_values, dim=2)
        return tokens.reshape(B, sides * self.token_dim), pred


class StackDepthEncoder(nn.Module):
    def __init__(self, base_backbone: DepthOnlyFCBackbone58x87, buffer_len=None,
                 temporal_channels=64, output_dim=128) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.output_dim = output_dim
        self.buffer_len = buffer_len

        # Temporal conv over time axis; pooling makes it length-agnostic.
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=base_backbone.output_dim, out_channels=base_backbone.output_dim,
                      kernel_size=3, padding=1),
            activation,
            nn.Conv1d(in_channels=base_backbone.output_dim, out_channels=temporal_channels,
                      kernel_size=3, padding=1),
            activation,
        )
        self.mlp = nn.Sequential(nn.Linear(temporal_channels * 2, output_dim), activation)

    def forward(self, depth_image):
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(depth_image.flatten(0, 1))  # [batch_size * num, 128]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 128]
        depth_latent = depth_latent.transpose(1, 2)  # [batch_size, 128, num]
        depth_latent = self.temporal(depth_latent)
        depth_mean = depth_latent.mean(dim=-1)
        depth_max = torch.max(depth_latent, dim=-1).values
        depth_latent = torch.cat([depth_mean, depth_max], dim=-1)
        depth_latent = self.mlp(depth_latent)
        return depth_latent

class StackDepthEncoderGRU(nn.Module):
    def __init__(self, base_backbone: DepthOnlyFCBackbone58x87,
                 hidden_dim=128, num_layers=1, output_dim=128) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=base_backbone.output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + base_backbone.output_dim, output_dim),
            activation,
        )
        self.seg_decoder = SegDecoder()

    def forward(self, depth_image: torch.Tensor, return_masks: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # depth_image shape: [batch_size, num, 64, 64]
        B, T = depth_image.shape[:2]
        latents, conv_feats = [], []
        for t in range(T):
            lat, cf = self.base_backbone(depth_image[:, t])
            latents.append(lat)
            if return_masks:
                conv_feats.append(cf)
        depth_latent = torch.stack(latents, dim=1)  # [B, T, 128]

        _, hidden = self.gru(depth_latent)
        history_feature = hidden[-1]
        latest_feature = depth_latent[:, -1]
        depth_feature = self.mlp(torch.cat([history_feature, latest_feature], dim=-1))

        if not return_masks:
            return depth_feature, None

        conv_feats = torch.stack(conv_feats, dim=1)  # [B, T, 64, 5, 5]
        pred_masks = self.seg_decoder(conv_feats.flatten(0, 1))  # [B*T, H, W]
        H_m, W_m = pred_masks.shape[1], pred_masks.shape[2]
        pred_masks = pred_masks.reshape(B, T, H_m, W_m)  # [B, T, H, W]
        return depth_feature, pred_masks

class ActorCriticDepth(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        his_encoder_dims=[1024, 512, 128],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        his_latent_dim = 64 + 3,
                        history_dim = 570,
                        depth_buffer_len = 2,
                        depth_encoder_type = "gru",
                        depth_temporal_channels = 64,
                        depth_gru_hidden_dim = 128,
                        depth_gru_num_layers = 1,
                        enable_foot_affordance = False,
                        affordance_token_dim = 32,
                        affordance_hidden_dim = 64,
                        affordance_candidate_x = (0.10, 0.25, 0.40, 0.55, 0.70),
                        affordance_candidate_y_offsets = (-0.08, 0.0, 0.08),
                        affordance_foot_lateral_offset = 0.11,
                        activation='elu',
                        init_noise_std=1.0,
                        max_grad_norm=10.0,
                        **kwargs):
        if kwargs:
            print("ActorCriticEst.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDepth, self).__init__()
        activation = get_activation(activation)

        self.his_latent_dim = his_latent_dim
        self.max_grad_norm = max_grad_norm
        self.enable_foot_affordance = enable_foot_affordance
        self.affordance_token_dim = affordance_token_dim

        # depth encoder
        depth_backbone = DepthOnlyFCBackbone58x87(output_dim=128, output_activation=activation)
        if depth_encoder_type == "gru":
            self.depth_encoder = StackDepthEncoderGRU(
                depth_backbone,
                hidden_dim=depth_gru_hidden_dim,
                num_layers=depth_gru_num_layers,
                output_dim=depth_backbone.output_dim,
            )
        elif depth_encoder_type == "conv_pool":
            self.depth_encoder = StackDepthEncoder(
                depth_backbone,
                buffer_len=depth_buffer_len,
                temporal_channels=depth_temporal_channels,
                output_dim=depth_backbone.output_dim,
            )
        else:
            raise ValueError(f"Unsupported depth_encoder_type: {depth_encoder_type}")

        mlp_input_dim_a = num_actor_obs + his_latent_dim + depth_backbone.output_dim
        if self.enable_foot_affordance:
            self.foot_affordance_head = FootAffordanceHead(
                num_actor_obs=num_actor_obs,
                his_latent_dim=his_latent_dim,
                depth_dim=depth_backbone.output_dim,
                token_dim=affordance_token_dim,
                hidden_dim=affordance_hidden_dim,
                candidate_x=affordance_candidate_x,
                candidate_y_offsets=affordance_candidate_y_offsets,
                foot_lateral_offset=affordance_foot_lateral_offset,
            )
            mlp_input_dim_a += 2 * affordance_token_dim
        else:
            self.foot_affordance_head = None
        mlp_input_dim_c = num_critic_obs + his_latent_dim
        
        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, his_encoder_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(his_encoder_dims)):
            if l == len(his_encoder_dims) - 1:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_latent_dim))
            else:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_encoder_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def _build_actor_input(self, observations, his_feature, depth_feature):
        if self.enable_foot_affordance:
            affordance_tokens, self._last_affordance_pred = self.foot_affordance_head(observations, his_feature, depth_feature)
            return torch.cat((observations, his_feature, depth_feature, affordance_tokens), dim=-1)
        self._last_affordance_pred = None
        return torch.cat((observations, his_feature, depth_feature), dim=-1)

    def act(self, observations, history, depth, return_masks: bool = False, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        depth_feature, self._last_pred_masks = self.depth_encoder(depth, return_masks=return_masks)
        actor_input = self._build_actor_input(observations, his_feature, depth_feature)
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, depth, return_masks: bool = False, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        depth_feature, self._last_pred_masks = self.depth_encoder(depth, return_masks=return_masks)
        actor_input = self._build_actor_input(observations, his_feature, depth_feature)
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def evaluate(self, critic_observations, history, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        actor_input = torch.cat((critic_observations, his_feature), dim=-1)
        value = self.critic(actor_input)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
