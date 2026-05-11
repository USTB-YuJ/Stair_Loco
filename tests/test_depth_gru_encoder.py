import sys
from pathlib import Path

import torch


RSL_RL_ROOT = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.insert(0, str(RSL_RL_ROOT))

from rsl_rl.modules.actor_critic_depth import ActorCriticDepth  # noqa: E402


def test_actor_critic_depth_gru_encoder_forward_shape():
    model = ActorCriticDepth(
        num_actor_obs=39,
        num_critic_obs=39,
        num_actions=10,
        history_dim=10 * 39,
        depth_buffer_len=10,
        depth_encoder_type="gru",
        depth_gru_hidden_dim=128,
        depth_gru_num_layers=1,
    )

    obs = torch.zeros(2, 39)
    history = torch.zeros(2, 10, 39)
    depth = torch.zeros(2, 10, 64, 64)

    actions = model.act_inference(obs, history, depth)

    assert model.depth_encoder.__class__.__name__ == "StackDepthEncoderGRU"
    assert actions.shape == (2, 10)
    assert model.depth_encoder.output_dim == 128
