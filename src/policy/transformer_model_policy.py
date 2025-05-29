# Created at 2025

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from src.config import Config
from src.rl_data_record import RLDataRecord


class TransformerBlockPolicy(nn.Module):
    """This module implement the Transfer block architecture."""

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None
        self.config = config

        self.q = nn.Parameter(
            torch.zeros(size=(config.embedding, config.qkv_projection)).uniform_(0, 1.0)
        )  # Emb x qkv_projection
        self.k = nn.Parameter(
            torch.zeros(size=(config.embedding, config.qkv_projection)).uniform_(0, 1.0)
        )  # Emb x qkv_projection
        self.v = nn.Parameter(
            torch.zeros(size=(config.embedding, config.qkv_projection)).uniform_(0, 1.0)
        )  # Emb x qkv_projection

        self.mlp = nn.Sequential(
            nn.LayerNorm(config.qkv_projection),
            nn.Linear(config.qkv_projection, config.transformer_block_layer1),
            nn.ReLU(inplace=True),
            nn.LayerNorm(config.transformer_block_layer1),
            nn.Linear(config.transformer_block_layer1, config.embedding),
        )
        self._init_parameters()

    def _init_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, fov_emb: torch.Tensor):
        assert fov_emb is not None  # B, T, Emb

        q = fov_emb @ self.q  # B x T x qkv_projection
        k = fov_emb @ self.k  # B x T x qkv_projection
        v = fov_emb @ self.v  # B x T x qkv_projection

        B, T, qkv_projection = q.size()
        qh = q.reshape((B, T, self.config.multi_heads, -1))  # B x T x H x Emb_H
        qh = qh.permute(0, 2, 1, 3)  # B x H x T x Emb_H

        kh = k.reshape((B, T, self.config.multi_heads, -1))  # B x T x H x Emb_H
        kh = kh.permute(0, 2, 3, 1)  # B x H x Emb_H x T
        att = (
            qh
            @ kh
            / torch.sqrt(torch.tensor(T, dtype=torch.float, device=self.config.device))
        )  # B x H x T x T
        att = att.softmax(dim=-1)  # image no causal attention.

        vh = v.reshape((B, T, self.config.multi_heads, -1))  # B x T x H x Emb_H
        vh = vh.permute(0, 2, 1, 3)  # B x H x T x Emb_H

        vatt = att @ vh  # B x H x T x Emb_H
        vatt = vatt.permute((0, 2, 1, 3))  # B x T x H x Emb_H
        vatt = vatt.reshape((B, T, -1))  # B x T x qkv_projection

        feat = self.mlp(vatt)  # B x T x Emb
        return feat  # B x T x Emb


class TransformerPolicy(nn.Module):
    """This module implement the policy with Transfer architecture."""

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert config is not None
        self.config = config

        # fov image to embedding
        # B x T x Emb
        self.img_to_emb = nn.Conv2d(
            in_channels=config.img_in_channels,
            out_channels=config.embedding,
            kernel_size=config.img_kernel_size,
            stride=config.img_kernel_size,
            bias=True,
        )
        self.img_layer_norm = nn.LayerNorm(config.img_tokens * config.embedding)

        self.transformer_blocks = nn.Sequential(
            OrderedDict(
                [
                    (f"transformer_block_{i}", TransformerBlockPolicy(config=config))
                    for i in range(config.transformer_blocks)
                ]
            )
        )
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    (
                        "trunk1",
                        nn.Linear(
                            config.img_tokens * config.embedding,
                            2 * config.trunk_features,
                        ),
                    ),
                    ("relu", nn.ReLU()),
                    (
                        "trunk2",
                        nn.Linear(
                            2 * config.trunk_features,
                            config.trunk_features,
                        ),
                    ),
                    ("relu", nn.ReLU()),
                ]
            )
        )
        self.head = nn.Linear(config.trunk_features + 4, len(config.possible_actions))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.img_to_emb.weight)

        nn.init.kaiming_uniform_(self.head.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight = nn.init.kaiming_uniform_(layer.weight)

    def _prepare_feature(self, batch_rl_data_record: RLDataRecord) -> torch.Tensor:
        """Prepare the train/eval feature data for the model"""

        assert batch_rl_data_record is not None
        batch_fov = batch_rl_data_record.fov2d()
        batch_cur_position = batch_rl_data_record.batch_agent_current_pos
        batch_target_position = batch_rl_data_record.batch_agent_target_pos

        return batch_fov, batch_cur_position, batch_target_position

    def forward(self, batch_rl_data_record: RLDataRecord):
        batch_fov, batch_cur_position, batch_target_position = self._prepare_feature(
            batch_rl_data_record=batch_rl_data_record
        )
        assert batch_fov is not None
        assert batch_cur_position is not None
        assert batch_target_position is not None
        assert (
            batch_fov.size(0)
            == batch_cur_position.size(0)
            == batch_target_position.size(0)
        )

        assert batch_fov is not None
        B, C, H, W = batch_fov.size()

        fov_emb = self.img_to_emb(batch_fov)  # B x T x H x W
        fov_emb = fov_emb.reshape(B, self.config.embedding, -1)  # B x Emb x T
        fov_emb = fov_emb.permute(0, 2, 1)  # B x T x Emb

        feature = self.transformer_blocks(fov_emb)  # B x T x Emb
        feature = feature.reshape((B, -1))  # B x T

        feature = self.img_layer_norm(feature)

        feature = self.mlp(feature)  # B x Emb

        feature = torch.concat(
            [batch_cur_position, batch_target_position, feature], dim=-1
        )  # B x (Emb+4)
        logits = self.head(feature)
        return logits  # B x 9 (possible_actions)
