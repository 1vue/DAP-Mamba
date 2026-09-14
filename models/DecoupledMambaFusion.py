import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F


class DecoupledMambaFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for decoupled channel splitting"
        self.d_temp = d_model // 2
        self.d_spat = d_model // 2

        self.mamba_fwd = Mamba(d_model=self.d_temp, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=self.d_temp, d_state=16, d_conv=4, expand=2)

        self.spatial_refiner = nn.Sequential(
            nn.Conv1d(self.d_spat, self.d_spat, kernel_size=3, padding=1, groups=self.d_spat),
            nn.BatchNorm1d(self.d_spat),
            nn.SiLU()
        )

        self.norm = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        print('Initializing decoupled Mamba')
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # x shape: [B, T, C]
        res = x
        x_norm = self.norm(x)

        # x_temp: [B, T, C/2], x_spat: [B, T, C/2]
        x_temp, x_spat = torch.split(x_norm, [self.d_temp, self.d_spat], dim=-1)


        out_fwd = self.mamba_fwd(x_temp)
        out_bwd = self.mamba_bwd(x_temp.flip(dims=[1])).flip(dims=[1])
        y_temp = out_fwd + out_bwd

        y_spat = x_spat.transpose(1, 2)
        y_spat = self.spatial_refiner(y_spat)
        y_spat = y_spat.transpose(1, 2)

        combined = torch.cat([y_temp, y_spat], dim=-1)
        out = self.out_proj(combined)

        return res + out
