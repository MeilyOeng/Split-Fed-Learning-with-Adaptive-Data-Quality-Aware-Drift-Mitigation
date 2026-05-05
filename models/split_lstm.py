from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float = 0.0
    batch_first: bool = True
    bidirectional: bool = False
    output_size: int = 1
    learning_rate: float = 0.001


class _StackedSingleLayerLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, *, batch_first: bool, bidirectional: bool, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        current_input = input_size
        for layer_idx in range(num_layers):
            self.layers.append(
                nn.LSTM(
                    input_size=current_input,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=batch_first,
                    dropout=0.0,
                    bidirectional=bidirectional,
                )
            )
            if layer_idx < num_layers - 1 and dropout > 0.0:
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(nn.Identity())
            current_input = self.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, dropout in zip(self.layers, self.dropouts):
            out, _ = layer(out)
            out = dropout(out)
        return out


class SplitLSTMClient(nn.Module):
    def __init__(self, config: LSTMConfig, split_layer: int) -> None:
        super().__init__()
        if not (1 <= split_layer < config.num_layers):
            raise ValueError(f"split_layer must be in [1, num_layers-1], got {split_layer} for num_layers={config.num_layers}")
        self.config = config
        self.split_layer = split_layer
        self.backbone = _StackedSingleLayerLSTM(
            config.input_size,
            config.hidden_size,
            split_layer,
            batch_first=config.batch_first,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
        )

    @property
    def hidden_dim(self) -> int:
        return self.backbone.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SplitLSTMServer(nn.Module):
    def __init__(self, config: LSTMConfig, split_layer: int) -> None:
        super().__init__()
        if not (1 <= split_layer < config.num_layers):
            raise ValueError(f"split_layer must be in [1, num_layers-1], got {split_layer} for num_layers={config.num_layers}")
        self.config = config
        self.split_layer = split_layer
        hidden_dim = config.hidden_size * (2 if config.bidirectional else 1)
        self.backbone = _StackedSingleLayerLSTM(
            hidden_dim,
            config.hidden_size,
            config.num_layers - split_layer,
            batch_first=config.batch_first,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
        )
        self.fc = nn.Linear(hidden_dim, config.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        last_timestep = out[:, -1, :]
        return self.fc(last_timestep)

# class SplitLSTMServer(nn.Module):
#     def __init__(self, config: LSTMConfig, split_layer: int) -> None:
#         super().__init__()

#         if not (1 <= split_layer < config.num_layers):
#             raise ValueError(
#                 f"split_layer must be in [1, num_layers-1], got {split_layer}"
#             )

#         hidden_dim = config.hidden_size * (2 if config.bidirectional else 1)

#         self.backbone = _StackedSingleLayerLSTM(
#             input_size=hidden_dim,
#             hidden_size=config.hidden_size,
#             num_layers=config.num_layers - split_layer,
#             batch_first=config.batch_first,
#             bidirectional=config.bidirectional,
#             dropout=config.dropout,
#         )

#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(hidden_dim, config.output_size),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.backbone(x)          # server LSTM is really used
#         out = out[:, -1, :]             # last timestep
#         out = self.head(out)            # activation head
#         return out.squeeze(-1)

def build_split_lstm(config: LSTMConfig, split_layer: int) -> Tuple[SplitLSTMClient, SplitLSTMServer]:
    return SplitLSTMClient(config, split_layer), SplitLSTMServer(config, split_layer)


# from dataclasses import dataclass
# from typing import Tuple

# import torch
# import torch.nn as nn


# @dataclass
# class LSTMConfig:
#     input_size: int
#     hidden_size: int
#     num_layers: int
#     dropout: float = 0.0
#     batch_first: bool = True
#     bidirectional: bool = False
#     output_size: int = 1
#     learning_rate: float = 0.001


# class _StackedSingleLayerLSTM(nn.Module):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_size: int,
#         num_layers: int,
#         *,
#         batch_first: bool,
#         bidirectional: bool,
#         dropout: float,
#     ) -> None:
#         super().__init__()
#         self.hidden_dim = hidden_size * (2 if bidirectional else 1)
#         self.layers = nn.ModuleList()
#         self.dropouts = nn.ModuleList()

#         current_input = input_size
#         for layer_idx in range(num_layers):
#             self.layers.append(
#                 nn.LSTM(
#                     input_size=current_input,
#                     hidden_size=hidden_size,
#                     num_layers=1,
#                     batch_first=batch_first,
#                     dropout=0.0,
#                     bidirectional=bidirectional,
#                 )
#             )

#             if layer_idx < num_layers - 1 and dropout > 0.0:
#                 self.dropouts.append(nn.Dropout(dropout))
#             else:
#                 self.dropouts.append(nn.Identity())

#             current_input = self.hidden_dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = x
#         for layer, dropout in zip(self.layers, self.dropouts):
#             out, _ = layer(out)
#             out = dropout(out)
#         return out


# class SplitLSTMClient(nn.Module):
#     def __init__(self, config: LSTMConfig, split_layer: int) -> None:
#         super().__init__()

#         if not (1 <= split_layer < config.num_layers):
#             raise ValueError(
#                 f"split_layer must be in [1, num_layers-1], "
#                 f"got {split_layer} for num_layers={config.num_layers}"
#             )

#         self.config = config
#         self.split_layer = split_layer

#         self.backbone = _StackedSingleLayerLSTM(
#             input_size=config.input_size,
#             hidden_size=config.hidden_size,
#             num_layers=split_layer,
#             batch_first=config.batch_first,
#             bidirectional=config.bidirectional,
#             dropout=config.dropout,
#         )

#     @property
#     def hidden_dim(self) -> int:
#         return self.backbone.hidden_dim

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.backbone(x)


# class SplitLSTMServer(nn.Module):
#     def __init__(self, config: LSTMConfig, split_layer: int) -> None:
#         super().__init__()

#         if not (1 <= split_layer < config.num_layers):
#             raise ValueError(
#                 f"split_layer must be in [1, num_layers-1], "
#                 f"got {split_layer} for num_layers={config.num_layers}"
#             )

#         self.config = config
#         self.split_layer = split_layer

#         hidden_dim = config.hidden_size * (2 if config.bidirectional else 1)

#         self.backbone = _StackedSingleLayerLSTM(
#             input_size=hidden_dim,
#             hidden_size=config.hidden_size,
#             num_layers=config.num_layers - split_layer,
#             batch_first=config.batch_first,
#             bidirectional=config.bidirectional,
#             dropout=config.dropout,
#         )

#         self.head = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(config.dropout),
#             nn.Linear(hidden_dim, config.output_size),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.backbone(x)
#         out = out[:, -1, :]
#         out = self.head(out)

#         # Important:
#         # Do NOT use squeeze(-1) here.
#         # server.py reshapes targets to [batch, 1], so predictions must also be [batch, 1].
#         return out


# def build_split_lstm(
#     config: LSTMConfig,
#     split_layer: int,
# ) -> Tuple[SplitLSTMClient, SplitLSTMServer]:
#     return SplitLSTMClient(config, split_layer), SplitLSTMServer(config, split_layer)