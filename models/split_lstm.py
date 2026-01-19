# models/split_lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple

@dataclass
class LSTMConfig:
    """
    Configuration for the split LSTM model.

    All fields are meant to be filled from external config
    (e.g. Helm values -> env vars -> Python).

    Attributes
    ----------
    input_size : int
        Number of input features per time step (11 in your case).
    hidden_size : int
        Hidden size of each LSTM layer.
    num_layers : int
        Total number of LSTM layers (client + server).
    output_size : int
        Size of the prediction head output (1 in your case).
    dropout : float
        Dropout between LSTM layers (0.0 to disable).
    bidirectional : bool
        Whether to use bidirectional LSTMs.
    """
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float = 0.0
    batch_first: bool = True
    bidirectional: bool = False
    output_size: int = 1
    learning_rate: float = 0.001

class SplitLSTMClient(nn.Module):
    """
    Client-side part of a split LSTM model.

    This module holds the first `split_layer` LSTM layers.
    It takes input of shape [batch, seq_len, input_size] and
    returns an intermediate representation of shape
    [batch, seq_len, hidden_dim], where hidden_dim depends on
    bidirectionality.
    """
    def __init__(self, config: LSTMConfig, split_layer: int) -> None:
        super().__init__()

        if not (1 <= split_layer < config.num_layers):
            raise ValueError(
                f"split_layer must be in [1, num_layers-1], "
                f"got split_layer={split_layer}, num_layers={config.num_layers}"
            )

        self.config = config
        self.split_layer = split_layer

        num_directions = 2 if config.bidirectional else 1
        hidden_dim = config.hidden_size * num_directions

        layers = []

        # First LSTM layer goes from input_size -> hidden_dim
        layers.append(
            nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
            )
        )

        # Remaining client layers (if any) go from hidden_dim -> hidden_dim
        for _ in range(1, split_layer):
            layers.append(
                nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=config.dropout,
                    bidirectional=config.bidirectional,
                )
            )

        self.layers = nn.ModuleList(layers)
    @property
    def hidden_dim(self) -> int:    
        """
        Hidden dimension of the LSTM layers.
        """
        return self.config.hidden_size * (2 if self.config.bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM layers.
        """
        for layer in self.layers:
            x, _ = layer(x)  # LSTM returns (output, (hidden, cell)), we only need output
        return x


class SplitLSTMServer(nn.Module):
    """
    Server-side part of a split LSTM model.

    This module holds the remaining `num_layers - split_layer` LSTM layers.
    It takes input of shape [batch, seq_len, hidden_dim] and
    returns a final representation of shape [batch, output_size].
    """
    def __init__(self, config: LSTMConfig, split_layer: int) -> None:
        super().__init__()

        if not (1 <= split_layer < config.num_layers):
            raise ValueError(
                f"split_layer must be in [1, num_layers-1], "
                f"got split_layer={split_layer}, num_layers={config.num_layers}"
            )

        self.config = config
        self.split_layer = split_layer

        num_directions = 2 if config.bidirectional else 1
        hidden_dim = config.hidden_size * num_directions

        layers = []

        # All server-side LSTM layers operate on hidden_dim -> hidden_dim
        for _ in range(split_layer, config.num_layers):
            layers.append(
                nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=config.hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=config.dropout,
                    bidirectional=config.bidirectional,
                )
            )

        self.layers = nn.ModuleList(layers)

        # Final prediction head: use last time step's features
        self.fc = nn.Linear(hidden_dim, config.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, seq_len, hidden_dim], i.e. the output
            of the client model.

        Returns
        -------
        torch.Tensor
            Shape [batch, output_size].
        """
        out = x
        for lstm in self.layers:
            out, _ = lstm(out)

        # Use the last time step for prediction
        # out: [batch, seq_len, hidden_dim]
        last_timestep = out[:, -1, :]  # [batch, hidden_dim]
        logits = self.fc(last_timestep)
        return logits


def build_split_lstm(
    config: LSTMConfig, split_layer: int
) -> Tuple[SplitLSTMClient, SplitLSTMServer]:
    """
    Convenience factory to build client and server models from a shared config.

    Parameters
    ----------
    config : LSTMConfig
        Model hyperparameters.
    split_layer : int
        Layer index at which to split the model. Layers
        [0, ..., split_layer-1] live on the client, and
        [split_layer, ..., num_layers-1] live on the server.

    Returns
    -------
    (client_model, server_model) : (SplitLSTMClient, SplitLSTMServer)
    """
    client = SplitLSTMClient(config, split_layer)
    server = SplitLSTMServer(config, split_layer)
    return client, server
