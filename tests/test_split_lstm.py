import torch
from models.split_lstm import SplitLSTMClient, SplitLSTMServer, LSTMConfig, build_split_lstm

def main():

    config = LSTMConfig(
        input_size=11,     # number of features per timestep
        hidden_size=32,
        num_layers=9,      # total LSTM layers
        output_size=1,
        dropout=0.0,
        bidirectional=False,
    )

    split_layer = 7  # client: layers 0–1, server: layers 2–3

    client_model, server_model = build_split_lstm(config, split_layer)

    print("Client model:")
    print(client_model)
    print("\nServer model:")
    print(server_model)


    batch_size = 8
    seq_len = 24

    x = torch.randn(batch_size, seq_len, config.input_size)
    y = torch.randn(batch_size, config.output_size)

    activations = client_model(x)
    activations.retain_grad()  # IMPORTANT for split learning

    preds = server_model(activations)

    print("\nOutput shape:", preds.shape)


    loss = torch.nn.functional.mse_loss(preds, y)
    loss.backward()

    assert activations.grad is not None, "No gradient flowed back to client!"
    print("\nBackward pass successful.")
    print("Gradient shape at split:", activations.grad.shape)

if __name__ == "__main__":
    main()
