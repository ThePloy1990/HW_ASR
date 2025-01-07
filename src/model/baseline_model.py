import torch
from torch import nn


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, n_tokens, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features (часто = 128).
            n_tokens (int): размер словаря (включая blank).
            fc_hidden (int): скрытый размер в полносвязном слое.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_tokens),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Args:
            spectrogram: (B, T, n_feats) => (B, T, 128)
            spectrogram_length: (B,) длины T
        Returns:
            {
                "log_probs": (B, T, n_tokens),
                "log_probs_length": (B,) same as spectrogram_length
            }
        """
        # УБИРАЕМ transpose(1,2)!
        # output = self.net(spectrogram.transpose(1, 2))
        # Вместо этого подаём [B, T, n_feats]
        output = self.net(spectrogram)  # => (B, T, fc_hidden)

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        We do not reduce the time dimension in MLP,
        so the lengths remain the same.
        """
        return input_lengths

    def __str__(self):
        all_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        s = super().__str__()
        s += f"\nAll parameters: {all_params}"
        s += f"\nTrainable parameters: {train_params}"
        return s

