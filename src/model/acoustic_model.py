import torch
import torch.nn as nn
import torch.nn.functional as F

class AcousticModel(nn.Module):
    """
    Пример акустической модели на основе bidirectional LSTM.
    CTC-friendly: на выходе тензор [B, T, num_tokens].
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_tokens):
        """
        Args:
            input_dim (int): число признаков на каждый фрейм (чаще всего = 128).
            hidden_dim (int): скрытая размерность в LSTM.
            num_layers (int): сколько LSTM-слоёв.
            num_tokens (int): размер словаря (включая blank).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # bidirectional => выходим из LSTM размером hidden_dim*2
        self.fc = nn.Linear(hidden_dim * 2, num_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Args:
            spectrogram: тензор [B, T, input_dim], где T - число фреймов,
                         input_dim = 128 (число мел-фильтров).
            spectrogram_length: (B,) длины (по времени) без паддинга.
        Returns:
            dict с:
                "log_probs" (B, T, num_tokens)
                "log_probs_length" (B,) — то же, что spectrogram_length
        """
        # lstm_out: [B, T, hidden_dim*2]
        lstm_out, _ = self.lstm(spectrogram)
        logits = self.fc(lstm_out)  # [B, T, num_tokens]
        log_probs = F.log_softmax(logits, dim=-1)
        return {
            "log_probs": log_probs,
            "log_probs_length": spectrogram_length
        }

    def transform_input_lengths(self, input_lengths):
        """
        LSTM не уменьшает длину (stride=1).
        """
        return input_lengths

    def __str__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + f"\nAll parameters: {total}\nTrainable parameters: {trainable}\n"
