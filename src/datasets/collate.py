import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate и паддинг полей в dataset_items.
    Превращает список отдельных примеров (sample) из датасета
    в один батч.

    dataset_items[i] обычно содержит поля:
      {
        "spectrogram": Tensor[time_i, freq] или [freq, time_i],
        "text_encoded": Tensor[text_len_i],
        ...
      }

    Returns:
        batch (dict[Tensor]): словарь с собранным батчем:
          {
            "spectrogram": Tensor[batch_size, max_time, freq],
            "spectrogram_length": Tensor[batch_size],
            "text_encoded": Tensor[batch_size, max_text_len],
            "text_encoded_length": Tensor[batch_size]
          }
    """
    # Извлекаем спектрограммы и их длины
    spectrograms = [item["spectrogram"] for item in dataset_items]
    # Извлекаем закодированный текст и их длины
    texts_encoded = [item["text_encoded"] for item in dataset_items]

    # Паддим спектрограммы
    # Если у вас [time, freq], то используем pad_sequence(..., batch_first=True)
    spectrograms_padded = pad_sequence(
        spectrograms,
        batch_first=True,
        padding_value=0.0
    )

    # Паддим тексты
    texts_padded = pad_sequence(
        texts_encoded,
        batch_first=True,
        padding_value=0
    )

    # Запоминаем реальные длины до паддинга
    spectrogram_lengths = torch.tensor(
        [s.shape[0] for s in spectrograms],
        dtype=torch.long
    )
    text_lengths = torch.tensor(
        [t.shape[0] for t in texts_encoded],
        dtype=torch.long
    )

    # Собираем батч в словарь
    batch = {
        "spectrogram": spectrograms_padded,  # [B, T, F]
        "spectrogram_length": spectrogram_lengths,  # [B]
        "text_encoded": texts_padded,  # [B, T_text]
        "text_encoded_length": text_lengths,  # [B]
    }

    return batch
