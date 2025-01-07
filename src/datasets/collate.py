import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate и паддинг полей в dataset_items.
    """

    # 1) Извлекаем все спектрограммы
    spectrograms = [item["spectrogram"] for item in dataset_items]

    # 2) Извлекаем тексты
    texts_encoded = [item["text_encoded"] for item in dataset_items]

    # 3) Извлекаем *сырые* тексты
    texts = [item["text"] for item in dataset_items]

    # >>> Извлекаем аудиопути
    audio_paths = [item["audio_path"] for item in dataset_items]  # <<< добавлено для audio_path

    # 4) Паддим спектрограммы
    spectrograms_padded = pad_sequence(
        spectrograms, batch_first=True, padding_value=0.0
    )

    # 5) Паддим тексты
    texts_padded = pad_sequence(
        texts_encoded, batch_first=True, padding_value=0
    )

    # 6) Длины
    spectrogram_lengths = torch.tensor(
        [s.shape[0] for s in spectrograms], dtype=torch.long
    )
    text_lengths = torch.tensor(
        [t.shape[0] for t in texts_encoded], dtype=torch.long
    )

    # 7) Собираем batch
    batch = {
        "spectrogram": spectrograms_padded,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": texts_padded,
        "text_encoded_length": text_lengths,
        "text": texts,                  # сырые тексты
        "audio_path": audio_paths,      # <<< добавлено поле со списком путей
    }
    return batch
