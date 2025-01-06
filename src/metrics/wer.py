from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    """
    Calculate WER (Word Error Rate) по argmax предсказаниям.

    text_encoder: объект, имеющий методы:
        - .normalize_text(...)  # Для нормализации true-текста
        - .ctc_decode(...)      # Для расшифровки индексов модели
    """

    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
            self,
            log_probs: Tensor,
            log_probs_length: Tensor,
            text: List[str],
            **kwargs
    ) -> float:
        """
        Args:
            log_probs (Tensor): [B, T, vocab_size], логи вероятностей (после log_softmax).
            log_probs_length (Tensor): [B], длины (T) каждого батча (без паддинга).
            text (List[str]): список целевых строк.

        Returns:
            avg_wer (float): средний WER по батчу.
        """
        wers = []

        # Берем аргмакс по последней размерности => [B, T]
        predictions = torch.argmax(log_probs.detach().cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        # Для каждого элемента в батче декодируем и считаем WER
        for pred_inds, length, target_text in zip(predictions, lengths, text):
            # Нормализуем эталонный текст (lowercase, удаляем лишние символы, и т.п.)
            target_text = self.text_encoder.normalize_text(target_text)

            # Декодируем логиты модели:
            # pred_inds[:length] - учитываем длину
            pred_text = self.text_encoder.ctc_decode(pred_inds[:length])

            # Считаем WER
            wers.append(calc_wer(target_text, pred_text))

        # Средний WER по батчу
        return sum(wers) / len(wers) if len(wers) > 0 else 0.0
