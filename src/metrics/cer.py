from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    """
    Calculate CER (Character Error Rate) по argmax предсказаниям.

    - text_encoder: объект, у которого есть .normalize_text(...)
      и .ctc_decode(...).
    - log_probs: (B, T, vocab_size) — выход модели до CTC (log_softmax).
      Мы будем брать argmax по последней оси.
    - log_probs_length: (B,) — длины валидной части log_probs по оси T.
    - text: list[str] целевые (ground-truth) строки.

    CER рассчитывается как edit_distance(ref, hyp) / len(ref).
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
            log_probs (Tensor): [B, T, vocab_size] - логи вероятностей (pos CTC),
                где B - batch, T - time, vocab_size - размер словаря (включая blank).
            log_probs_length (Tensor): [B] - длины по времени (T).
            text (list[str]): список целевых строк для CER.

        Returns:
            avg_cer (float): средний CER по батчу (от 0 до 1).
        """
        cers = []

        # Аргмакс по последней оси => [B, T]
        predictions = torch.argmax(log_probs.detach().cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()

        # Считаем CER для каждого элемента в батче
        for pred_inds, length, target_text in zip(predictions, lengths, text):
            # Нормализуем true-текст
            target_text = self.text_encoder.normalize_text(target_text)
            # Учитываем, что valid T = length, поэтому берем pred_inds[:length]
            pred_text = self.text_encoder.ctc_decode(pred_inds[:length])

            # Считаем
            cers.append(calc_cer(target_text, pred_text))

        return sum(cers) / len(cers) if len(cers) > 0 else 0.0
