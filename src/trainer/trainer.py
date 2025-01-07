from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        # 1) Переносим на девайс и применяем batch-трансформации
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        # 2) Выбираем список метрик: для train или inference
        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        # 3) Модельный forward
        outputs = self.model(**batch)
        batch.update(outputs)

        # 4) Функция лосса. Предположим, она возвращает dict
        #    с ключом "loss" (и при необходимости другими).
        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        # 5) Если тренировка — делаем backward, клип градиентов, шаг оптимизатора
        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always in batch["loss"]
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # 6) Логируем значения лоссов
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        # 7) Обновляем пользовательские метрики (CER, WER и т.д.)
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # Пример схемы логгирования
        if mode == "train":
            # логируем раз в self.log_step шагов
            self.log_spectrogram(**batch)
        else:
            # Для inference можно дополнительно логировать предсказания
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        """
        Пример логгирования спектрограммы первого элемента в батче.
        Использует plot_spectrogram и self.writer.add_image.
        """
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        examples_to_log=10,
        use_beam_search=False,
        beam_size=3,
        **batch
    ):
        """
        Логгирование предсказаний (argmax и/или beam search).
        Если у вас несколько вариантов декодирования (greedy vs beam),
        можно выводить оба для сравнения.

        Args:
            text (list[str]): список эталонных строк.
            log_probs (Tensor): [B, T, vocab_size] - выход модели.
            log_probs_length (Tensor): [B] - длины для log_probs.
            audio_path (list[str]): пути к аудиофайлам (для идентификации).
            examples_to_log (int): сколько примеров логгировать.
            use_beam_search (bool): хотим ли мы использовать beam search.
            beam_size (int): размер бима при поиске.
        """

        # 1) greedy decode (argmax)
        argmax_inds = log_probs.cpu().argmax(dim=-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.cpu().numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        beam_search_texts = []
        if use_beam_search:
            for i in range(log_probs.size(0)):
                lp = log_probs[i, : log_probs_length[i]].detach().cpu()
                beam_text = self._beam_search_stub(lp, beam_size)
                beam_search_texts.append(beam_text)
        else:
            # Если beam search не используется
            beam_search_texts = ["" for _ in range(log_probs.size(0))]

        # 3) Собираем данные для логгирования
        tuples = list(zip(argmax_texts, argmax_texts_raw, beam_search_texts, text, audio_path))

        # 4) Формируем табличку. Пример логгирования первых examples_to_log записей
        rows = {}
        for i, (pred_greedy, raw_pred, pred_beam, target, audio_p) in enumerate(tuples[:examples_to_log]):
            # нормализуем target
            target_norm = self.text_encoder.normalize_text(target)

            # считаем метрики для greedy
            wer_g = calc_wer(target_norm, pred_greedy) * 100
            cer_g = calc_cer(target_norm, pred_greedy) * 100

            # если есть beam search
            if pred_beam:
                wer_b = calc_wer(target_norm, pred_beam) * 100
                cer_b = calc_cer(target_norm, pred_beam) * 100
            else:
                wer_b = cer_b = None

            rows[Path(audio_p).name] = {
                "target": target_norm,
                "greedy_raw": raw_pred,
                "greedy_pred": pred_greedy,
                "greedy_WER%": wer_g,
                "greedy_CER%": cer_g,
                "beam_pred": pred_beam if pred_beam else "--",
                "beam_WER%": wer_b if wer_b is not None else "--",
                "beam_CER%": cer_b if cer_b is not None else "--",
            }

        # 5) Логгируем табличку в writer
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
