import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
            self,
            model,
            config,
            device,
            dataloaders,
            text_encoder,
            save_path,
            metrics=None,
            batch_transforms=None,
            skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (Path): path to save model predictions and other
                information (can be None if you don't want to save).
            metrics (dict): dict with the definition of metrics for
                inference (metrics["inference"]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
                skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition
        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model weights from checkpoint
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition (e.g. test, valid, etc.)

        Returns:
            part_logs (dict): part_logs[partition] = logs (dict of metrics)
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        (optionally) save predictions to disk.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics (e.g. CER, WER).
            part (str): name of the partition (for save_path if used).

        Returns:
            batch (dict): updated batch with model outputs, if needed.
        """
        batch = self.move_batch_to_device(batch)
        # Трансформации на батч, если заданы
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        # Обновляем метрики
        if metrics is not None and "inference" in self.metrics:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            batch_size = batch["log_probs"].shape[0]
            current_id = batch_idx * batch_size

            for i in range(batch_size):
                # Получаем log_probs_i
                log_probs_i = batch["log_probs"][i].detach().cpu()
                length_i = batch["log_probs_length"][i].detach().cpu()
                # Декодируем
                argmax_inds = torch.argmax(log_probs_i[:length_i], dim=-1).numpy()
                predicted_text = self.text_encoder.ctc_decode(argmax_inds)

                reference_text = batch.get("text", [""] * batch_size)[i]

                output_id = current_id + i
                output = {
                    "ref_text": reference_text,
                    "pred_text": predicted_text,
                }

                # Сохраняем в файл
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Inference for a specific partition (e.g. 'test').
        We pass every batch through process_batch and gather metrics.

        Args:
            part (str): partition name.
            dataloader (DataLoader): Dataloader with data for partition.

        Returns:
            logs (dict): metrics results for this partition.
        """
        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
