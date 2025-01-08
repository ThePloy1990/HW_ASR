import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    # 1) Рандом сид
    set_random_seed(config.trainer.seed)

    # 2) Логгер + writer
    project_config = OmegaConf.to_container(config, resolve=True)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    # 3) Девайс
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    logger.info(f"Using device: {device}")

    # 4) Инициализируем text_encoder
    text_encoder = instantiate(config.text_encoder)
    logger.info(f"TextEncoder vocab size: {len(text_encoder)}")

    # 5) Создаём даталоадеры + batch_transforms
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # 6) Создаём модель (акустическую)
    #    Подаём num_tokens=len(text_encoder)
    model = instantiate(config.model, num_tokens=len(text_encoder)).to(device)
    logger.info(model)

    # 7) Создаём лосс
    loss_function = instantiate(config.loss_function).to(device)

    # 8) Метрики
    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(
                instantiate(metric_config, text_encoder=text_encoder)
            )

    # 9) Оптимизатор и LR-шедулер
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # 10) Подключаем LM, если use_lm=True
    huggingface_lm = None
    if "lm" in config and config.lm.get("use_lm", False):
        huggingface_lm = instantiate(config.lm)  # создаём объект LM
        huggingface_lm.to(device)
        logger.info(f"LM from HuggingFace loaded: {config.lm.model_name}")

    # 11) Включаем Beam Search, если trainer.use_beam_search=True
    if config.trainer.get("use_beam_search", False):
        from src.text_encoder.beam_search_decoder import BeamSearchDecoder

        beam_size = config.trainer.get("beam_size", 5)
        lm_alpha = config.trainer.get("lm_alpha", 0.5)

        decoder = BeamSearchDecoder(
            text_encoder=text_encoder,
            beam_size=beam_size,
            cutoff_prob=1.0,
            cutoff_top_n=40,
            lm=huggingface_lm,
            lm_alpha=lm_alpha,
        )

        # Запишем decoder в text_encoder, чтобы потом в trainer можно было дергать:
        # text_encoder.beam_search_decode(log_probs, log_probs_length, beam_size=...)
        text_encoder.beam_search_decoder = decoder
        logger.info("Beam Search Decoder is created. LM={}".format(
            "YES" if huggingface_lm else "NO"
        ))
    else:
        logger.info("No beam search is used (greedy only).")

    # 12) Собираем Trainer
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    # 13) Запускаем обучение
    trainer.train()


if __name__ == "__main__":
    main()
