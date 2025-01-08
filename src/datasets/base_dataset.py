import logging
import random

import numpy as np
import torchaudio
from torch.utils.data import Dataset

from src.text_encoder import CTCTextEncoder

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        text_encoder=None,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            text_encoder (CTCTextEncoder): text encoder.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.text_encoder = text_encoder
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        audio_path = data_dict["path"]

        # 1) Загрузим аудиофайл
        audio = self.load_audio(audio_path)

        # 2) Применим аугментации к аудиосигналу (если есть)
        if (
            self.instance_transforms is not None
            and "audio" in self.instance_transforms
        ):
            audio = self.instance_transforms["audio"](audio)

        # 3) Генерируем спектрограмму (MelSpectrogram или Spectrogram)
        spectrogram = self.get_spectrogram(audio)
        # !!! Здесь уже логика приведения к [time, freq] внутри get_spectrogram

        # 4) Кодируем текст
        text = data_dict["text"]
        text_encoded = self.text_encoder.encode(text)

        # 5) Собираем всё в словарь
        instance_data = {
            "audio": audio,                  # аудиосигнал (1, num_frames)
            "spectrogram": spectrogram,      # [time, freq]
            "text": text,                    # raw text
            "text_encoded": text_encoded,    # encoded text
            "audio_path": audio_path,        # путь к аудиофайлу
        }

        # 6) Применяем остальные instance-трансформы (если есть)
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        """
        Считываем аудио с диска, при необходимости ресэмплим до target_sr.
        """
        audio_tensor, sr = torchaudio.load(path)
        # Оставим один канал (если аудио многоканальное)
        audio_tensor = audio_tensor[0:1, :]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )
        return audio_tensor

    def get_spectrogram(self, audio):
        """
        Вызываем instance_transforms["get_spectrogram"], приводим результат к [time, freq].
        """
        # 1) Применяем MelSpectrogram / Spectrogram (как настроено в instance_transforms)
        spec = self.instance_transforms["get_spectrogram"](audio)

        # 2) Удаляем channel=1, если есть ([1, freq, time] -> [freq, time])
        if spec.dim() == 3 and spec.shape[0] == 1:
            spec = spec.squeeze(0)

        # 3) Транспонируем => [time, freq]
        spec = spec.transpose(0, 1)

        # print(f"[DEBUG] spectrogram shape = {spec.shape}")  # контрольный принт
        return spec

    def preprocess_data(self, instance_data):
        """
        Применяем остальные instance-трансформы на уровне отдельных полей
        (кроме get_spectrogram, который уже вызвали).
        """
        if self.instance_transforms is not None:
            for transform_name, transform_func in self.instance_transforms.items():
                # Пропустим генерацию спектрограммы (уже применена)
                if transform_name == "get_spectrogram":
                    continue
                # Если поле есть в instance_data, применим трансформацию
                if transform_name in instance_data:
                    instance_data[transform_name] = transform_func(
                        instance_data[transform_name]
                    )
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(index, max_audio_length, max_text_length):
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                np.array(
                    [len(CTCTextEncoder.normalize_text(el["text"])) for el in index]
                )
                >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length
        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records from dataset"
            )
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry, "Each dataset item needs 'path'."
            assert "text" in entry, "Each dataset item needs 'text'."
            assert "audio_len" in entry, "Each dataset item needs 'audio_len'."

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
