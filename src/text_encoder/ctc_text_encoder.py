import re
from string import ascii_lowercase
import torch

from .beam_search_decoder import BeamSearchDecoder

class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): список символов алфавита для языка.
                Если None, используется ascii (a-z и пробел).
        """
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        # vocab[0] = '' (пустой символ — blank)
        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        # Явно указываем индекс blank
        self.blank_idx = 0

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert isinstance(item, int)
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        """
        Превращает строку в список индексов.
        """
        text = self.normalize_text(text)
        try:
            return torch.tensor(
                [self.char2ind[char] for char in text],
                dtype=torch.long
            )
        except KeyError:
            unknown_chars = [char for char in text if char not in self.char2ind]
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Простое декодирование (без учёта CTC).
        Возвращает строку с повторяющимися символами и blank-токенами.
        """
        return "".join(
            [self.ind2char[int(ind)] for ind in inds]
        ).strip()

    def ctc_decode(self, inds) -> str:
        """
        Greedy CTC-декодирование:
          - Удаляем blank-токен (индекс 0 = self.EMPTY_TOK).
          - Сжимаем подряд идущие одинаковые символы (чтобы "lllooo" стало "lo").
        """
        decoded_chars = []
        prev_char = None
        for i in inds:
            char = self.ind2char[int(i)]
            # Пропускаем blank
            if char == self.EMPTY_TOK:
                prev_char = None
                continue
            # Сжимаем подряд идущие одинаковые символы
            if char == prev_char:
                continue
            decoded_chars.append(char)
            prev_char = char

        return "".join(decoded_chars).strip()

    @staticmethod
    def normalize_text(text: str):
        """
        Приведение к нижнему регистру и удаление
        лишних символов (не a-z и пробел).
        """
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


    def beam_search_decode(
        self,
        log_probs: torch.Tensor,       # (B, T, vocab_size) – log_softmax
        log_probs_length: torch.Tensor,# (B,) – длины T
        beam_size=5,
        cutoff_prob=1.0,
        cutoff_top_n=40
    ):
        """
        Запускает beam search на батче выходов CTC-модели (log_probs).
        Возвращает список списков строк (до beam_size гипотез на каждый элемент батча).
        """
        decoder = BeamSearchDecoder(
            text_encoder=self,
            beam_size=beam_size,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n
        )
        return decoder(log_probs, log_probs_length)
