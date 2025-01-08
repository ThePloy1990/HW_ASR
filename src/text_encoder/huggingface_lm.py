import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class HuggingFaceLM(nn.Module):

    def __init__(self, model_name="gpt2", device="cpu", use_lm=True):
        super().__init__()
        self.device = device
        self.use_lm = use_lm
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        # GPT-2 не имеет "pad_token" по умолчанию, при необходимости можно добавить
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def forward(self, text: str) -> float:
        """
        Возвращает log(prob(text)) = log P(text).
        Реализуем как "минус loss * длина", чтобы перевести
        ср. кроссэнтропию (loss) в общую логвероятность.

        Args:
            text (str): входная строка
        Returns:
            lm_score (float): log P(text)
        """
        if not text.strip():
            return 0.0  # пустая строка => пусть score=0

        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        # labels те же, что и вход — стандартная LM-задача
        outputs = self.model(inputs, labels=inputs)
        # outputs.loss — это средний cross-entropy по токенам
        # log P(text) = - loss * (число токенов)
        n_tokens = inputs.size(1)
        lm_score = -outputs.loss.item() * n_tokens
        return lm_score
