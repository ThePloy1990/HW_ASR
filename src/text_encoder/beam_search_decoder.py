# text_encoder/beam_search_decoder.py

import math
import torch

class BeamSearchDecoder:
    """
    Beam search c учетом (опционально) LM из HuggingFace (GPT2 и пр.).
    """

    def __init__(
        self,
        text_encoder,
        beam_size=5,
        cutoff_prob=1.0,
        cutoff_top_n=40,
        lm=None,        # ссылка на класс HuggingFaceLM
        lm_alpha=0.5,   # вес LM при пересчёте
    ):
        """
        Args:
            text_encoder: (CTCTextEncoder) кодировщик текста.
            beam_size (int): ширина бима.
            cutoff_prob (float): игнорировать символы с prob < cutoff_prob.
            cutoff_top_n (int): игнорировать все, кроме top_n символов.
            lm: (HuggingFaceLM) внешняя LM.
            lm_alpha (float): насколько сильно влиять LM на итоговый скор.
        """
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.blank_idx = text_encoder.blank_idx

        self.lm = lm
        self.lm_alpha = lm_alpha

        # кэш для уже посчитанных префиксов (чтобы не гонять LM тысячу раз)
        self.lm_cache = {}

    @torch.no_grad()
    def __call__(self, ctc_log_probs: torch.Tensor, seq_lens: torch.Tensor):
        """
        Batch-обёртка: на каждом элементе батча запускаем beam search.
        """
        batch_size = ctc_log_probs.size(0)
        results = []

        for b in range(batch_size):
            log_probs_single = ctc_log_probs[b]  # (max_time, n_tokens)
            seq_len = seq_lens[b].item()
            log_probs_single = log_probs_single[:seq_len]  # обрезаем по длине

            beams_for_one = self.beam_search_single(log_probs_single)
            top_hyp_strs = [h[0] for h in beams_for_one]  # берем только строки
            results.append(top_hyp_strs)

        return results

    def beam_search_single(self, log_probs: torch.Tensor):
        """
        CTC Beam Search для одного аудиосэмпла + опциональный LM re-scoring.
        """
        time_steps = log_probs.size(0)

        # beam = list of (prefix_idxs, acoustic_score, prob_b, prob_nb)
        # prefix_idxs — список индексов символов
        # acoustic_score — сумма логитов CTC (без учёта LM)
        # prob_b, prob_nb — расклад для blank / non-blank
        beam = [
            ([], 0.0, math.log(1.0), -float("inf"))
        ]

        for t in range(time_steps):
            step_log_probs = log_probs[t]  # (n_tokens,)

            # cutoff_top_n
            if (self.cutoff_top_n < step_log_probs.size(0)) or (self.cutoff_prob < 1.0):
                top_values, top_indices = step_log_probs.topk(self.cutoff_top_n)
                mask = top_values >= math.log(self.cutoff_prob)
                top_values = top_values[mask]
                top_indices = top_indices[mask]
            else:
                top_values = step_log_probs
                top_indices = torch.arange(
                    0, step_log_probs.size(0), device=step_log_probs.device
                )

            next_beam_candidates = {}

            for (prefix_idxs, prefix_score, pb, pnb) in beam:
                # 1) BLANK
                blank_lp = step_log_probs[self.blank_idx].item()
                new_pb = math.logsumexp([pb + blank_lp, pnb + blank_lp], 0)
                new_score_blank = prefix_score + blank_lp

                self._add_beam(
                    next_beam_candidates,
                    prefix_idxs,
                    new_score_blank,
                    new_pb,
                    pnb
                )

                # 2) Другие символы
                for token_id, token_lp in zip(top_indices, top_values):
                    token_id = token_id.item()
                    if token_id == self.blank_idx:
                        continue  # blank уже учли
                    last_char = prefix_idxs[-1] if prefix_idxs else None

                    # CTC merge
                    if token_id == last_char:
                        new_pnb = math.logsumexp([
                            pnb + token_lp.item(),
                            pb + token_lp.item()
                        ], 0)
                        new_prefix = prefix_idxs  # не добавляем символ, сливается
                    else:
                        new_pnb = math.logsumexp([
                            pnb + token_lp.item(),
                            pb + token_lp.item()
                        ], 0)
                        new_prefix = prefix_idxs + [token_id]

                    # Акустический скор: складываем логит
                    new_acoustic_score = prefix_score + token_lp.item()

                    # Если есть LM — прибавим её скор умноженный на lm_alpha
                    if self.lm is not None and len(new_prefix) > 0:
                        prefix_str = self.text_encoder.decode(
                            new_prefix, ctc_merge_repeated=False
                        )
                        lm_score = self._get_lm_score(prefix_str)
                        new_score_total = new_acoustic_score + self.lm_alpha * lm_score
                    else:
                        new_score_total = new_acoustic_score

                    self._add_beam(
                        next_beam_candidates,
                        new_prefix,
                        new_score_total,
                        pb,
                        new_pnb
                    )

            # выбираем top beam_size
            beam = sorted(
                next_beam_candidates.values(),
                key=lambda x: x[1],  # x[1] = total_score
                reverse=True
            )[: self.beam_size]

        # сортируем итог
        beam = sorted(beam, key=lambda x: x[1], reverse=True)

        final_results = []
        for (p_idxs, total_score, pb, pnb) in beam:
            hyp_str = self.text_encoder.ctc_decode(p_idxs)
            final_results.append((hyp_str, total_score))

        return final_results

    def _add_beam(self, beam_dict, prefix_idxs, new_score, new_prob_b, new_prob_nb):
        key = tuple(prefix_idxs)
        if key not in beam_dict:
            beam_dict[key] = (prefix_idxs, new_score, new_prob_b, new_prob_nb)
        else:
            _, old_score, old_pb, old_pnb = beam_dict[key]
            if new_score > old_score:
                beam_dict[key] = (prefix_idxs, new_score, new_prob_b, new_prob_nb)

    def _get_lm_score(self, prefix_str):
        """
        Возвращает log P(prefix_str) через нашу LM.
        Кэшируем, чтобы не считать по 100 раз одну и ту же строку.
        """
        if prefix_str in self.lm_cache:
            return self.lm_cache[prefix_str]
        score = self.lm(prefix_str)  # вызов HuggingFaceLM.forward(text)
        self.lm_cache[prefix_str] = score
        return score
