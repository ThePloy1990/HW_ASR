# Отчёт

## 1. Как воспроизвести модель?

**Первый этап** (пример):  
- Запускать на датасете LibriSpeech (`train-clean-100`).  
- Использовать конфиг `baseline.yaml`, где настроены параметры:
  - `n_epochs = 45`,  
  - `optimizer = AdamW, lr=3e-4`,  
  - `lr_scheduler = OneCycleLR (max_lr=1e-3)`,  
  - Модель: BiLSTM (2 слоя, hidden=256, bidirectional). 

**Второй этап** (пример):
- Дообучить ещё на 20–30 эпох, включая Beam Search + GPT‑2 LM, с параметрами `lm_alpha=0.5`.  
- В итоге ~50–60 эпох суммарно.

Запуск:
```bash
python train.py trainer.n_epochs=45
# затем
python train.py trainer.n_epochs=60 trainer.resume_from="model_best.pth"
```

---

## 2. Журналы обучения (logs) финальной модели

Ниже представлены ключевые метрики по эпохам (последние 10–15 эпох):
```
epoch |   train_loss   |   val_loss   | test_loss | val_WER | test_WER | ...
  ...
  40   |  ~0.57         |  ~0.76       |  ~0.73    |  ~0.63  |  ~0.62
  41   |  ~0.58         |  ~0.76       |  ~0.73    |  ~0.62  |  ~0.61
  ...
  45   |  ~0.52         |  ~0.73       |  ~0.71    |  ~0.61  |  ~0.60
  46   |  ~0.49         |  ~0.72       |  ~0.70    |  ~0.60  |  ~0.59
  47   |  ~0.51         |  ~0.72       |  ~0.70    |  ~0.60  |  ~0.59
  48   |  ~0.49         |  ~0.71       |  ~0.70    |  ~0.60  |  ~0.58
```
Данные с Wandb:
![Снимок экрана 2025-01-11 в 14.28.08.png](%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202025-01-11%20%D0%B2%2014.28.08.png)
![Снимок экрана 2025-01-11 в 14.29.20.png](%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202025-01-11%20%D0%B2%2014.29.20.png)
---

## 3. Как обучили финальную модель?

1. **Архитектура**:  
   - BiLSTM, 2 слоя, bidirectional=True, hidden=256, выход → Linear(512, vocab_size=28).  

2. **Аугментации** (пример):  
   - Gain, TimeStretch, PolarityInversion, AddColoredNoise (torch_audiomentations).  
   - Применяются на аудио при `train` (p=0.5).  

3. **Гиперпараметры**:  
   - Batch size: (N), Learning Rate: 3e-4 (OneCycleLR с max_lr=1e-3),  
   - Epochs: 45 (потом ещё 15 до 60).  
   - Clip grad norm: 5.0.  

4. **Внешняя LM**:  
   - GPT‑2 (huggingface_lm), shallow fusion при beam search (beam_size=5, alpha=0.5).  
   - Подключено в `beam_search_decoder.py`, складывается `acoustic_score + alpha * lm_score`.  

---

## 4. Что пробовали?

1. **Без LM**: Argmax CTC давал WER около ~0.92–0.95 после 50 эпох.  
2. **С LM** (GPT‑2, alpha=0.5): улучшил WER до ~0.58–0.60.  
3. **Увеличение эпох**: c 20 до 45 – существенный прирост, потом улучшения замедлились, но всё ещё заметны на 40–50 эпохах.  
4. **Аугментации**: включили 4 вида (Gain, Stretch, Polarity, Noise). Это помогло избежать overfitting.

---

## 5. Что сработало, а что нет?

- **Работает**:  
  - Подключение внешней LM (GPT-2) через shallow fusion дало заметное улучшение WER.  
  - Gradient clipping (max_grad_norm=5) позволил избежать NaN (взрыв градиентов).  
  - Аугментации (особенно Gain / Noise) стабилизировали обучение.  

- **Не очень помогло**:  
  - Слишком агрессивные lr (1e-2) вызывали NaN в начале, поэтому снизили до 1e-3.  
  - Увеличение beam_size свыше 5 сильно замедляет инференс, прирост качества небольшой.  

---

## 6. Обзор бонусных задач

1. **(+10 баллов) Внешняя LM**: реализовано GPT‑2 через `huggingface_lm.py`, суммирование `acoustic_score + alpha * lm_score` в `beam_search_decoder.py`.

---

Таким образом, итоговая **итоговая метрика** на test-clean:  
- **WER ≈ 0.58–0.60** (с LM).  
- **CER ≈ 0.21–0.22**.
