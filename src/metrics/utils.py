def calc_cer(ref: str, hyp: str) -> float:
    """
    Calculate the Character Error Rate (CER) между двумя строками:
    CER = edit_distance / len(ref).

    Если ref пустая, по определению возвращаем len(hyp).
    """
    # Если целевой текст пуст - считаем CER = длина гипотезы
    if len(ref) == 0:
        return float(len(hyp))

    # Создадим DP-таблицу размера (len(ref)+1) x (len(hyp)+1)
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    # Инициализация
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    # Заполняем таблицу
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # удаление символа из ref
                d[i][j - 1] + 1,        # вставка символа из hyp
                d[i - 1][j - 1] + cost  # замена (cost=0 если совпадают)
            )

    edit_distance = d[len(ref)][len(hyp)]
    return edit_distance / len(ref)


def calc_wer(target_text: str, predicted_text: str) -> float:
    """
    Calculate the Word Error Rate (WER) between two strings:
    WER = edit_distance / len(ref_words)

    Edit distance здесь считается классическим алгоритмом Левенштейна,
    но на уровне слов, а не символов.

    Args:
        target_text (str): целевая строка (ground truth).
        predicted_text (str): предсказанная строка (model output).

    Returns:
        wer (float): результат в диапазоне [0..∞),
                     обычно от 0 (идеально) до 1+ (если ошибок больше, чем слов).
    """
    # Разбиваем строки на слова
    ref_words = target_text.split()
    hyp_words = predicted_text.split()

    # Если целевая строка пустая, WER = кол-во слов в гипотезе
    if len(ref_words) == 0:
        return float(len(hyp_words))

    # Создаём DP-таблицу, размеры (ref_len + 1) x (hyp_len + 1)
    ref_len = len(ref_words)
    hyp_len = len(hyp_words)
    d = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    # Инициализация
    for i in range(ref_len + 1):
        d[i][0] = i
    for j in range(hyp_len + 1):
        d[0][j] = j

    # Заполнение таблицы Левенштейна
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # удаление (delete)
                d[i][j - 1] + 1,      # вставка (insert)
                d[i - 1][j - 1] + cost  # замена (substitute) - cost=0, если совпадают
            )

    edit_distance = d[ref_len][hyp_len]
    wer = edit_distance / ref_len
    return wer

