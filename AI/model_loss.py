from data.save_load_csv import load_cows, load_bulls, load_pairs_results, save_csv
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

def model_loss(tensor: torch.Tensor, base_penalty: float=3000, loss_enbreeding: float = 1000, loss_overuse: float = 1000, info: bool = False) -> float:
    """
    Возвращает штраф за:
    - новые пары (bull_id, cow_id), которых нет в истории
    - избыточное использование одних и тех же быков (>10% от всех коров)
    При этом:
    - уменьшает штраф на среднее значение 'average_ebv' по найденным парам

    Параметры:
        tensor (torch.Tensor): Тензор размером [N, M], где N — число коров, M — число быков
        loss_enbreeding (float): Штраф за каждую новую пару
        loss_overuse (float): Штраф за каждую **лишнюю** корову на перегруженного быка

    Возвращает:
        float: Общий штраф (с учётом качества найденных пар)
    """

    # Загрузка данных
    pairs = load_pairs_results()
    cows_df = load_cows()
    bulls_df = load_bulls()

    # Преобразуем в массивы numpy для эффективного доступа
    cows = cows_df['id'].values
    bulls = bulls_df['id'].values

    # Создаём словарь для быстрого поиска значения average_ebv и difference_ebv по паре
    pair_average = dict(zip(zip(pairs['bull_id'], pairs['cow_id']), pairs['average_ebv']))
    pair_difference = dict(zip(zip(pairs['bull_id'], pairs['cow_id']), pairs['difference_ebv']))

    existing_pairs = set(zip(pairs['bull_id'], pairs['cow_id']))

    # Получаем предсказанные индексы быков
    out_model = tensor.argmax(dim=1).cpu().numpy()

    # Превращаем в bull_id
    predicted_bull_ids = bulls[out_model]

    # Формируем пары и считаем пропущенные + собираем quality
    predicted_pairs = list(zip(predicted_bull_ids, cows))

    missing_count = 0
    found_pair_average = []
    found_pair_difference = []

    for pair in predicted_pairs:
        if pair not in existing_pairs:
            missing_count += 1
        else:
            # Если пара существует — сохраняем её качество
            found_pair_average.append(pair_average[pair])
            found_pair_difference.append(pair_difference[pair])

    enbreeding_penalty = loss_enbreeding * missing_count

    # Считаем, сколько раз используется каждый бык
    bull_usage = Counter(predicted_bull_ids)

    # Порог: 10% от общего числа коров
    usage_threshold = 0.1 * len(cows)

    # Считаем общее **превышение** порога по всем быкам
    overused_extra_cows = sum(max(count - usage_threshold, 0) for bull, count in bull_usage.items())

    # Штраф за избыток (только на лишние коровы)
    overuse_penalty = loss_overuse * overused_extra_cows

    # Учёт качества найденных пар
    avg_average = np.mean(found_pair_average) if found_pair_average else 0.0
    avg_difference = np.mean(found_pair_difference) if found_pair_difference else 0.0

    # Общий штраф с поправкой на качество
    total_penalty = base_penalty + enbreeding_penalty + overuse_penalty - avg_average - avg_difference

    if info:
        print(f"Enbreeding penalty: {enbreeding_penalty}")
        print(f"Overuse penalty: {overuse_penalty}")
        print(f"Avg EBV of found pairs: {avg_average}")
        print(f"Difference EBV of found pairs: {avg_difference}")
        print(f"Total penalty: {total_penalty}")

    return total_penalty

if __name__ == '__main__':
    # Создаем случайный тензор размером [4, 6]
    logits = torch.randn(17000, 39)

    # Применяем softmax по последней оси (по классам)
    probabilities = F.softmax(logits, dim=1)

    #print(probabilities)
    print(model_loss(probabilities, info=True))