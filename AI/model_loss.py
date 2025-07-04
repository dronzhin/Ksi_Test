from data.save_load_csv import load_cows, load_bulls, load_pairs_results, save_csv
import torch
import torch.nn.functional as F
import numpy as np
import pickle

def differentiable_loss(logits: torch.Tensor,
                        existing_pairs_mask: torch.Tensor,
                        avg_ebv_matrix: torch.Tensor,
                        diff_ebv_matrix: torch.Tensor,
                        num_cows: int,
                        loss_enbreeding: float = 1000,
                        loss_overuse: float = 1000,
                        base_penalty: float = 3000,
                        usage_threshold_ratio: float = 0.1,
                        info: bool = False):
    """
    Дифференцируемая версия функции model_loss для обучения модели.

    Параметры:
        logits (torch.Tensor): [N_cows, N_bulls] — выход модели (логиты)
        existing_pairs_mask (torch.Tensor): бинарная матрица размера [N_cows, N_bulls], где 1 — новая пара
        avg_ebv_matrix (torch.Tensor): матрица средних EBV для пар [N_cows, N_bulls]
        num_cows (int): количество коров
        остальные параметры аналогичны оригиналу

    Возвращает:
        torch.Tensor: скалярный loss
    """
    probs = F.softmax(logits, dim=1)

    # Штраф за новые пары (взвешиваем вероятности на маску новых пар)
    enbreeding_penalty = (probs * existing_pairs_mask).sum() * loss_enbreeding

    # Штраф за перегрузку быков: сумма превышения порога
    usage_threshold = usage_threshold_ratio * num_cows
    bull_usage = probs.sum(dim=0)  # [N_bulls]
    overuse_penalty = torch.relu(bull_usage - usage_threshold).sum() * loss_overuse

    # Награда за качество найденных пар (мы минимизируем loss => награда вычитается)
    avg_ebv_reward = (probs * avg_ebv_matrix).sum()/num_cows
    diff_ebv_reward = (probs * diff_ebv_matrix).sum() / num_cows

    # Общий loss
    total_loss = base_penalty + enbreeding_penalty + overuse_penalty - avg_ebv_reward - diff_ebv_reward

    if info:
        print(f"Enbreeding penalty: {enbreeding_penalty.item()}")
        print(f"Overuse penalty: {overuse_penalty.item()}")
        print(f"Avg_ebv reward (avg_ebv): {avg_ebv_reward.item()}")
        print(f"Diff_ebv reward (avg_ebv): {diff_ebv_reward.item()}")
        print(f"Total loss: {total_loss.item()}")

    return total_loss

if __name__ == '__main__':

    # # Загрузка данных
    # pairs = load_pairs_results()
    # cows_df = load_cows()
    # bulls_df = load_bulls()
    #
    # # Предположим, у нас есть bulls и cows как списки ID
    # bull_ids = bulls_df['id'].values
    # cow_ids = cows_df['id'].values
    #
    # # Создаём матрицу EBV и маску новых пар
    # avg_ebv_tensor = torch.zeros(len(cow_ids), len(bull_ids))
    # diff_ebv_tensor = torch.zeros(len(cow_ids), len(bull_ids))
    #
    # existing_pairs_mask = torch.ones(len(cow_ids), len(bull_ids))  # 1 — новая пара, 0 — существующая
    #
    # for _, row in pairs.iterrows():
    #     bull_idx = np.where(bull_ids == row['bull_id'])[0][0]
    #     cow_idx = np.where(cow_ids == row['cow_id'])[0][0]
    #     avg_ebv_tensor[cow_idx, bull_idx] = row['average_ebv']
    #     diff_ebv_tensor[cow_idx, bull_idx] = row['difference_ebv']
    #
    #     existing_pairs_mask[cow_idx, bull_idx] = 0  # существующая пара — нет штрафа
    #
    #
    # data = {
    #     'existing_pairs_mask': existing_pairs_mask,
    #     'avg_ebv_tensor': avg_ebv_tensor,
    #     'diff_ebv_tensor': diff_ebv_tensor,
    #     'num_cows': len(cow_ids)
    # }
    #
    # with open('loss_params.pickle', 'wb') as f:
    #     pickle.dump(data, f)

    with open('loss_params.pickle', 'rb') as f:
        data_out = pickle.load(f)

    existing_pairs_mask = data_out['existing_pairs_mask']
    avg_ebv_tensor = data_out['avg_ebv_tensor']
    diff_ebv_tensor = data_out["diff_ebv_tensor"]
    num_cows = data_out['num_cows']

    # Создаем случайный тензор размером [4, 6]
    logits = torch.randn(17177, 39)

    # # Применяем softmax по последней оси (по классам)
    # probabilities = F.softmax(logits, dim=1)
    # probabilities = probabilities.argmax(dim=1)

    loss = differentiable_loss(
                    logits=logits,
                    existing_pairs_mask=existing_pairs_mask,
                    avg_ebv_matrix=avg_ebv_tensor,
                    diff_ebv_matrix=diff_ebv_tensor,
                    num_cows=num_cows,
                    info=True
                )

    print(loss)