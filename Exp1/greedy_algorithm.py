import pandas as pd
from data.save_load_csv import load_cows, load_bulls, load_pairs_results, DATA_DIR, save_csv

# Загрузка данных
pairs = load_pairs_results() # Все пары
cows = load_cows()            # Список коров
bulls = load_bulls()          # Список быков

# Список параметров функции
func_params = {0.6, 0.65, 0.7, 0.75, 0.8, 0.85}
final_result = {
    'name': [],
    'average_ebv': [],
    'difference_ebv': []
}

for i in func_params:

    # Расчет целевой функции
    pairs["fitness"] = i * pairs["average_ebv"] + (1-i) * pairs["difference_ebv"]

    # Сортировка по убыванию целевой функции
    pairs.sort_values(by="fitness", ascending=False, inplace=True)

    # Ограничение на использование быков (≤10% от числа коров)
    max_usage = int(0.1 * len(cows))
    bull_usage = {bull_id: 0 for bull_id in bulls["id"]}

    # Назначение пар
    assigned_cows = set()
    final_pairs = []

    for _, row in pairs.iterrows():
        bull_id = row["bull_id"]
        cow_id = row["cow_id"]
        if bull_usage[bull_id] < max_usage and cow_id not in assigned_cows:
            final_pairs.append(row)
            bull_usage[bull_id] += 1
            assigned_cows.add(cow_id)

    # Сохранение результата
    final_df = pd.DataFrame(final_pairs)
    path = f"optimal_pairs_greedy_{int(i*100)}.csv"
    save_csv(final_df,path, ['bull_ebv', 'cow_ebv', 'average_ebv', 'difference_ebv', 'inbreeding_level'])

    # Сохранение итоговых данных
    final_result['name'].append(path)
    final_result['average_ebv'].append(final_df['average_ebv'].mean())
    final_result['difference_ebv'].append(final_df['difference_ebv'].mean())

final_result = pd.DataFrame(final_result)
save_csv(final_result, 'final_result.csv', ['average_ebv', 'difference_ebv'])

print(final_result)