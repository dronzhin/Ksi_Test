import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
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
    # Создание модели
    model = LpProblem(name=f"optimal_pairs_{int(i * 100)}", sense=LpMaximize)

    # Переменные: x_ij = 1, если корова i спаривается с быком j
    x = {(cow, bull): LpVariable(name=f"x_{cow}_{bull}", cat="Binary")
         for cow in cows["id"]
         for bull in bulls["id"]}

    # Расчет целевой функции
    fitness_dict = pairs.set_index(["cow_id", "bull_id"])["fitness"].to_dict()

    # Целевая функция
    model += lpSum([
        fitness_dict.get((cow, bull), 0) * x[(cow, bull)]  # 0, если пара отсутствует
        for cow in cows["id"]
        for bull in bulls["id"]
    ])

    for cow in cows["id"]:
        model += lpSum([x[(cow, bull)] for bull in bulls["id"]]) == 1

    max_usage = int(0.1 * len(cows))
    for bull in bulls["id"]:
        model += lpSum([x[(cow, bull)] for cow in cows["id"]]) <= max_usage

    # Решение задачи
    model.solve()

    # Сохранение результата
    final_pairs = []
    for cow in cows["id"]:
        for bull in bulls["id"]:
            if value(x[(cow, bull)]) == 1:
                # Поиск строки в pairs для текущей пары (cow, bull)
                pair_row = pairs[(pairs["cow_id"] == cow) & (pairs["bull_id"] == bull)]
                # Проверка, существует ли такая пара
                if not pair_row.empty:
                    final_pairs.append({
                        "cow_id": cow,
                        "cow_ebv": pair_row["cow_ebv"].values[0],
                        "bull_id": bull,
                        "bull_ebv": pair_row["bull_ebv"].values[0],
                        "average_ebv": pair_row["average_ebv"].values[0],
                        "difference_ebv": pair_row["difference_ebv"].values[0],
                        "inbreeding_level": pair_row["inbreeding_level"].values[0]
                    })

    # Сохранение итоговых данных
    final_df = pd.DataFrame(final_pairs)
    path = f"optimal_pairs_lp_{int(i * 100)}.csv"
    final_result['name'].append(path)
    final_result['average_ebv'].append(float(final_df['average_ebv'].mean()))
    final_result['difference_ebv'].append(float(final_df['difference_ebv'].mean()))
    save_csv(final_df, path, ['bull_ebv', 'cow_ebv', 'average_ebv', 'difference_ebv', 'inbreeding_level'])

final_result = pd.DataFrame(final_result)
save_csv(final_result, 'final_result.csv', ['average_ebv', 'difference_ebv'])

print(final_result)

