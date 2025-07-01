import pickle

with open('all_pairs.pickle', 'rb') as f:
  all_pairs = pickle.load(f)

# Фильтрация по инбридингу (≤5%)
valid_pairs = all_pairs[all_pairs["inbreeding"] <= 5]

# Ограничение на использование быков (≤10% от общего числа коров)
bull_usage = valid_pairs["bull_id"].value_counts(normalize=True)
valid_bulls = bull_usage[bull_usage <= 0.1].index
final_pairs = valid_pairs[valid_pairs["bull_id"].isin(valid_bulls)]

# Расчет EBV потомства
final_pairs["ebv_offspring"] = (final_pairs["ebv_bull"] + final_pairs["ebv_cow"]) / 2

# Расчет разброса (стандартного отклонения)
final_pairs["ebv_std"] = abs(final_pairs["ebv_bull"] - final_pairs["ebv_cow"]) / 2

# Целевая функция
final_pairs["fitness"] = 0.7 * final_pairs["ebv_offspring"] + 0.3 * final_pairs["ebv_std"]

# Сортировка по целевой функции
final_pairs.sort_values(by="fitness", ascending=False, inplace=True)