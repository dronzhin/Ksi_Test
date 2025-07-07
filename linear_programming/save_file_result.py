import pandas as pd

# Загрузка данных из файла
df = pd.read_csv('optimal_pairs_lp_85.csv')

# Оставляем только нужные столбцы
df_filtered = df[['bull_id', 'cow_id']]

# Сохраняем в новый CSV-файл
df_filtered.to_csv('cow_bull_assignments.csv', index=False)

print("Файл успешно сохранен как cow_bull_assignments.csv")