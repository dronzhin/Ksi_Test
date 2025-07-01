import pandas as pd
import networkx as nx
import os
from save_load_csv import DATA_DIR, load_bulls, load_cows, load_pedigree
from tqdm import tqdm
from itertools import product

# Загрузка данных из CSV файлов
bulls_df = load_bulls()
cows_df = load_cows()
pedigree_df = load_pedigree()

# Добавление отсутствующих родителей в граф
all_ids = frozenset(pedigree_df["id"].unique())
all_parents = frozenset(pedigree_df["mother_id"].dropna().unique()).union(frozenset(pedigree_df["father_id"].dropna().unique()))

# Создание графа (родитель → ребёнок)
G = nx.DiGraph()
for node in all_ids.union(all_parents):
    G.add_node(node)

# Добавление рёбер
for _, row in pedigree_df.iterrows():
    child = row["id"]
    for parent in [row["mother_id"], row["father_id"]]:
        if pd.notna(parent):
            G.add_edge(parent, child)

# Функция для расчёта уровня родства с проверкой
def calculate_inbreeding_level(G, animal1, animal2):
    try:
        # Проверка наличия узлов в графе
        if animal1 not in G or animal2 not in G:
            print(f"Один из ID отсутствует в графе: {animal1}, {animal2}")
            return 0.0

        # Проверка прямого родства: animal2 является предком animal1?
        if nx.has_path(G, animal2, animal1):
            path_length = nx.shortest_path_length(G, animal2, animal1)
            return (1/2)**path_length * 100

        # Поиск общих предков
        ancestors1 = nx.ancestors(G, animal1)
        ancestors2 = nx.ancestors(G, animal2)
        common = ancestors1.intersection(ancestors2)
        return len(common) / max(len(ancestors1), len(ancestors2), 1) * 100
    except nx.NetworkXError:
        print("❌ Животное отсутствует в графе")
        return 0.0



# Создание списка для хранения результатов
results = []

# Общее число пар
total_pairs = len(bulls_df) * len(cows_df)

# Проход по всем парам быков и коров
for bull, cow in tqdm(product(bulls_df.iterrows(), cows_df.iterrows()), total=total_pairs, desc="Обработка пар"):
    # Извлечение данных из iterrows()
    _, bull_row = bull
    _, cow_row = cow

    bull_id = bull_row['id']
    cow_id = cow_row['id']

    # Проверка наличия идентификаторов в графе
    if bull_id not in G or cow_id not in G:
        print(f"Идентификатор {bull_id} или {cow_id} отсутствует в графе.")
        inbreeding_level = 0
    else:
        # Расчет коэффициента родства
        inbreeding_level = calculate_inbreeding_level(G, bull_id, cow_id)

    # Расчет среднего значения и разницы EBV
    average_ebv = (bull_row['ebv'] + cow_row['ebv']) / 2
    difference_ebv = abs(bull_row['ebv'] - cow_row['ebv'])

    # Сохранение результатов
    results.append({
        'bull_id': bull_id,
        'bull_ebv': bull_row['ebv'],
        'cow_id': cow_id,
        'cow_ebv': cow_row['ebv'],
        'average_ebv': average_ebv,
        'difference_ebv': difference_ebv,
        'inbreeding_level': inbreeding_level
    })

# Создание DataFrame из результатов
results_df = pd.DataFrame(results)
results_df[['bull_ebv', 'cow_ebv', 'average_ebv', 'difference_ebv', 'inbreeding_level']] = \
results_df[['bull_ebv', 'cow_ebv', 'average_ebv', 'difference_ebv', 'inbreeding_level']].round(1)

# Фильтрация по уровню инбридинга
filtered_results_df = results_df[results_df['inbreeding_level'] <= 5]

# Сохранение результатов в новый CSV файл
filtered_results_df.to_csv(os.path.join(DATA_DIR,'pairs_results.csv'), index=False)

print("Результаты сохранены в файл 'pairs_results.csv'.")
