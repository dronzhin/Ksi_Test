import pandas as pd
import networkx as nx

# Тестовые данные
data = {
    "id":        [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009],
    "mother_id": [None, None, 1001, 1008, 1003, 1005, 1008, 1003, None],
    "father_id": [None, None, 1002, 1002, 1004, None, 1009, None, None]
}
df = pd.DataFrame(data, dtype="Int32")

# Добавление отсутствующих родителей в граф
all_ids = frozenset(df["id"].unique())
all_parents = frozenset(df["mother_id"].dropna().unique()).union(frozenset(df["father_id"].dropna().unique()))


# Создание графа (родитель → ребёнок)
G = nx.DiGraph()
for node in all_ids.union(all_parents):
    G.add_node(node)

# Добавление рёбер
for _, row in df.iterrows():
    child = row["id"]
    for parent in [row["mother_id"], row["father_id"]]:
        if pd.notna(parent):
            G.add_edge(parent, child)

# Функция для расчёта уровня родства с корректной проверкой
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

# Пример использования
bull_id = 1006
cow_id = 1001
f_pair = calculate_inbreeding_level(G, bull_id, cow_id)

print(f"Коэффициент родства: {f_pair:.2f}%")
if f_pair > 5:
    print("Пара не подходит из-за высокого уровня родства.")
else:
    print("Пара допустима.")