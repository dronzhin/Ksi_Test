import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

class Dataset_pair:
    """
    Класс для создания датасета парных результатов.

    Атрибуты:
        data (pd.DataFrame): Датафрейм с данными.
        le_bull (LabelEncoder): Объект LabelEncoder для быков.
        le_cow (LabelEncoder): Объект LabelEncoder для коров.
        scaler (MinMaxScaler): Объект MinMaxScaler для нормализации числовых признаков.

    Методы:
        __init__(self, data):
            Инициализирует объект Dataset_pair с данными и конвертирует их в числовые форматы.

        get_date(self):
            Возвращает датафрейм с данными.

        get_scaler(self):
            Возвращает объект MinMaxScaler для нормализации числовых признаков.

        get_le_bull(self):
            Возвращает объект LabelEncoder для булевых признаков.

        get_le_cow(self):
            Возвращает объект LabelEncoder для коротких признаков.
    """

    def __init__(self, data, embedding_dim=32):
        """
        Инициализирует объект Dataset_pair с данными и конвертирует их в числовые форматы.

        Аргументы:
            data (pd): данные пандас
        """
        self.__data = data
        self.__embedding_dim = embedding_dim  # Размерность эмбеддингов
        self.__convert_id_to_number()
        self.__data_scaler()
        self.__create_embeddings()

    def __create_embeddings(self):
        """Создает эмбеддинг-слои для быков и коров."""
        # Получение количества уникальных категорий из OrdinalEncoder
        self.num_bulls = len(self.__ordinal_encoder.categories_[0]) + 1  # Сохраните как атрибут
        self.num_cows = len(self.__ordinal_encoder.categories_[1]) + 1  # Сохраните как атрибут

        # Инициализация эмбеддинг-слоев
        self.__bull_embedding = nn.Embedding(self.num_bulls, self.__embedding_dim)
        self.__cow_embedding = nn.Embedding(self.num_cows, self.__embedding_dim)

    def __convert_id_to_number(self):
        """Конвертирует ID в числовые значения без предположения о порядке."""
        # Убедитесь, что данные имеют тип str (object), а не float
        self.__data['bull_id'] = self.__data['bull_id'].astype(str)
        self.__data['cow_id'] = self.__data['cow_id'].astype(str)

        # Используйте OrdinalEncoder для обработки категориальных признаков
        self.__ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',  # Обработка новых значений
            unknown_value=-1  # Присваивать -1 для неизвестных категорий
        )

        # Преобразование столбцов
        self.__data[['bull_id', 'cow_id']] = self.__ordinal_encoder.fit_transform(
            self.__data[['bull_id', 'cow_id']]
        )

    def __row_to_tensor(self, row):
        """Преобразует строку датафрейма в тензор."""
        # Создание DataFrame с именами признаков
        row_df = pd.DataFrame([[str(row['bull_id']), str(row['cow_id'])]],
                              columns=['bull_id', 'cow_id'])

        # Преобразование с использованием OrdinalEncoder
        encoded = self.__ordinal_encoder.transform(row_df)[0]

        # Замена -1 на максимальный индекс
        bull_index = int(encoded[0]) if encoded[0] != -1 else self.num_bulls - 1
        cow_index = int(encoded[1]) if encoded[1] != -1 else self.num_cows - 1

        # Получение эмбеддингов
        bull_emb = self.__bull_embedding(torch.tensor(bull_index))
        cow_emb = self.__cow_embedding(torch.tensor(cow_index))

        # Нормализованные числовые признаки
        numeric_cols = ['average_ebv', 'difference_ebv']
        numeric_tensor = torch.tensor(row[numeric_cols].values.astype('float32'))

        # Объединение эмбеддингов и числовых признаков
        combined_tensor = torch.cat([bull_emb, cow_emb, numeric_tensor], dim=0)
        return combined_tensor

    # Нормализация числовых признаков
    def __data_scaler(self):
        """Нормализует числовые признаки."""
        numeric_cols = ['average_ebv', 'difference_ebv']
        self.__scaler = MinMaxScaler()
        self.__data[numeric_cols] = self.__scaler.fit_transform(self.__data[numeric_cols])

    def get_tensors(self):
        """Возвращает все данные в виде тензоров."""
        tensors = []
        # Оборачиваем цикл в tqdm для отслеживания прогресса
        for _, row in tqdm(self.__data.iterrows(), total=len(self.__data), desc="Processing rows"):
            tensor = self.__row_to_tensor(row)
            tensors.append(tensor.unsqueeze(0))  # Добавление размерности батча
        return torch.cat(tensors, dim=0)  # Объединение в один тензор

    def get_date(self):
        """Возвращает датафрейм с данными."""
        return self.__data

    def get_scaler(self):
        """Возвращает объект MinMaxScaler для нормализации числовых признаков."""
        return self.__scaler

    def get_ordinal_encoder(self):
        """Возвращает объект ordinal_encoder"""
        return self.__ordinal_encoder

from data.save_load_csv import load_pairs_results

# Загрузка данных
pairs = load_pairs_results() # Все пары
pairs.drop(['bull_ebv', 'cow_ebv', 'inbreeding_level'], axis=1, inplace=True)
dataset = Dataset_pair(pairs)

# Получение тензоров
tensors = dataset.get_tensors()
# Конвертация (сериализация) объекта data в формат pickle
with open('data.pickle', 'wb') as f:
  pickle.dump(tensors, f)

print(tensors.shape)  # Пример: [300000, 68] (32 + 32 + 4)