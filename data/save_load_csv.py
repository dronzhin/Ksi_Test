import pandas as pd
import os

# Константа для папки с данными
DATA_DIR = os.path.dirname(__file__)

# Функция загрузки файлов
def load_file(file_name, dtype):
    """
    Загружает файл с данными в формате CSV.

    :param file_name: Имя файла для загрузки.
    :param dtype: Словарь с типами данных для столбцов. Например, {'id': 'str', 'mother_id': 'str', 'father_id': 'str'}.
    :return: Датафрейм с данными или None, если файл не найден.
    """
    file_path = os.path.join(DATA_DIR,file_name)
    try:
        return pd.read_csv(file_path, dtype=dtype)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def load_pedigree():
    """
    Загружает файл pedigree.csv с информацией о семействе коров и быков
    """
    return load_file("pedigree.csv", dtype={"id": "str", "mother_id": "str", "father_id": "str"})

def load_bulls():
    """
    Загружает файл bulls.csv с информацией о быках.
    Проверяет на наличие Null в столбике ebv:
        - Если есть, заполняет их средним значением (без учета Null).
    Удаляет ненужный столбец 'descendants_count', если он есть.
    """
    # Загрузка файла с данными
    df = load_file("bulls.csv", dtype={"id": "str", "ebv": "float32"})

    # Удаление столбца descendants_count, если существует
    if 'descendants_count' in df.columns:
        df = df.drop(columns=['descendants_count'])

    # Заполнение пропущенных значений в 'ebv'
    if df['ebv'].isnull().any():
        mean_ebv = df['ebv'].mean(skipna=True)
        df['ebv'] = df['ebv'].fillna(mean_ebv)

    return df

def load_cows():
    """
    Загружает файл cows.csv с информацией о коровах.
    Проверяет на наличие Null в столбике ebv:
    - Если есть, заполняет их средним значением (без учета Null).
    """

    # Загрузка файла с данными
    df = load_file("cows.csv", dtype={"id": "str", "ebv": "float32"})

    # Заполнение пропущенных значений в 'ebv'
    if df['ebv'].isnull().any():
        mean_ebv = df['ebv'].mean(skipna=True)
        df['ebv'] = df['ebv'].fillna(mean_ebv)

    return df

def load_pairs_results():
    """
    Загружает файл pairs_results с информацией о всех парах кород и быков с уровнем родства не более 5%
    """
    dtype = {
        'bull_id': 'str',
        'bull_ebv': 'float32',
        'cow_id': 'str',
        'cow_ebv': 'float32',
        'average_ebv': 'float32',
        'difference_ebv': 'float32',
        'inbreeding_level': 'float32'
    }
    return load_file("pairs_results.csv", dtype=dtype)

def save_csv(df, path, col):
    """
    Сохраняет DataFrame в CSV-файл с указанным путем.

    :param df: Датафрейм для сохранения.
    :param path: Путь к файлу, в который будет сохранен датафрейм. Может быть абсолютным или относительным.
    """
    # Округляем числа с плавающей точкой до одного знака после запятой
    df[col] = df[col].round(1)

    # Сохраняем датафрейм в CSV-файл
    df.to_csv(path, index=False)