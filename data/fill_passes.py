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
    Загружает файл pedigree.csv с информацией о семействе коров.
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
