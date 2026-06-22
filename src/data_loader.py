"""Модуль для загрузки данных."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data_from_excel(file_path: Path, question_col: str = "вопрос", answer_col: str = "ответ") -> tuple[list[str], list[str]]:
    """
    Загружает данные из Excel файла.

    Args:
        file_path: Путь к Excel файлу
        question_col: Название колонки с вопросами
        answer_col: Название колонки с ответами

    Returns:
        Tuple[List[str], List[str]]: Списки вопросов и ответов

    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если колонки не найдены
    """
    logger.info(f"Загрузка данных из {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        df = pd.read_excel(file_path)
        logger.info(f"Загружено {len(df)} строк")
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        raise

    # Проверяем наличие колонок
    if question_col not in df.columns:
        raise ValueError(f"Колонка '{question_col}' не найдена. Доступные колонки: {df.columns.tolist()}")

    if answer_col not in df.columns:
        raise ValueError(f"Колонка '{answer_col}' не найдена. Доступные колонки: {df.columns.tolist()}")

    # Удаляем пустые строки
    df = df.dropna(subset=[question_col, answer_col])
    df = df[df[question_col].astype(str).str.strip() != ""]
    df = df[df[answer_col].astype(str).str.strip() != ""]

    questions = df[question_col].astype(str).tolist()
    answers = df[answer_col].astype(str).tolist()

    logger.info(f"Загружено {len(questions)} вопросов и ответов (после очистки)")
    return questions, answers


def load_data_from_csv(file_path: Path, question_col: str = "вопрос", answer_col: str = "ответ") -> tuple[list[str], list[str]]:
    """
    Загружает данные из CSV файла.

    Args:
        file_path: Путь к CSV файлу
        question_col: Название колонки с вопросами
        answer_col: Название колонки с ответами

    Returns:
        Tuple[List[str], List[str]]: Списки вопросов и ответов
    """
    logger.info(f"Загрузка данных из {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        logger.info(f"Загружено {len(df)} строк")
    except UnicodeDecodeError:
        # Пробуем другую кодировку
        df = pd.read_csv(file_path, encoding="cp1251")
        logger.info(f"Загружено {len(df)} строк (кодировка cp1251)")
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        raise

    # Проверяем наличие колонок
    if question_col not in df.columns:
        raise ValueError(f"Колонка '{question_col}' не найдена. Доступные колонки: {df.columns.tolist()}")

    if answer_col not in df.columns:
        raise ValueError(f"Колонка '{answer_col}' не найдена. Доступные колонки: {df.columns.tolist()}")

    # Удаляем пустые строки
    df = df.dropna(subset=[question_col, answer_col])
    df = df[df[question_col].astype(str).str.strip() != ""]
    df = df[df[answer_col].astype(str).str.strip() != ""]

    questions = df[question_col].astype(str).tolist()
    answers = df[answer_col].astype(str).tolist()

    logger.info(f"Загружено {len(questions)} вопросов и ответов (после очистки)")
    return questions, answers


def load_data(file_path: Path, question_col: str = "вопрос", answer_col: str = "ответ") -> tuple[list[str], list[str]]:
    """
    Универсальная функция загрузки данных (автоматически определяет формат).

    Args:
        file_path: Путь к файлу
        question_col: Название колонки с вопросами
        answer_col: Название колонки с ответами

    Returns:
        Tuple[List[str], List[str]]: Списки вопросов и ответов
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        return load_data_from_excel(file_path, question_col, answer_col)
    elif file_path.suffix.lower() == ".csv":
        return load_data_from_csv(file_path, question_col, answer_col)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}. Поддерживаются: .xlsx, .xls, .csv")
