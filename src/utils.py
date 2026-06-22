"""Вспомогательные функции."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Path,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Настройка логирования.
    
    Args:
        log_dir: Директория для логов
        log_level: Уровень логирования
        log_format: Формат логов
    
    Returns:
        Logger: Настроенный логгер
    """
    log_dir.mkdir(exist_ok=True)
    
    # Создаем имя файла лога с датой
    log_filename = f"qa_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = log_dir / log_filename
    
    # Очищаем существующие хендлеры у корневого логгера
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Настраиваем корневой логгер
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Только файловый хендлер (без консоли)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    # Получаем логгер для текущего модуля
    logger = logging.getLogger("qa_system")
    logger.info(f"=== Q&A система запущена ===")
    logger.info(f"Логи сохраняются в: {log_file}")
    
    return logger


def save_metadata(metadata: dict, file_path: Path) -> None:
    """
    Сохраняет метаданные в JSON файл.
    
    Args:
        metadata: Словарь с метаданными
        file_path: Путь к файлу
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_metadata(file_path: Path) -> dict:
    """
    Загружает метаданные из JSON файла.
    
    Args:
        file_path: Путь к файлу
    
    Returns:
        dict: Загруженные метаданные
    """
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Обрезает текст до указанной длины.
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина
    
    Returns:
        str: Обрезанный текст
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_score(score: float) -> str:
    """
    Форматирует score для вывода.
    
    Args:
        score: Числовое значение
    
    Returns:
        str: Отформатированная строка
    """
    return f"{score:.3f}"