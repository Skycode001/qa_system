"""Конфигурация системы."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Базовые пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"

# Создаем необходимые директории
for dir_path in [DATA_DIR, CACHE_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True)

# Модель эмбеддингов
MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# TF-IDF параметры
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", 5000))
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

# Гибридные веса
HYBRID_WEIGHT_TFIDF = float(os.getenv("HYBRID_WEIGHT_TFIDF", 0.3))
HYBRID_WEIGHT_EMBEDDING = float(os.getenv("HYBRID_WEIGHT_EMBEDDING", 0.7))

# Порог сходства
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.3))

# Batch size для энкодинга
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", 32))

# Логирование
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Данные
DATA_FILE = os.getenv("DATA_FILE", "questions.xlsx")
DATA_FILE_PATH = DATA_DIR / DATA_FILE
