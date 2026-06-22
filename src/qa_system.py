"""Основной класс гибридной системы поиска ответов."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CACHE_DIR,
    ENCODE_BATCH_SIZE,
    HYBRID_WEIGHT_EMBEDDING,
    HYBRID_WEIGHT_TFIDF,
    MODEL_NAME,
    SIMILARITY_THRESHOLD,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
)

logger = logging.getLogger(__name__)


class HybridQASystem:
    """
    Гибридная система поиска ответов на вопросы.
    
    Объединяет TF-IDF и эмбеддинги для поиска наиболее похожих вопросов в базе.
    """

    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        model_name: str = MODEL_NAME,
        cache_dir: Path = CACHE_DIR,
        tfidf_max_features: int = TFIDF_MAX_FEATURES,
        hybrid_weight_tfidf: float = HYBRID_WEIGHT_TFIDF,
        hybrid_weight_embedding: float = HYBRID_WEIGHT_EMBEDDING,
        threshold: float = SIMILARITY_THRESHOLD,
    ):
        """
        Инициализация системы.
        
        Args:
            questions: Список вопросов
            answers: Список ответов
            model_name: Название модели Sentence Transformers
            cache_dir: Директория для кэша
            tfidf_max_features: Максимальное количество признаков для TF-IDF
            hybrid_weight_tfidf: Вес TF-IDF в гибридном подходе
            hybrid_weight_embedding: Вес эмбеддингов в гибридном подходе
            threshold: Порог сходства для выдачи ответа
        """
        if len(questions) != len(answers):
            raise ValueError("Количество вопросов и ответов должно совпадать")

        if len(questions) == 0:
            raise ValueError("Список вопросов не может быть пустым")

        self.questions = questions
        self.answers = answers
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.hybrid_weight_tfidf = hybrid_weight_tfidf
        self.hybrid_weight_embedding = hybrid_weight_embedding
        self.threshold = threshold

        logger.info(f"Инициализация системы с {len(questions)} вопросами")
        logger.info(f"Модель: {model_name}")
        logger.info(f"Вес TF-IDF: {hybrid_weight_tfidf}, Вес эмбеддингов: {hybrid_weight_embedding}")

        # Инициализация компонентов
        self._init_tfidf(tfidf_max_features)
        self._init_embeddings()

        logger.info("Система инициализирована")

    def _init_tfidf(self, max_features: int) -> None:
        """Инициализация TF-IDF компонента."""
        logger.info("Вычисление TF-IDF матрицы...")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words=TFIDF_STOP_WORDS,
            lowercase=True,
            strip_accents="unicode",
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.questions)
        logger.info(f"TF-IDF матрица создана: {self.tfidf_matrix.shape}")

    def _get_cache_path(self) -> Path:
        """Путь к файлу кэша эмбеддингов."""
        safe_name = self.model_name.replace("/", "_")
        return self.cache_dir / f"embeddings_{safe_name}.pkl"

    def _init_embeddings(self) -> None:
        """Инициализация компонента эмбеддингов с кэшированием."""
        cache_path = self._get_cache_path()

        # Пытаемся загрузить из кэша
        if cache_path.exists():
            logger.info(f"Загрузка эмбеддингов из кэша: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    self.question_embeddings = pickle.load(f)
                logger.info(f"Эмбеддинги загружены: {self.question_embeddings.shape}")
                self.embedding_model = None
                return
            except Exception as e:
                logger.warning(f"Ошибка загрузки кэша: {e}. Пересоздаем...")

        # Вычисляем эмбеддинги
        logger.info(f"Вычисление эмбеддингов для {len(self.questions)} вопросов...")
        logger.info("Это может занять 1-2 минуты...")

        self.embedding_model = SentenceTransformer(self.model_name)

        self.question_embeddings = self.embedding_model.encode(
            self.questions,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=ENCODE_BATCH_SIZE,
        )

        logger.info(f"Эмбеддинги вычислены: {self.question_embeddings.shape}")

        # Сохраняем в кэш
        logger.info(f"Сохранение эмбеддингов в кэш: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(self.question_embeddings, f)

        logger.info("Эмбеддинги сохранены в кэш")

    def _ensure_model_loaded(self) -> None:
        """Загружает модель, если она еще не загружена."""
        if self.embedding_model is None:
            logger.info("Загрузка модели эмбеддингов...")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info("Модель загружена")

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Нормализует массив к диапазону [0, 1]."""
        if arr.max() == arr.min():
            return np.zeros_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    def find_answer(
        self,
        query: str,
        method: str = "hybrid",
        threshold: Optional[float] = None,
        return_top_k: int = 1,
    ) -> Tuple[str, float]:
        """
        Находит ответ на вопрос.
        
        Args:
            query: Текст вопроса
            method: Метод поиска ('tfidf', 'embedding', 'hybrid')
            threshold: Порог сходства (если None, используется значение из инициализации)
            return_top_k: Количество лучших ответов для возврата
        
        Returns:
            Tuple[str, float]: Ответ и оценка сходства
        """
        if not query or not query.strip():
            return "❌ Вопрос не может быть пустым.", 0.0

        threshold = threshold if threshold is not None else self.threshold

        if method == "tfidf":
            similarities = self._search_tfidf(query)
        elif method == "embedding":
            similarities = self._search_embedding(query)
        elif method == "hybrid":
            similarities = self._search_hybrid(query)
        else:
            raise ValueError(f"Неизвестный метод: {method}. Доступны: 'tfidf', 'embedding', 'hybrid'")

        # Находим лучший результат
        best_indices = np.argsort(similarities)[::-1][:return_top_k]
        best_scores = similarities[best_indices]

        # Логируем запрос
        query_short = query[:50] + "..." if len(query) > 50 else query
        status = "✅ Найден" if best_scores[0] >= threshold else "❌ Не найден"
        logger.info(f"📝 Запрос: '{query_short}' | Метод: {method} | Сходство: {best_scores[0]:.3f} | {status}")

        if best_scores[0] < threshold:
            return "❌ Извините, я не нашел подходящего ответа на ваш вопрос.", best_scores[0]

        if return_top_k == 1:
            answer_short = self.answers[best_indices[0]][:100] + "..." if len(self.answers[best_indices[0]]) > 100 else self.answers[best_indices[0]]
            logger.info(f"   Ответ: {answer_short}")
            return self.answers[best_indices[0]], best_scores[0]
        else:
            # Для нескольких ответов возвращаем строку с ними
            result = []
            for idx, score in zip(best_indices, best_scores):
                result.append(f"{self.answers[idx]} (сходство: {score:.3f})")
            return "\n".join(result), best_scores[0]

    def _search_tfidf(self, query: str) -> np.ndarray:
        """Поиск с использованием TF-IDF."""
        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return similarities

    def _search_embedding(self, query: str) -> np.ndarray:
        """Поиск с использованием эмбеддингов."""
        self._ensure_model_loaded()
        query_emb = self.embedding_model.encode(query, convert_to_numpy=True)
        similarities = cosine_similarity([query_emb], self.question_embeddings).flatten()
        return similarities

    def _search_hybrid(self, query: str) -> np.ndarray:
        """Гибридный поиск (TF-IDF + эмбеддинги)."""
        # TF-IDF часть
        tfidf_sim = self._search_tfidf(query)

        # Эмбеддинг часть
        emb_sim = self._search_embedding(query)

        # Нормализация
        tfidf_norm = self._normalize(tfidf_sim)
        emb_norm = self._normalize(emb_sim)

        # Комбинация
        similarities = (
            self.hybrid_weight_tfidf * tfidf_norm +
            self.hybrid_weight_embedding * emb_norm
        )

        return similarities

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику системы.
        
        Returns:
            Dict[str, Any]: Словарь со статистикой
        """
        cache_exists = self._get_cache_path().exists()

        return {
            "num_questions": len(self.questions),
            "num_answers": len(self.answers),
            "model_name": self.model_name,
            "embeddings_shape": self.question_embeddings.shape,
            "cache_exists": cache_exists,
            "cache_path": str(self._get_cache_path()),
            "tfidf_shape": self.tfidf_matrix.shape,
            "hybrid_weights": {
                "tfidf": self.hybrid_weight_tfidf,
                "embedding": self.hybrid_weight_embedding,
            },
            "threshold": self.threshold,
        }

    def add_question_answer(self, question: str, answer: str) -> None:
        """
        Добавляет новый вопрос и ответ в систему.
        
        Args:
            question: Новый вопрос
            answer: Ответ на вопрос
        """
        # Добавляем в списки
        self.questions.append(question)
        self.answers.append(answer)

        # Пересчитываем TF-IDF
        logger.info("Пересчет TF-IDF...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.questions)

        # Пересчитываем эмбеддинги
        logger.info("Пересчет эмбеддингов...")
        self._ensure_model_loaded()
        new_emb = self.embedding_model.encode([question], convert_to_numpy=True)
        self.question_embeddings = np.vstack([self.question_embeddings, new_emb])

        # Обновляем кэш
        cache_path = self._get_cache_path()
        with open(cache_path, "wb") as f:
            pickle.dump(self.question_embeddings, f)

        logger.info(f"Добавлен новый вопрос: {question[:50]}...")
