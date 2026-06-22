"""Точка входа в приложение."""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from .config import DATA_FILE_PATH, LOG_DIR, LOG_FORMAT, LOG_LEVEL
from .data_loader import load_data
from .qa_system import HybridQASystem
from .utils import format_score, setup_logging

# Загружаем .env файл из корня проекта
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, dir_okay=False),
    default=str(DATA_FILE_PATH),
    help="Путь к файлу с данными",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["tfidf", "embedding", "hybrid"]),
    default=os.getenv("METHOD", "hybrid"),
    help="Метод поиска",
)
@click.option(
    "--question-col",
    "-q",
    default=os.getenv("QUESTION_COL", "вопрос"),
    help="Название колонки с вопросами",
)
@click.option(
    "--answer-col",
    "-a",
    default=os.getenv("ANSWER_COL", "ответ"),
    help="Название колонки с ответами",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=float(os.getenv("SIMILARITY_THRESHOLD", 0.3)),
    help="Порог сходства (0-1)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=os.getenv("NO_CACHE", "false").lower() == "true",
    help="Не использовать кэш эмбеддингов",
)
def main(
    data: str,
    method: str,
    question_col: str,
    answer_col: str,
    threshold: float,
    no_cache: bool,
) -> None:
    """
    Интерактивная система поиска ответов на вопросы.
    """
    # Настройка логирования (только в файл)
    setup_logging(LOG_DIR, LOG_LEVEL, LOG_FORMAT)
    
    # Отключаем вывод логов в консоль для библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    logger.info("=" * 50)
    logger.info("Запуск Q&A системы")
    logger.info(f"Файл данных: {data}")
    logger.info(f"Метод: {method}")
    logger.info(f"Порог: {threshold}")
    logger.info("=" * 50)

    try:
        # Загрузка данных
        data_path = Path(data)
        questions, answers = load_data(data_path, question_col, answer_col)

        if no_cache:
            # Если нужно очистить кэш
            import shutil

            from .config import CACHE_DIR
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                logger.info("Кэш очищен")

        # Создание системы
        logger.info("Инициализация системы...")
        qa_system = HybridQASystem(questions, answers, threshold=threshold)

        # Вывод статистики в лог
        stats = qa_system.get_stats()
        logger.info("Статистика системы:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Интерактивный режим (только в консоль, без логов)
        print("\n" + "=" * 60)
        print("🤖 Добро пожаловать в Q&A систему!")
        print(f"📚 В базе {len(questions)} вопросов")
        print(f"🔍 Метод поиска: {method}")
        print(f"🎯 Порог сходства: {threshold}")
        print("=" * 60)
        print("Введите 'exit', 'quit' или 'выход' для выхода")
        print("Введите 'stats' для показа статистики")
        print("Введите 'method <tfidf|embedding|hybrid>' для смены метода")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n❓ Ваш вопрос: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "выход"]:
                    logger.info("👋 Пользователь завершил сессию")
                    print("👋 До свидания!")
                    break

                if user_input.lower() == "stats":
                    stats = qa_system.get_stats()
                    print("\n📊 Статистика:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                if user_input.lower().startswith("method "):
                    new_method = user_input.split(" ")[1].lower()
                    if new_method in ["tfidf", "embedding", "hybrid"]:
                        method = new_method
                        logger.info(f"Метод изменен на: {method}")
                        print(f"✅ Метод изменен на: {method}")
                    else:
                        print(f"❌ Неизвестный метод: {new_method}")
                    continue

                # Поиск ответа
                answer, score = qa_system.find_answer(user_input, method=method)
                print(f"\n💡 Ответ (сходство: {format_score(score)}):")
                print(f"   {answer}")

            except KeyboardInterrupt:
                logger.info("👋 Пользователь прервал сессию (Ctrl+C)")
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                logger.error(f"Ошибка: {e}", exc_info=True)
                print(f"❌ Произошла ошибка: {e}")

    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()