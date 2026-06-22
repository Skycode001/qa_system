"""Тесты для Q&A системы."""


import pytest

from src.qa_system import HybridQASystem


def test_initialization():
    """Тест инициализации системы."""
    questions = ["Как дела?", "Что нового?"]
    answers = ["Хорошо!", "Ничего"]

    qa = HybridQASystem(questions, answers)
    assert len(qa.questions) == 2
    assert len(qa.answers) == 2
    assert qa.question_embeddings.shape == (2, 384)  # Размерность модели


def test_find_answer():
    """Тест поиска ответа."""
    questions = ["Как оплатить заказ?", "Где мой заказ?"]
    answers = ["Оплатить можно картой", "Заказ в пути"]

    qa = HybridQASystem(questions, answers)
    answer, score = qa.find_answer("Как я могу оплатить?", method="hybrid")

    assert answer == "Оплатить можно картой"
    assert score > 0


def test_threshold():
    """Тест порога сходства."""
    questions = ["Как оплатить заказ?", "Где мой заказ?"]
    answers = ["Оплатить можно картой", "Заказ в пути"]

    qa = HybridQASystem(questions, answers, threshold=0.9)
    answer, score = qa.find_answer("Привет мир")

    assert "не нашел" in answer


def test_stats():
    """Тест статистики."""
    questions = ["Вопрос 1", "Вопрос 2"]
    answers = ["Ответ 1", "Ответ 2"]

    qa = HybridQASystem(questions, answers)
    stats = qa.get_stats()

    assert stats["num_questions"] == 2
    assert "model_name" in stats


def test_empty_query():
    """Тест пустого запроса."""
    questions = ["Вопрос 1"]
    answers = ["Ответ 1"]

    qa = HybridQASystem(questions, answers)
    answer, score = qa.find_answer("")

    assert "не может быть пустым" in answer


def test_add_question():
    """Тест добавления нового вопроса."""
    questions = ["Вопрос 1"]
    answers = ["Ответ 1"]

    qa = HybridQASystem(questions, answers)
    qa.add_question_answer("Новый вопрос", "Новый ответ")

    assert len(qa.questions) == 2
    assert qa.question_embeddings.shape == (2, 384)


if __name__ == "__main__":
    pytest.main([__file__])
