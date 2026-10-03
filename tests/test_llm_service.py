from llm_service import (
    ask,
    calc_cost,
    CACHE,)
import pytest


# ============================================================
# TEST 1
# Cache işləyirmi?
# ============================================================

def test_cache_calls_llm_only_once():

    # Əvvəlki testlərdən qalan cache-i təmizləyirik.
    CACHE.clear()

    # Fake LLM-in neçə dəfə çağırıldığını saymaq üçün
    # dəyişən yaradırıq.
    calls = []

    # Bu real OpenAI deyil!
    # Sadəcə test üçün özümüz yaratdığımız funksiyadır.
    def fake_llm(question, model):

        # Hər çağırışda siyahıya bir element əlavə edirik.
        calls.append(question)

        # OpenAI-dan gəlmiş kimi görünən
        # fake response yaradırıq.

        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = 20

        class FakeMessage:
            content = "Bu fake cavabdır."

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        return FakeResponse()


    # Eyni sualı birinci dəfə göndəririk.
    first_answer = ask(
        "Python nədir?",
        llm_call=fake_llm,
    )


    # Eyni sualı ikinci dəfə göndəririk.
    second_answer = ask(
        "Python nədir?",
        llm_call=fake_llm,
    )


    # Hər iki cavab eyni olmalıdır.
    assert first_answer == second_answer


    # Ən vacib hissə:
    #
    # LLM yalnız 1 dəfə çağırılmalıdır.
    #
    # Çünki ikinci dəfə cache-dən cavab gəlməlidir.
    assert len(calls) == 1


# ============================================================
# TEST 2
# Xərc hesablanması düzgündürmü?
# ============================================================

def test_calc_cost():

    # gpt-4.1-mini üçün dərsdə:
    #
    # input = $0.40 / 1M token
    # output = $1.60 / 1M token

    cost = calc_cost(
        "gpt-4.1-mini",
        1000,
        500,
    )

    # Hesab:
    #
    # 1000 / 1,000,000 * 0.40
    # +
    # 500 / 1,000,000 * 1.60
    #
    # = 0.0004 + 0.0008
    #
    # = 0.0012

    assert cost == pytest.approx(0.0012)