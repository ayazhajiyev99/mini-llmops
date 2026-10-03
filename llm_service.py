import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,)


# ============================================================
# 1. .env faylını oxuyuruq
# ============================================================

load_dotenv()

# OpenAI API key-i .env-dən götürürük.
# API key kodun içində yazılmır.
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Əsas model
PRIMARY_MODEL = os.getenv(
    "PRIMARY_MODEL",
    "gpt-4.1-mini"
)

# Əsas model işləməsə istifadə ediləcək model
FALLBACK_MODEL = os.getenv(
    "FALLBACK_MODEL",
    "gpt-4.1-nano"
)


# ============================================================
# 2. OpenAI client yaradırıq
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY,

    # Model maksimum 15 saniyə cavab gözləyəcək.
    # 15 saniyədən çox çəkərsə timeout olacaq.
    timeout=15.0,

    # SDK özü retry etməsin.
    # Retry-i biz özümüz idarə edəcəyik.
    max_retries=0,
)


# ============================================================
# 3. Sadə cache
# ============================================================

CACHE = {}

# ============================================================
# 4. Model qiymətləri
# ============================================================

PRICES = {
    "gpt-4.1": {
        "in": 2.00,
        "out": 8.00,
    },

    "gpt-4.1-mini": {
        "in": 0.40,
        "out": 1.60,
    },

    "gpt-4.1-nano": {
        "in": 0.10,
        "out": 0.40,
    },
}


# ============================================================
# 5. Xərc hesablama funksiyası
# ============================================================

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    LLM sorğusunun neçə dollara başa gəldiyini hesablayır.

    Məsələn:

    1000 input token
    200 output token

    verilsə, uyğun modelin qiymətinə əsasən
    dollarla xərc hesablanır.
    """

    # İstifadə etdiyimiz modelin qiymətini tapırıq.
    price = PRICES[model]

    # Dərsdəki formula:
    #
    # input xərc +
    # output xərc
    #
    cost = (
        input_tokens / 1_000_000 * price["in"]
        +
        output_tokens / 1_000_000 * price["out"]
    )

    return cost


# ============================================================
# 6. Xərci CSV faylına yazmaq
# ============================================================

def save_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
):
    """
    Uğurlu LLM çağırışının xərcini costs.csv faylına yazır.
    """

    file_name = "costs.csv"

    # Fayl mövcud deyil?
    # Onda əvvəlcə başlıqları yazacağıq.
    file_exists = os.path.exists(file_name)

    with open(
        file_name,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        # Əgər bu ilk yazıdırsa, sütun adlarını yazırıq.
        if not file_exists:
            writer.writerow([
                "tarix",
                "model",
                "input_tokens",
                "output_tokens",
                "cost_usd",
            ])

        # Sonra real məlumatı yazırıq.
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model,
            input_tokens,
            output_tokens,
            f"{cost:.8f}",
        ])


# ============================================================
# 7. Retry mexanizmi
# ============================================================

@retry(
    # Maksimum 3 cəhd.
    stop=stop_after_attempt(3),

    # 1, 2, 4 saniyə kimi artan gözləmə.
    # Üstünə kiçik random vaxt əlavə olunur.
    wait=wait_random_exponential(
        min=1,
        max=10,
    ),

    # Yalnız bu xətalarda retry et.
    retry=retry_if_exception_type(
        (
            RateLimitError,
            APITimeoutError,
        )
    ),

    # 3 cəhdin hamısı uğursuz olsa,
    # sonuncu xətanı yuxarı göndər.
    reraise=True,
)
def call_model(
    question: str,
    model: str,
    llm_call=None,
):
    """
    Modelə sual göndərir.

    llm_call test zamanı fake LLM göndərmək
    üçün istifadə olunur.
    """

    # Əgər test zamanı fake LLM verilibsə,
    # real OpenAI API çağırmırıq.
    if llm_call is not None:
        return llm_call(question, model)

    # Real OpenAI çağırışı.
    response = client.chat.completions.create(
        model=model,

        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],

        # Model maksimum 200 token cavab yaratsın.
        max_tokens=200,
    )

    return response


# ============================================================
# 8. Əsas ASK funksiyası
# ============================================================

def ask(
    question: str,
    llm_call=None,
) -> str:
    """
    İstifadəçidən sual alır və cavab qaytarır.

    İş prinsipi:

    1. Cache yoxla
    2. Əsas modeli çağır
    3. Alınmasa fallback modeli çağır
    4. Cavab uğurludursa xərc hesabla
    5. Xərci CSV-yə yaz
    6. Cavabı cache-ə yaz
    7. Cavabı qaytar
    """

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    # Sual əvvəldən verilib?
    if question in CACHE:

        print("CACHE-dən cavab qaytarılır.")

        # API-yə getmirik.
        # Buna görə yeni token xərci də yoxdur.
        return CACHE[question]


    # --------------------------------------------------------
    # PRIMARY MODEL
    # --------------------------------------------------------

    try:

        print(
            f"Əsas model istifadə olunur: {PRIMARY_MODEL}"
        )

        response = call_model(
            question,
            PRIMARY_MODEL,
            llm_call,
        )

        model_used = PRIMARY_MODEL

    except Exception as primary_error:

        print(
            f"Əsas model uğursuz oldu: {primary_error}"
        )

        print(
            f"Fallback model istifadə olunur: {FALLBACK_MODEL}"
        )

        # ----------------------------------------------------
        # FALLBACK MODEL
        # ----------------------------------------------------

        try:

            response = call_model(
                question,
                FALLBACK_MODEL,
                llm_call,
            )

            model_used = FALLBACK_MODEL

        except Exception as fallback_error:

            print(
                f"Fallback model də uğursuz oldu: "
                f"{fallback_error}"
            )

            # Hər iki model işləmirsə,
            # istifadəçiyə texniki error göstərmirik.
            return (
                "Hazırda sorğunuzu emal edə bilmirik. "
                "Zəhmət olmasa bir neçə dəqiqə sonra "
                "yenidən cəhd edin."
            )


    # --------------------------------------------------------
    # CAVABI GÖTÜRÜRÜK
    # --------------------------------------------------------

    answer = response.choices[0].message.content


    # --------------------------------------------------------
    # TOKENLƏR
    # --------------------------------------------------------

    input_tokens = response.usage.prompt_tokens

    output_tokens = response.usage.completion_tokens


    # --------------------------------------------------------
    # XƏRC
    # --------------------------------------------------------

    cost = calc_cost(
        model_used,
        input_tokens,
        output_tokens,
    )


    # --------------------------------------------------------
    # CSV-YƏ YAZIRIQ
    # --------------------------------------------------------

    save_cost(
        model_used,
        input_tokens,
        output_tokens,
        cost,
    )


    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    # Cavabı yadda saxlayırıq.
    #
    # Eyni sual yenidən gələndə
    # API çağırılmayacaq.
    CACHE[question] = answer


    print(
        f"Model: {model_used}"
    )

    print(
        f"Input token: {input_tokens}"
    )

    print(
        f"Output token: {output_tokens}"
    )

    print(
        f"Xərc: ${cost:.8f}"
    )


    return answer

