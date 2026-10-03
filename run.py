from llm_service import ask


# 1-ci sual
answer1 = ask("Python nədir?")
print("CAVAB:", answer1)

print("-" * 50)


# 2-ci sual
answer2 = ask("API nədir?")
print("CAVAB:", answer2)

print("-" * 50)


# 3-cü sual
answer3 = ask("Cache nədir?")
print("CAVAB:", answer3)

print("-" * 50)


# 4-cü sual
answer4 = ask("LLMOps nədir?")
print("CAVAB:", answer4)

print("-" * 50)


# 5-ci sual
# Bu sual birinci sualla eynidir.
# Ona görə artıq OpenAI API-yə getməməlidir.
# Cache-dən cavab gəlməlidir.

answer5 = ask("Python nədir?")
print("CAVAB:", answer5)