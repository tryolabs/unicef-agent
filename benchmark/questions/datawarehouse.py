from benchmark.schemas import BechmarkQuestion

# DataWarehouse Questions
questions_data = [
    {
        "question": "Whats the Vast Majority Income Ratio in Angola?",
        "answer": 0.643125,
    },
    {
        "question": "How many infant deaths were there in Norway in 2021?",
        "answer": 105,
    },
    {
        "question": "How many children under 5 years died in Iceland in 2015?",
        "answer": 1,
    },
    {
        "question": "What was the completion rate for children of primary school age in 2015 in Rwanda?",
        "answer": 54.299999,
    },
    {
        "question": "What was the estimated number of female children aged 0 to 14 years living with HIV in Central African Republic in 2018?",
        "answer": 2100,
    },
    {
        "question": "What was the percentage of children under 5 years old with weight-for-age <-2 SD (underweight) in Namibia in 2000?",
        "answer": 29.4,
    },
    {
        "question": "What's the percentage of births without a birth weight registered in Nigeria?",
        "answer": 77,
    },
    {
        "question": "What was the percentage of children vaccinated for tuberculosis in Ethiopia in 2020?",
        "answer": 70,
    },
]

benchmark_questions: list[BechmarkQuestion] = [
    BechmarkQuestion(
        question=str(question_data["question"]),
        answer=question_data["answer"],
        response_type="numerical",
    )
    for question_data in questions_data
]
