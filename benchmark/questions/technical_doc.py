from benchmark.schemas import BechmarkQuestion

questions = [
    {
        "question": "What is the primary aim of the Children’s Climate Risk Index (CCRI)?",
        "answer": "The CCRI aims to rank countries where vulnerable children are also exposed to a wide range of climate and environmental hazards, providing a comprehensive assessment of how these hazards intersect with children’s existing vulnerabilities.",
    },
    {
        "question": "What is the foundational risk framework adopted by CCRI 2025?",
        "answer": "CCRI 2025 adopted the hazard x exposure x vulnerability as the foundational risk framework, based on IPCC (2020).",
    },
    {
        "question": "According to the CCRI 2025 framework, what does 'Exposure' refer to?",
        "answer": "In the CCRI framework, 'Exposure' refers to the presence of children that could be adversely affected by hazards.",
    },
    {
        "question": "What are the two main pillars that structure the CCRI 2025 analysis?",
        "answer": "The two main pillars are Pillar 1, focusing on children’s exposure to climate and climate-sensitive hazards, and Pillar 2, considering the inherent vulnerabilities of children.",
    },
    {
        "question": "How does the focus on hazard types differ between CCRI 2025 and CCRI 2021?",
        "answer": "CCRI 2025 focuses more specifically on climate and climate-sensitive hazards, whereas CCRI 2021 included a broader mix of climate and environmental hazards.",
    },
    {
        "question": "What significant geographic group is included in CCRI 2025 that was omitted in the 2021 version?",
        "answer": "CCRI 2025 includes the Small Island Developing States (SIDS), which were omitted from CCRI 2021 due to previous data limitations.",
    },
    {
        "question": "Does CCRI 2025 quantify the specific impact of human-induced climate change on children?",
        "answer": "No, CCRI 2025 does not quantify the impact of human-induced climate change separately from naturally occurring extreme weather events; it identifies children's exposure to all hazards.",
    },
    {
        "question": "Are future projections of climate risks included in the CCRI 2025 analysis?",
        "answer": "No, CCRI 2025 does not include analysis on future projections of climate risks on children due to uncertainty in available projections and child population data, and lack of reliable future child vulnerability estimates.",
    },
    {
        "question": "List three criteria used for selecting input data for the CCRI 2025.",
        "answer": "The criteria for input data selection include being Open source, Global, Quantifiable, Comparable, Spatially distributed, and Child centric. (Any three of these).",
    },
    {
        "question": "What specific data source is used for estimating the global child population in CCRI 2025?",
        "answer": "WorldPop’s global gridded children population estimate (Under 18) for 2024 at 100m spatial resolution was used.",
    },
    {
        "question": "What data source is utilized for the Fluvial Flood hazard analysis in CCRI 2025?",
        "answer": "JRC’s gridded Global Flood Hazard model (Baugh, 2024) at 100-year return period and 90m spatial resolution.",
    },
    {
        "question": "What water depth threshold is considered for riverine flood exposure analysis?",
        "answer": "A water depth above 1cm (pixel value greater than or equal to 1 in the processing pipeline) is considered for riverine flood exposure analysis.",
    },
    {
        "question": "How is Agricultural Drought defined within the CCRI 2025 document?",
        "answer": "Agricultural drought is defined as occurring when there is insufficient soil moisture to meet the needs of a particular crop at a particular time.",
    },
    {
        "question": "What index is used as a proxy to estimate children's exposure to agricultural drought?",
        "answer": "The Agricultural Stress Index (ASI) is used as a proxy for agricultural drought exposure.",
    },
    {
        "question": "What wind speed threshold is used to classify a tropical storm in the CCRI 2025 methodology?",
        "answer": "A wind speed greater than 63 km/hr is considered as a named tropical storm, following WMO guidance.",
    },
]

benchmark_questions: list[BechmarkQuestion] = [
    BechmarkQuestion(**question, response_type="textual") for question in questions
]
